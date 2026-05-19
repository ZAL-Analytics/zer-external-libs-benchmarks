#!/usr/bin/env python3
"""splink link-only benchmark (two sources).

Usage:
    python run.py --dataset <csv_a> --dataset <csv_b>
                  [--ground-truth <csv_path>] [--scenario <slug>] --out <dir>

The first --dataset is source A; the second is source B.
Writes *_benchmark.json (zer-compatible schema) and *_summary.csv (compare.rs compat).
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SPLINK_DIR = Path(__file__).resolve().parents[1]
_UTILS_DIR  = _SPLINK_DIR.parent / "utils"
for _p in [str(_SPLINK_DIR), str(_UTILS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_metrics import (
    avg_precision, best_threshold_metrics, blocking_recall,
    write_scored_pairs_csv, norm_id,
)
from utils import load_toml, add_blocking_keys
from strategies import strategy_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      action="append", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--scenario",     default=None)
    parser.add_argument("--out",          default="bench_results")
    args = parser.parse_args()

    datasets = args.dataset
    if len(datasets) < 2:
        print("[splink/link_only] need two --dataset arguments", file=sys.stderr)
        sys.exit(1)

    if args.scenario:
        dataset_name = args.scenario.replace("/", "_")
    else:
        p = Path(datasets[0])
        dataset_name = f"{p.parent.parent.name}_{p.parent.name}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        from splink import DuckDBAPI, Linker, SettingsCreator, block_on
        import splink.comparison_library as cl
    except ImportError as e:
        print(f"[splink/link_only] import error: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()

    toml_data = load_toml(datasets[0])

    # Build renames from TOML (b_field → a_field for cross-schema alignment)
    renames = {m["b_field"]: m["a_field"]
               for m in toml_data.get("field_mappings", [])
               if m.get("a_field") and m.get("b_field") and m["a_field"] != m["b_field"]}

    df_a = pd.read_csv(datasets[0], dtype=str).fillna("")
    df_b = pd.read_csv(datasets[1], dtype=str).fillna("")
    if renames:
        df_b = df_b.rename(columns=renames)
    _orig_id = df_a.columns[0]
    if _orig_id != "unique_id":
        df_a = df_a.rename(columns={_orig_id: "unique_id"})
        df_b = df_b.rename(columns={_orig_id: "unique_id"})

    build = strategy_for(dataset_name)
    comparisons, blocking_rules, em_col, surname_col = build(
        toml_data, [df_a, df_b], link_type="link_only")

    if comparisons is None:
        _ADDR_KEYS = ("straat", "adres", "street", "address", "city", "place", "woon")
        name_cols = [c for c in df_a.columns if any(k in c.lower() for k in ("naam", "name", "nomen", "alias"))
                     and not any(k in c.lower() for k in _ADDR_KEYS)]
        date_cols = [c for c in df_a.columns if any(k in c.lower() for k in ("datum", "date", "dob", "birth"))]
        addr_cols = [c for c in df_a.columns if any(k in c.lower() for k in _ADDR_KEYS)]
        id_cols   = [c for c in df_a.columns if any(k in c.lower() for k in ("id", "nummer", "bsn", "number", "code", "postcode"))]
        df_a = add_blocking_keys(df_a, name_cols, date_cols, addr_cols, id_cols)
        df_b = add_blocking_keys(df_b, name_cols, date_cols, addr_cols, id_cols)
        comparisons = []
        for c in name_cols[:2]:
            comparisons.append(cl.LevenshteinAtThresholds(c, [1, 2, 3]))
        for c in date_cols[:1]:
            comparisons.append(cl.ExactMatch(c))
        for c in addr_cols[:1]:
            comparisons.append(cl.LevenshteinAtThresholds(c, [2, 4]))
        if not comparisons:
            for c in list(df_a.select_dtypes("object").columns)[:4]:
                comparisons.append(cl.ExactMatch(c))
        bk_cols = [c for c in df_a.columns if c.startswith("_bk_")]
        blocking_rules = [block_on(c) for c in bk_cols] or [block_on(df_a.columns[0])]
        em_col      = name_cols[0]  if name_cols else None
        surname_col = name_cols[-1] if name_cols else None

    settings = SettingsCreator(
        link_type="link_only",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
    )

    # splink UNION ALL requires both frames to share identical columns.
    # Use None (SQL NULL) so splink's null_level fires for structurally absent
    # fields — empty string "" is a non-NULL value that triggers a ~0 similarity
    # comparison, unfairly penalising cross-source true matches.
    for c in df_a.columns:
        if c not in df_b.columns:
            df_b[c] = None
    for c in df_b.columns:
        if c not in df_a.columns:
            df_a[c] = None

    linker = Linker([df_a, df_b], settings, db_api=DuckDBAPI(),
                    input_table_aliases=["source_a", "source_b"])
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
    if em_col and em_col in df_a.columns:
        linker.training.estimate_parameters_using_expectation_maximisation(block_on(em_col))
    if surname_col and surname_col in df_a.columns and surname_col != em_col:
        linker.training.estimate_parameters_using_expectation_maximisation(block_on(surname_col))

    # Get all candidate pairs with scores (threshold=0.0); use ≥0.5 for P/R/F1
    all_pairs    = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
    elapsed_ms   = int((time.monotonic() - t0) * 1000)

    total_records   = len(df_a) + len(df_b)
    candidate_pairs = len(all_pairs)

    id_col = "unique_id"

    ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"splink_link_only_{dataset_name}_{ts}"
    ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    csv_path = f"{run_id}_summary.csv"

    tp_n = fp_n = fn_n = None
    prec = rec = f1_v = opt_thr = pr_auc = blk_rec = scored_pairs_csv = None

    if args.ground_truth:
        gt_pairs = set()
        with open(args.ground_truth) as f:
            for row in csv.DictReader(f):
                if str(row.get("is_match", "")).lower() in ("true", "1", "yes"):
                    a = norm_id(row["record_id_a"]); b = norm_id(row["record_id_b"])
                    gt_pairs.add((min(a, b), max(a, b)))

        labels = []; scores = []
        all_pair_ids = []
        for _, row in all_pairs.iterrows():
            a = norm_id(row[f"{id_col}_l"])
            b = norm_id(row[f"{id_col}_r"])
            key = (min(a, b), max(a, b))
            score = float(row["match_probability"])
            labels.append(1 if key in gt_pairs else 0)
            scores.append(score)
            all_pair_ids.append((a, b))

        f1_v, prec, rec, opt_thr, tp_n, fp_n, fn_n = best_threshold_metrics(labels, scores, n_total_pos=len(gt_pairs))
        pr_auc  = avg_precision(labels, scores, n_total_pos=len(gt_pairs))
        blk_rec = blocking_recall(all_pair_ids, gt_pairs)
        scored_pairs_csv = f"{run_id}_scored_pairs.csv"
        write_scored_pairs_csv(out_dir / scored_pairs_csv, scores, labels)

    # ── *_benchmark.json (zer-compatible schema) ──────────────────────────────
    record = {
        "run_id":         run_id,
        "library":        "splink",
        "scenario":       args.scenario,
        "mode":           "link-only",
        "dataset":        dataset_name,
        "target":         "cpu",
        "timestamp_unix": int(datetime.now(timezone.utc).timestamp()),
        "files": {
            "summary_csv":      csv_path,
            "pairs_ndjson":     None,
            "strat_csv":        None,
            "scored_pairs_csv": scored_pairs_csv,
        },
        "metrics": {
            "total_records":   total_records,
            "candidate_pairs": candidate_pairs,
            "elapsed_ms":      elapsed_ms,
            "precision":          round(prec,   3) if prec   is not None else None,
            "recall":             round(rec,    3) if rec    is not None else None,
            "f1":                 round(f1_v,   3) if f1_v   is not None else None,
            "optimal_threshold":  opt_thr,
            "pr_auc":             pr_auc,
            "blocking_recall":    blk_rec,
            "true_pos":        tp_n,
            "false_pos":       fp_n,
            "false_neg":       fn_n,
        },
        "strat": [],
        "scored_pairs": None,
    }
    json_path = out_dir / f"{run_id}_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    # ── *_summary.csv (compare.rs compat) ────────────────────────────────────
    with open(out_dir / csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["library","mode","dataset","run_id","timestamp",
                    "total_records","candidate_pairs","auto_matched","borderline","auto_rejected",
                    "elapsed_ms","true_pos","false_pos","false_neg","precision","recall","f1"])
        w.writerow(["splink","link-only",dataset_name,run_id,ts_iso,
                    total_records,candidate_pairs,0,0,0,
                    elapsed_ms,
                    tp_n if tp_n is not None else "",
                    fp_n if fp_n is not None else "",
                    fn_n if fn_n is not None else "",
                    f"{prec:.3f}" if prec is not None else "",
                    f"{rec:.3f}"  if rec  is not None else "",
                    f"{f1_v:.3f}" if f1_v is not None else ""])

    print(f"[splink/link_only] written: {json_path}")
    print(f"[splink/link_only] records={total_records}  pairs={candidate_pairs}  elapsed_ms={elapsed_ms}")
    if prec is not None:
        print(f"[splink/link_only] precision={prec:.3f}  recall={rec:.3f}  f1={f1_v:.3f}  opt_thr={opt_thr}  pr_auc={pr_auc}  blocking_recall={blk_rec}")


if __name__ == "__main__":
    main()
