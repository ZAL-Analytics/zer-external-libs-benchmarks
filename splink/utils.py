"""Shared splink helper functions used by dedupe, link_only, and link_and_dedupe benchmarks."""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import jellyfish
except ImportError:
    jellyfish = None


def add_blocking_keys(df, name_cols, date_cols, addr_cols, id_cols):
    """Pre-compute zer-equivalent blocking key columns (heuristic fallback)."""
    surname_col = name_cols[-1] if name_cols else None
    first_col   = name_cols[0]  if name_cols else None
    dob_col     = date_cols[0]  if date_cols else None
    addr_col    = addr_cols[0]  if addr_cols else None

    if jellyfish and surname_col and dob_col:
        df["_bk_name_dob"] = (
            df[surname_col].apply(lambda x: jellyfish.soundex(x) if x else "")
            + "_" + df[dob_col].str[:4]
        )
    if dob_col:
        df["_bk_dob_ym"] = df[dob_col].str[:7]
    if addr_col and first_col:
        df["_bk_addr_init"] = (
            df[addr_col].str.split().str[0].str[:1].fillna("")
            + "_" + df[first_col].str[:1].fillna("")
        )
    for id_col in id_cols[:2]:
        safe = id_col.replace(" ", "_")
        df[f"_bk_id4_{safe}"] = df[id_col].str[-4:].fillna("")
    return df


def build_fallback_comparisons(dfs):
    """Heuristic comparison + blocking rules when no strategy is defined for this scenario."""
    import splink.comparison_library as cl
    from splink import block_on

    ref_df = dfs[0]
    _ADDR_KEYS = ("straat", "adres", "street", "address", "city", "place", "woon")
    name_cols = [c for c in ref_df.columns if any(k in c.lower() for k in ("naam", "name", "nomen", "alias"))
                 and not any(k in c.lower() for k in _ADDR_KEYS)]
    date_cols = [c for c in ref_df.columns if any(k in c.lower() for k in ("datum", "date", "dob", "birth"))]
    addr_cols = [c for c in ref_df.columns if any(k in c.lower() for k in _ADDR_KEYS)]
    id_cols   = [c for c in ref_df.columns if any(k in c.lower() for k in ("id", "nummer", "bsn", "number", "code", "postcode"))]

    for i, df in enumerate(dfs):
        dfs[i] = add_blocking_keys(df, name_cols, date_cols, addr_cols, id_cols)
    ref_df = dfs[0]

    comparisons = []
    for c in name_cols[:2]:
        comparisons.append(cl.LevenshteinAtThresholds(c, [1, 2, 3]))
    for c in date_cols[:1]:
        comparisons.append(cl.ExactMatch(c))
    for c in addr_cols[:1]:
        comparisons.append(cl.LevenshteinAtThresholds(c, [2, 4]))
    if not comparisons:
        for c in list(ref_df.select_dtypes("object").columns)[:4]:
            comparisons.append(cl.ExactMatch(c))

    bk_cols = [c for c in ref_df.columns if c.startswith("_bk_")]
    blocking_rules = [block_on(c) for c in bk_cols] or [block_on(ref_df.columns[0])]
    em_col      = name_cols[0]  if name_cols else None
    surname_col = name_cols[-1] if name_cols else None
    return comparisons, blocking_rules, em_col, surname_col


def write_summary_csv(out_dir, run_id, mode_label, dataset_name, ts_iso,
                      total_records, candidate_pairs, elapsed_ms,
                      tp_n, fp_n, fn_n, prec, rec, f1_v):
    """Write the zer-compare-compatible summary CSV."""
    csv_path = out_dir / f"{run_id}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["library", "mode", "dataset", "run_id", "timestamp",
                    "total_records", "candidate_pairs", "auto_matched", "borderline", "auto_rejected",
                    "elapsed_ms", "true_pos", "false_pos", "false_neg", "precision", "recall", "f1"])
        w.writerow(["splink", mode_label, dataset_name, run_id, ts_iso,
                    total_records, candidate_pairs, 0, 0, 0,
                    elapsed_ms,
                    tp_n if tp_n is not None else "",
                    fp_n if fp_n is not None else "",
                    fn_n if fn_n is not None else "",
                    f"{prec:.3f}" if prec is not None else "",
                    f"{rec:.3f}"  if rec  is not None else "",
                    f"{f1_v:.3f}" if f1_v is not None else ""])
    return csv_path


def run_splink_benchmark(dfs, names, link_type, comparisons, blocking_rules,
                         em_col, surname_col, ground_truth_path, scenario,
                         dataset_name, out_dir, t0):
    """Full splink run: train → predict → evaluate → write JSON + CSV."""
    try:
        from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    except ImportError as e:
        import sys
        print(f"[splink] import error: {e}", file=sys.stderr)
        sys.exit(1)

    # Imported lazily so sys.path is already configured by the time this is called.
    from bench_metrics import (
        avg_precision, best_threshold_metrics, blocking_recall,
        write_scored_pairs_csv, norm_id,
    )

    ref_df = dfs[0]
    id_col = "unique_id"

    # splink UNION ALL requires identical column sets across all frames.
    if len(dfs) > 1:
        all_cols = list(dict.fromkeys(c for df in dfs for c in df.columns))
        for df in dfs:
            for c in all_cols:
                if c not in df.columns:
                    df[c] = None

    settings = SettingsCreator(
        link_type=link_type,
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
    )

    if len(dfs) == 1:
        linker = Linker(dfs[0], settings, db_api=DuckDBAPI())
    else:
        linker = Linker(dfs, settings, db_api=DuckDBAPI(), input_table_aliases=names)

    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
    if em_col and em_col in ref_df.columns:
        linker.training.estimate_parameters_using_expectation_maximisation(block_on(em_col))
    if surname_col and surname_col in ref_df.columns and surname_col != em_col:
        linker.training.estimate_parameters_using_expectation_maximisation(block_on(surname_col))

    all_pairs  = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    total_records   = sum(len(df) for df in dfs)
    candidate_pairs = len(all_pairs)

    ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    prefix_map = {
        "dedupe_only":     "splink_dedup",
        "link_only":       "splink_link_only",
        "link_and_dedupe": "splink_link_and_dedupe",
    }
    mode_label_map = {
        "dedupe_only":     "deduplicate",
        "link_only":       "link-only",
        "link_and_dedupe": "link-and-dedupe",
    }
    run_id_prefix = prefix_map.get(link_type, "splink")
    mode_label    = mode_label_map.get(link_type, link_type)
    run_id        = f"{run_id_prefix}_{dataset_name}_{ts}"
    csv_path_name = f"{run_id}_summary.csv"
    print_prefix  = f"splink/{link_type.replace('_', '-').replace('--', '-')}"

    tp_n = fp_n = fn_n = None
    prec = rec = f1_v = opt_thr = pr_auc = blk_rec = scored_pairs_csv = None

    if ground_truth_path:
        gt_pairs = set()
        with open(ground_truth_path) as f:
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

        f1_v, prec, rec, opt_thr, tp_n, fp_n, fn_n = best_threshold_metrics(
            labels, scores, n_total_pos=len(gt_pairs))
        pr_auc  = avg_precision(labels, scores, n_total_pos=len(gt_pairs))
        blk_rec = blocking_recall(all_pair_ids, gt_pairs)
        scored_pairs_csv = f"{run_id}_scored_pairs.csv"
        write_scored_pairs_csv(out_dir / scored_pairs_csv, scores, labels)

    record = {
        "run_id":         run_id,
        "library":        "splink",
        "scenario":       scenario,
        "mode":           mode_label,
        "dataset":        dataset_name,
        "target":         "cpu",
        "timestamp_unix": int(datetime.now(timezone.utc).timestamp()),
        "files": {
            "summary_csv":      csv_path_name,
            "pairs_ndjson":     None,
            "strat_csv":        None,
            "scored_pairs_csv": scored_pairs_csv,
        },
        "metrics": {
            "total_records":     total_records,
            "candidate_pairs":   candidate_pairs,
            "elapsed_ms":        elapsed_ms,
            "precision":         round(prec,  3) if prec  is not None else None,
            "recall":            round(rec,   3) if rec   is not None else None,
            "f1":                round(f1_v,  3) if f1_v  is not None else None,
            "optimal_threshold": opt_thr,
            "pr_auc":            pr_auc,
            "blocking_recall":   blk_rec,
            "true_pos":          tp_n,
            "false_pos":         fp_n,
            "false_neg":         fn_n,
        },
        "strat": [],
        "scored_pairs": None,
    }
    json_path = out_dir / f"{run_id}_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    write_summary_csv(out_dir, run_id, mode_label, dataset_name, ts_iso,
                      total_records, candidate_pairs, elapsed_ms,
                      tp_n, fp_n, fn_n, prec, rec, f1_v)

    print(f"[{print_prefix}] written: {json_path}")
    print(f"[{print_prefix}] records={total_records}  pairs={candidate_pairs}  elapsed_ms={elapsed_ms}")
    if prec is not None:
        print(f"[{print_prefix}] precision={prec:.3f}  recall={rec:.3f}  f1={f1_v:.3f}"
              f"  opt_thr={opt_thr}  pr_auc={pr_auc}  blocking_recall={blk_rec}")
