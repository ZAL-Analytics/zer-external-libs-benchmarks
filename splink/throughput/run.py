#!/usr/bin/env python3
"""splink throughput benchmark — measures load / train / predict stage latencies.

Usage:
    python run.py --dataset <csv_path> [--scenario <slug>] --out <dir>

Writes *_benchmark.json (zer throughput schema) and *_summary.csv (compare.rs compat).
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

_PROCESS = psutil.Process(os.getpid())


def _read_rss_mb():
    try:
        return round(_PROCESS.memory_info().rss / 1024 / 1024, 1)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--scenario",    default=None)
    parser.add_argument("--out",         default="bench_results")
    parser.add_argument("--max-records", type=int, default=50000)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if args.scenario:
        dataset_name = args.scenario.replace("/", "_")
    else:
        p = dataset_path
        dataset_name = f"{p.parent.parent.name}_{p.parent.name}"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        from splink import DuckDBAPI, Linker, SettingsCreator, block_on
        import splink.comparison_library as cl
    except ImportError as e:
        print(f"[splink/throughput] import error: {e}", file=sys.stderr)
        print("[splink/throughput] run: pip install splink pandas", file=sys.stderr)
        sys.exit(1)

    # ── Stage 1: load ─────────────────────────────────────────────────────────
    t0 = time.monotonic()
    df = pd.read_csv(dataset_path, dtype=str).fillna("")
    _orig_id = df.columns[0]
    if _orig_id != "unique_id":
        df = df.rename(columns={_orig_id: "unique_id"})
    df = df.iloc[:args.max_records]
    load_ms = int((time.monotonic() - t0) * 1000)
    mem_after_load = _read_rss_mb()
    print(f"[splink/throughput] capped to {len(df)} records (--max-records={args.max_records})", file=sys.stderr)

    total_records = len(df)
    _ADDR_KEYS = ("straat", "adres", "street", "address")
    name_cols = [c for c in df.columns if any(k in c.lower() for k in ("naam", "name", "nomen"))
                 and not any(k in c.lower() for k in _ADDR_KEYS)]
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ("datum", "date", "dob", "birth"))]
    addr_cols = [c for c in df.columns if any(k in c.lower() for k in _ADDR_KEYS)]

    comparisons = []
    for c in name_cols[:2]:
        comparisons.append(cl.LevenshteinAtThresholds(c, [1, 2, 3]))
    for c in date_cols[:1]:
        comparisons.append(cl.ExactMatch(c))
    for c in addr_cols[:1]:
        comparisons.append(cl.LevenshteinAtThresholds(c, [2, 4]))
    if not comparisons:
        for c in list(df.select_dtypes("object").columns)[:4]:
            comparisons.append(cl.ExactMatch(c))

    surname_cols = [c for c in name_cols if any(k in c.lower() for k in ("achter", "surname", "last_name", "family"))]
    block_name = surname_cols[0] if surname_cols else (name_cols[0] if name_cols else None)
    blocking_rules = []
    if block_name:
        blocking_rules.append(block_on(block_name))
    if date_cols:
        blocking_rules.append(block_on(date_cols[0]))
    if not blocking_rules:
        blocking_rules = [block_on(df.columns[0])]
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
    )
    linker = Linker(df, settings, db_api=DuckDBAPI())

    # ── Stage 2a: u-sampling (setup, no zer equivalent) ───────────────────────
    t_u0 = time.monotonic()
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
    u_sample_ms = int((time.monotonic() - t_u0) * 1000)

    # ── Stage 2b: EM parameter estimation only ────────────────────────────────
    t1 = time.monotonic()
    for c in name_cols[:1]:
        linker.training.estimate_parameters_using_expectation_maximisation(block_on(c))
    train_ms = int((time.monotonic() - t1) * 1000)
    mem_after_train = _read_rss_mb()

    # ── Stage 3: predict (blocking + scoring, all candidates) ─────────────────
    t2 = time.monotonic()
    all_pairs = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
    predict_ms = int((time.monotonic() - t2) * 1000)
    mem_after_predict = _read_rss_mb()
    mem_peak = max(v for v in (mem_after_load, mem_after_train, mem_after_predict) if v is not None) or None

    total_ms        = load_ms + u_sample_ms + train_ms + predict_ms
    candidate_pairs = len(all_pairs)
    matched         = int((all_pairs["match_probability"] >= 0.5).sum())
    rejected        = candidate_pairs - matched
    pipeline_ms     = train_ms + predict_ms   # u-sampling excluded for fair comparison
    pipeline_pairs_s = int(candidate_pairs * 1000 / pipeline_ms) if pipeline_ms > 0 else 0

    ts     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"splink_throughput_{dataset_name}_{ts}"
    ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    csv_path = str(out_dir.resolve() / f"{run_id}_summary.csv")

    # ── *_benchmark.json (common throughput schema) ───────────────────────────
    # predict_ms captures blocking + compare + score as a single combined DuckDB stage.
    # train_ms = EM parameter estimation only (u-sampling excluded for fair comparison with zer).
    # u_sample_ms is stored separately under setup_ms and raw.stages.
    record = {
        "library":         "splink",
        "mode":            "throughput",
        "dataset":         dataset_name,
        "run_id":          run_id,
        "timestamp":       ts_iso,
        "backend":         "cpu",
        "total_records":   total_records,
        "candidate_pairs": candidate_pairs,
        "setup_ms":        u_sample_ms,     # u-sampling: no zer equivalent, excluded from pipeline
        "pipeline": {
            "block_ms":      None,          # included in predict
            "compare_ms":    predict_ms,    # blocking + compare + score combined
            "em_ms":         train_ms,      # EM parameter estimation only
            "score_ms":      None,          # included in predict
            "judge_ms":      None,
            "u_sample_ms":   u_sample_ms,   # u-sampling (setup, excluded from total_ms)
            "total_ms":      pipeline_ms,   # em_ms + compare_ms only
        },
        "memory_mb": {
            "peak_mb": mem_peak,
        },
        "throughput": {
            "pairs_per_s": pipeline_pairs_s,
        },
        "match_bands": {
            "auto_matched":  matched,
            "borderline":    0,
            "auto_rejected": rejected,
        },
        "lambda_est": None,
        "raw": {
            "stages": {
                "load_ms":      load_ms,
                "u_sample_ms":  u_sample_ms,
                "train_ms":     train_ms,
                "predict_ms":   predict_ms,
                "total_ms":     total_ms,
            },
            "throughput": {
                "predict_pairs_per_s": pipeline_pairs_s,
            },
            "memory_mb": {
                "after_load":    mem_after_load,
                "after_train":   mem_after_train,
                "after_predict": mem_after_predict,
                "peak":          mem_peak,
            },
        },
    }
    json_path = out_dir / f"{run_id}_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    # ── *_summary.csv (compare.rs compat) ────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["library", "mode", "dataset", "run_id", "timestamp",
                    "total_records", "candidate_pairs", "auto_matched", "borderline", "auto_rejected",
                    "elapsed_ms", "true_pos", "false_pos", "false_neg", "precision", "recall", "f1"])
        w.writerow(["splink", "throughput", dataset_name, run_id, ts_iso,
                    total_records, candidate_pairs, matched, 0, rejected,
                    pipeline_ms, "", "", "", "", "", ""])

    print(f"[splink/throughput] written: {json_path}")
    print(f"[splink/throughput] records={total_records}  pairs={candidate_pairs}  total_ms={total_ms}")
    print(f"[splink/throughput] load_ms={load_ms}  u_sample_ms={u_sample_ms}  train_ms={train_ms}  predict_ms={predict_ms}")
    print(f"[splink/throughput] pipeline_ms={pipeline_ms}  pipeline_pairs_per_s={pipeline_pairs_s}")


if __name__ == "__main__":
    main()
