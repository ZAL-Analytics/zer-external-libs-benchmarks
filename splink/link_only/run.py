#!/usr/bin/env python3
"""splink link-only benchmark (two sources).

Usage:
    python run.py --dataset <csv_a> --dataset <csv_b>
                  [--ground-truth <csv_path>] [--scenario <slug>] --out <dir>

The first --dataset is source A; the second is source B.
Writes *_benchmark.json (zer-compatible schema) and *_summary.csv (compare.rs compat).
"""

import argparse
import sys
import time
from pathlib import Path

_SPLINK_DIR = Path(__file__).resolve().parents[1]
_UTILS_DIR  = _SPLINK_DIR.parent / "utils"
for _p in [str(_SPLINK_DIR), str(_UTILS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import build_fallback_comparisons, run_splink_benchmark
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
    except ImportError as e:
        print(f"[splink/link_only] import error: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()

    df_a = pd.read_csv(datasets[0], dtype=str).fillna("")
    df_b = pd.read_csv(datasets[1], dtype=str).fillna("")

    _orig_id = df_a.columns[0]
    if _orig_id != "unique_id":
        df_a = df_a.rename(columns={_orig_id: "unique_id"})
        df_b = df_b.rename(columns={_orig_id: "unique_id"})

    comparisons, blocking_rules, em_col, surname_col, renames = strategy_for(dataset_name)(
        [df_a, df_b], "link_only")
    if renames:
        df_b = df_b.rename(columns=renames)

    if comparisons is None:
        comparisons, blocking_rules, em_col, surname_col = build_fallback_comparisons([df_a, df_b])

    run_splink_benchmark(
        dfs=[df_a, df_b], names=["source_a", "source_b"],
        link_type="link_only",
        comparisons=comparisons, blocking_rules=blocking_rules,
        em_col=em_col, surname_col=surname_col,
        ground_truth_path=args.ground_truth,
        scenario=args.scenario, dataset_name=dataset_name,
        out_dir=out_dir, t0=t0,
    )


if __name__ == "__main__":
    main()
