#!/usr/bin/env python3
"""splink link-and-dedupe benchmark.

Finds duplicates within each source AND matches across sources simultaneously.

Usage:
    python run.py --dataset <csv_a> --dataset <csv_b>
                  [--ground-truth <csv_path>] [--scenario <slug>] --out <dir>

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
        print("[splink/link_and_dedupe] need at least two --dataset arguments", file=sys.stderr)
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
        print(f"[splink/link_and_dedupe] import error: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()

    dfs   = [pd.read_csv(p, dtype=str).fillna("") for p in datasets]
    names = [Path(p).stem for p in datasets]

    _orig_id = dfs[0].columns[0]
    if _orig_id != "unique_id":
        dfs = [d.rename(columns={_orig_id: "unique_id"}) for d in dfs]

    comparisons, blocking_rules, em_col, surname_col, renames = strategy_for(dataset_name)(
        dfs, "link_and_dedupe")
    if renames:
        dfs = [dfs[0]] + [df.rename(columns=renames) for df in dfs[1:]]

    if comparisons is None:
        comparisons, blocking_rules, em_col, surname_col = build_fallback_comparisons(dfs)

    run_splink_benchmark(
        dfs=dfs, names=names,
        link_type="link_and_dedupe",
        comparisons=comparisons, blocking_rules=blocking_rules,
        em_col=em_col, surname_col=surname_col,
        ground_truth_path=args.ground_truth,
        scenario=args.scenario, dataset_name=dataset_name,
        out_dir=out_dir, t0=t0,
    )


if __name__ == "__main__":
    main()
