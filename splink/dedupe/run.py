#!/usr/bin/env python3
"""splink deduplication benchmark.

Usage:
    python run.py --dataset <csv_path> [--ground-truth <csv_path>]
                  [--scenario <slug>] --out <dir>

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
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--scenario",     default=None)
    parser.add_argument("--out",          default="bench_results")
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
    except ImportError as e:
        print(f"[splink/dedupe] import error: {e}", file=sys.stderr)
        print("[splink/dedupe] run: pip install splink pandas", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()

    df = pd.read_csv(dataset_path, dtype=str).fillna("")
    _orig_id = df.columns[0]
    if _orig_id != "unique_id":
        df = df.rename(columns={_orig_id: "unique_id"})

    comparisons, blocking_rules, em_col, surname_col, _renames = strategy_for(dataset_name)([df], "dedupe_only")

    if comparisons is None:
        comparisons, blocking_rules, em_col, surname_col = build_fallback_comparisons([df])

    run_splink_benchmark(
        dfs=[df], names=["source_a"],
        link_type="dedupe_only",
        comparisons=comparisons, blocking_rules=blocking_rules,
        em_col=em_col, surname_col=surname_col,
        ground_truth_path=args.ground_truth,
        scenario=args.scenario, dataset_name=dataset_name,
        out_dir=out_dir, t0=t0,
    )


if __name__ == "__main__":
    main()
