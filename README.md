# benchmarks

External library benchmarks for comparing zer against other record linkage tools.

## Structure

```
benchmarks/
├── splink/          # splink benchmark (Python, DuckDB-backed)
│   ├── dedupe/      # single-source deduplication
│   ├── link_only/   # two-source link-only
│   ├── link_and_dedupe/  # simultaneous dedup + link
│   ├── throughput/  # stage-level latency and memory profiling
│   ├── strategies/  # per-scenario splink configurations
│   └── utils.py     # TOML-driven comparison/blocking builder
└── utils/           # shared Python utilities
    ├── bench_metrics.py   # accuracy metrics (precision, recall, F1, PR-AUC, blocking recall)
    └── plot_results.py    # plot *_benchmark.json output files
```

## Libraries

| Library | Location | Modes |
|---|---|---|
| [splink](https://github.com/moj-analytical-services/splink) | `splink/` | `dedupe`, `link-only`, `link-and-dedupe`, `throughput` |

## Shared utilities

All benchmark scripts share the same output schema and metric functions from `utils/`:

- **`bench_metrics.py`** — `best_threshold_metrics`, `avg_precision`, `blocking_recall`,
  `write_scored_pairs_csv`, `load_scored_pairs_csv`, `norm_id`
- **`plot_results.py`** — renders accuracy and throughput comparison plots from `*_benchmark.json` files

See [utils/README.md](utils/README.md) for full API documentation.

## Output schema

Every benchmark run writes:

| File | Description |
|---|---|
| `<run_id>_benchmark.json` | zer-compatible metadata record with metrics and file references |
| `<run_id>_summary.csv` | single-row CSV consumed by `zer-bench compare` |
| `<run_id>_scored_pairs.csv` | `(score, is_match)` pairs for PR curve plotting (accuracy modes only) |

Throughput runs use a different `_benchmark.json` structure with pipeline stage timings and memory
snapshots instead of accuracy metrics.

## Plotting results

```bash
python benchmarks/utils/plot_results.py --input bench_results/data/<run>/
```

Produces accuracy comparison, PR-AUC, PR curves, stratified recall, and throughput plots.
See [utils/README.md](utils/README.md) for the full list of generated figures.
