# splink benchmark

[splink](https://github.com/moj-analytical-services/splink) is a Python probabilistic record linkage library
built on DuckDB. It uses the Fellegi-Sunter model with EM estimation, the same statistical approach as zer.

## Supported modes

| Mode | Script | Description |
|---|---|---|
| `dedupe` | `dedupe/run.py` | Single-source deduplication |
| `link-only` | `link_only/run.py` | Two-source linkage (pass `--dataset` twice) |
| `link-and-dedupe` | `link_and_dedupe/run.py` | Deduplicate within each source and link across sources simultaneously |
| `throughput` | `throughput/run.py` | Stage-level latency and memory profiling; no ground-truth evaluation |

## Installation

### Ubuntu 22.04 / 24.04

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### RHEL 10 / AlmaLinux 10 / Rocky Linux 10

```bash
sudo dnf install -y python3 python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `splink>=4.0.0`, `pandas>=2.0.0`, and `jellyfish>=0.9`. DuckDB is pulled in
automatically as a transitive dependency of splink. `jellyfish` is used for Soundex-based blocking key
generation in the `strategies/` module.

## Running via zer-bench (recommended)

`zer-bench library` runs `setup.sh` once (idempotent via a `.setup_done` sentinel) and then invokes the
correct `run.py` for the requested mode.

When `--dataset` is omitted, the dataset and ground-truth CSV are auto-discovered from
`data/benchmarks/<preset>/` in the workspace root (`brp_500k` by default).

```bash
# Dedup, uses data/benchmarks/brp_small/ automatically
cargo run --release -p zer-bench -- \
    library --library splink --mode dedupe --out bench_results/

# Dedup, different preset
cargo run --release -p zer-bench -- \
    library --library splink --mode dedupe --preset sis --out bench_results/

# Dedup, explicit dataset and ground-truth override
cargo run --release -p zer-bench -- \
    library --library splink --mode dedupe \
            --dataset data/benchmarks/brp_small/brp_persons.csv \
            --ground-truth data/benchmarks/brp_small/ground_truth_pairs.csv \
            --out bench_results/

# Throughput benchmark
cargo run --release -p zer-bench -- \
    library --library splink --mode throughput --out bench_results/

# Run all configured libraries for the dedupe mode (same preset defaulting applies)
cargo run --release -p zer-bench -- \
    library-all --mode dedupe --out bench_results/
```

## Running manually

All scripts accept an optional `--scenario <slug>` argument. When supplied it:
- selects the named entry from `strategies/` instead of auto-detecting from the file path
- uses the slug as `dataset_name` in output file names and JSON metadata

When `--scenario` is omitted, the script derives the name from the path as
`<dataset>_<mode>` (e.g. `data/benchmarks/brp/dedupe/source.csv` → `brp_dedupe`).

```bash
cd benchmarks/splink
source .venv/bin/activate   # if using a venv

# Dedup
python3 dedupe/run.py \
    --dataset data/benchmarks/brp/dedupe/source.csv \
    --ground-truth data/benchmarks/brp/dedupe/ground_truth.csv \
    --out bench_results/

# Dedup with explicit scenario (overrides path-based naming)
python3 dedupe/run.py \
    --dataset data/benchmarks/brp/dedupe/source.csv \
    --ground-truth data/benchmarks/brp/dedupe/ground_truth.csv \
    --scenario brp_dedupe \
    --out bench_results/

# Link-only (two CSV files, pass --dataset twice)
python3 link_only/run.py \
    --dataset data/benchmarks/brp/link/source_a.csv \
    --dataset data/benchmarks/brp/link/source_b.csv \
    --ground-truth data/benchmarks/brp/link/ground_truth.csv \
    --out bench_results/

# Link-and-dedupe (two CSV files)
python3 link_and_dedupe/run.py \
    --dataset data/benchmarks/brp/link_and_dedupe/source_a.csv \
    --dataset data/benchmarks/brp/link_and_dedupe/source_b.csv \
    --ground-truth data/benchmarks/brp/link_and_dedupe/ground_truth.csv \
    --out bench_results/

# Throughput (single dataset, capped at --max-records; default 50 000)
python3 throughput/run.py \
    --dataset data/benchmarks/brp/dedupe/source.csv \
    --max-records 50000 \
    --out bench_results/
```

## Input CSV format

Each `run.py` resolves comparisons and blocking rules in two steps:

1. **Strategy module (preferred)**: `strategies.strategy_for(dataset_name)` returns the registered
   `build()` function for the scenario. Each strategy module explicitly defines comparisons and
   blocking rules for its schema (e.g. `brp_dedupe.py` uses JaroWinkler on BRP name fields with
   soundex + year-month blocking keys).
2. **Heuristic fallback**: when the strategy returns `(None, …)` (i.e. no matching slug and
   `default.py` is used), comparisons and blocking rules are built automatically from column names
   via `utils.build_fallback_comparisons()`:

| Field kind | Keywords matched (case-insensitive) |
|---|---|
| Name (Jaro-Winkler / Levenshtein) | `naam`, `name`, `nomen`, `alias` |
| Date (exact match) | `datum`, `date`, `dob`, `birth` |
| Address (Levenshtein) | `straat`, `adres`, `street`, `address`, `city`, `place`, `woon` |
| Identifier (blocking) | `id`, `nummer`, `bsn`, `number`, `code`, `postcode` |

The first column is treated as the record ID (renamed to `unique_id` internally).

## Output

### Accuracy modes (dedupe, link-only, link-and-dedupe)

Each run writes up to three files under `<out>/`:

| File | Description |
|---|---|
| `<run_id>_benchmark.json` | zer-compatible metadata record |
| `<run_id>_summary.csv` | single-row CSV in the shared cross-library format (consumed by `zer-bench compare`) |
| `<run_id>_scored_pairs.csv` | scored pairs sorted by score descending; columns `score` (float), `is_match` (0/1). Written only when `--ground-truth` is supplied. |

The `_summary.csv` columns are:

```
library, mode, dataset, run_id, timestamp,
total_records, candidate_pairs, auto_matched, borderline, auto_rejected,
elapsed_ms, true_pos, false_pos, false_neg, precision, recall, f1
```

Accuracy columns are populated only when `--ground-truth` is supplied (auto-supplied when using a preset).
The ground-truth CSV must have columns `record_id_a`, `record_id_b`, `is_match` (values: `true`/`1`/`yes`).

The `_benchmark.json` `metrics` object contains:

```
total_records, candidate_pairs, elapsed_ms,
precision, recall, f1,
optimal_threshold,   # score threshold that maximises F1
pr_auc,              # area under the Precision–Recall curve
blocking_recall,     # fraction of GT pairs that appear in the candidate set
true_pos, false_pos, false_neg
```

The `files` object contains a `"scored_pairs_csv"` key pointing to the sidecar CSV (or `null` when no
ground truth was provided). The `"scored_pairs"` key in the JSON is always `null`.

### Throughput mode

The throughput `_benchmark.json` uses a different schema focused on stage timing and memory:

```
library, mode, dataset, run_id, timestamp, backend,
total_records, candidate_pairs,
setup_ms,            # u-sampling time,no zer equivalent, excluded from pipeline total
pipeline: {
    compare_ms,      # blocking + compare + score combined (single DuckDB pass)
    em_ms,           # EM parameter estimation only
    total_ms,        # em_ms + compare_ms (u-sampling excluded for fair comparison)
},
memory_mb: { peak_mb },
throughput: { pairs_per_s },
match_bands: { auto_matched, borderline, auto_rejected },
raw: { stages, throughput, memory_mb }   # per-stage breakdowns for detailed analysis
```

## Strategies

Per-scenario splink configurations live in `strategies/`. Each module exposes a `build()` function:

```python
def build(dfs, link_type) -> (comparisons, blocking_rules, em_col, surname_col, renames)
```

The `strategies/__init__.py` registry maps scenario slugs to their `build` function. When no slug matches,
`strategies/default.py` is used, which returns `(None, None, None, None, {})` to trigger the heuristic
fallback in the calling `run.py`.

Currently registered scenarios:

| Slug | Module |
|---|---|
| `brp_dedupe`, `micro_brp_dedupe` | `brp_dedupe.py` |
| `brp_link`, `micro_brp_link` | `brp_link.py` |
| `brp_link_and_dedupe`, `micro_brp_link_and_dedupe` | `brp_link_and_dedupe.py` |
| `brp_kvk_link` | `brp_kvk_link.py` |
| `brp_hks_link` | `brp_hks_link.py` |
| `brp_sis_link`, `micro_brp_sis_link` | `brp_sis_link.py` |
| `brp_kvk_hks_link_and_dedupe` | `brp_kvk_hks_link_and_dedupe.py` |
| `kvk_dedupe` | `kvk_dedupe.py` |

To add a new strategy: create `strategies/<slug>.py` with a `build()` function and register it in
`strategies/__init__.py`.

## Shared modules

Helper code is shared across all `run.py` scripts via three modules:

| Module | Location | Contents |
|---|---|---|
| `bench_metrics` | `benchmarks/utils/bench_metrics.py` | `norm_id`, `avg_precision`, `best_threshold_metrics`, `blocking_recall`, `write_scored_pairs_csv`, `load_scored_pairs_csv` |
| `utils` | `benchmarks/splink/utils.py` | `add_blocking_keys`, `build_fallback_comparisons`, `write_summary_csv`, `run_splink_benchmark` |
| `strategies` | `benchmarks/splink/strategies/` | Per-scenario `build()` functions; see [strategies/README.md](strategies/README.md) |

Both `bench_metrics` and `utils` are inserted into `sys.path` at import time using
`Path(__file__).resolve().parents[N]` so they can be used without installing a package.
