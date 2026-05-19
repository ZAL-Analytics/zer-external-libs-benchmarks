# benchmarks/utils

Shared Python utilities used across all external library benchmarks.

## bench_metrics.py

Accuracy metrics shared by every `run.py` script (splink, etc.).

| Function | Signature | Description |
|---|---|---|
| `norm_id(v)` | `str` | Normalise a record ID to string. Handles integer-valued floats (e.g. `"123.0"` → `"123"`) and preserves leading zeros in pure-digit strings. |
| `avg_precision(labels, scores, n_total_pos)` | `float \| None` | Area under the Precision–Recall curve (Average Precision). Pass `n_total_pos=len(gt_pairs)` to include blocking false negatives in the recall denominator. |
| `best_threshold_metrics(labels, scores, n_total_pos)` | `(f1, precision, recall, threshold, tp, fp, fn)` | Sweep all score thresholds and return the one that maximises F1. Pass `n_total_pos` to make recall comparable across systems with different blocking recall. |
| `blocking_recall(candidate_pair_ids, gt_pairs)` | `float \| None` | Fraction of ground-truth pairs that appear in the candidate set; reveals the blocking ceiling before scoring. |
| `write_scored_pairs_csv(path, scores, labels)` | — | Write `(score, is_match)` pairs to CSV, sorted by score descending. No row limit — avoids embedding large arrays in JSON. |
| `load_scored_pairs_csv(path)` | `list[dict]` | Load a `scored_pairs` CSV back as `[{"score": float, "is_match": bool}, ...]`. |

All accuracy metric functions accept `labels` and `scores` as parallel lists.

## plot_results.py

CLI tool that reads `*_benchmark.json` files produced by `zer-bench` or any external library benchmark
and renders comparison plots.

```bash
python benchmarks/utils/plot_results.py --input bench_results/data/<run>/
python benchmarks/utils/plot_results.py --input bench_results/data/<run>/ --output my_plots/
```

`--input` is required. `--output` defaults to `bench_results/plots/<run>/` where `<run>` is the folder
name of the input path. Accepts either a directory (scanned recursively for `*_benchmark.json`) or a
single JSON file.

### Plots produced

Each figure is saved as `.png`, `.svg`, and `.pdf`.

| Figure stem | Condition | Description |
|---|---|---|
| `accuracy_comparison` | any accuracy run present | Per-scenario grid of Precision / Recall / F1 / PR-AUC grouped bars, one colour per library |
| `pr_auc_bars` | any `pr_auc` value present | PR-AUC bar chart per scenario |
| `pr_curves` | `scored_pairs_csv` sidecar present | Per-scenario Precision–Recall curves, one line per library |
| `strat_recall` | `strat` data present in JSON | Recall broken down by match type (dedupe / link / cross\_dedup, etc.) |
| `judge_impact` | `zer+judge_*` runs present | F1 / Recall lift from the neural judge with delta annotations over the baseline zer run |
| `throughput_comparison` | ≥2 throughput runs present | Side-by-side bars: pipeline time, M pairs/s, peak memory |
| `<lib>/stage_pie/stage_pie` | ≥2 pipeline stages in a throughput run | Pie chart of stage durations for one library |
| `<lib>/memory_timeline/memory_timeline` | per-stage RSS readings present | RSS memory (MB) at each pipeline stage boundary over wall time |

### Library colour palette

Libraries are assigned consistent colours so the same library always appears in the same colour across
all plot types:

| Library | Fill | Border |
|---|---|---|
| `zer` | pastel blue | `#1E88E5` |
| `zer+judge_cpu` | pastel blue-indigo | `#1565C0` |
| `zer+judge_cuda` | pastel indigo | `#0D47A1` |
| `splink` | pastel red | `#E53935` |

### PR curve notes

- Pairs with the same score are grouped into a single point to avoid zigzag artifacts.
- When `true_pos + false_neg` is available from the JSON, recall is expressed as a fraction of **all**
  ground-truth pairs (including blocking false negatives), making the curve consistent with the scalar
  recall metric.
- The curve is extended to `(1.0, 0.0)` for every library regardless of blocking quality.

### Throughput schema support

The loader handles both the current common schema (`pipeline`, `memory_mb`, `throughput` top-level keys)
and the older library-specific stage names (`stages.index_ms`, `stages.predict_ms`, etc.) for
backwards compatibility.
