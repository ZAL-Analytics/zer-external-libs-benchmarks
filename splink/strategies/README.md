# splink strategies

Per-scenario splink comparison and blocking configurations for the accuracy benchmarks.

## Overview

Each strategy module exposes a single `build()` function:

```python
def build(dfs: list, link_type: str = "dedupe_only") \
    -> tuple[list, list, str | None, str | None, dict]:
    ...
```

Return value: `(comparisons, blocking_rules, em_col, surname_col, renames)`

| Return value | Type | Description |
|---|---|---|
| `comparisons` | `list` | splink comparison objects (e.g. `JaroWinklerAtThresholds`, `ExactMatch`) |
| `blocking_rules` | `list` | splink `block_on(...)` rules |
| `em_col` | `str \| None` | Column used for the first EM training pass (typically firstname) |
| `surname_col` | `str \| None` | Column used for the second EM training pass (typically surname) |
| `renames` | `dict` | Column renames applied to source B before passing frames to splink (used for cross-schema scenarios where field names differ between sources) |

Returning `(None, None, None, None, {})` signals the calling `run.py` to fall back to heuristic
auto-detection via `build_fallback_comparisons()` instead.

## Registered scenarios

The `__init__.py` registry maps scenario slugs to their `build` function. The slug comes from
`--scenario <slug>` on the command line or is auto-derived from the dataset path when `--scenario`
is omitted.

| Slug(s) | Module | Notes |
|---|---|---|
| `brp_dedupe`, `micro_brp_dedupe` | `brp_dedupe.py` | BRP single-source deduplication |
| `brp_link`, `micro_brp_link` | `brp_link.py` | BRP two-source link-only |
| `brp_link_and_dedupe`, `micro_brp_link_and_dedupe` | `brp_link_and_dedupe.py` | BRP link-and-dedupe |
| `brp_kvk_link` | `brp_kvk_link.py` | BRP ↔ KvK link-only |
| `brp_hks_link` | `brp_hks_link.py` | BRP ↔ HKS link-only |
| `brp_sis_link`, `micro_brp_sis_link` | `brp_sis_link.py` | BRP ↔ SIS link-only |
| `brp_kvk_hks_link_and_dedupe` | `brp_kvk_hks_link_and_dedupe.py` | BRP + KvK + HKS link-and-dedupe |
| `kvk_dedupe` | `kvk_dedupe.py` | KvK single-source deduplication |
| _(any other slug)_ | `default.py` | Returns `(None, …)` → triggers heuristic fallback |

## default strategy

`default.py` returns `(None, None, None, None, {})` unconditionally, signalling the calling `run.py`
to fall back to `build_fallback_comparisons()` in `utils.py`. This is the path taken for any scenario
slug that is not explicitly registered in `__init__.py`.

## Blocking key types

Per-scenario strategy modules pre-compute blocking key columns directly on the dataframes. Two
standard blocking key columns are produced by the registered BRP/KvK/HKS/SIS strategies:

| Column | Logic |
|---|---|
| `_bk_soundex_init_year` | `soundex(surname) + ":" + firstname[:1] + ":" + dob[:4]`. Falls back to `surname[:4]` when `jellyfish` is not installed. |
| `_bk_dob_ym` | `dob[:7]` (YYYY-MM) |

The heuristic fallback (`build_fallback_comparisons()` in `utils.py`) generates the same two blocking
key columns via `add_blocking_keys()` when the matching columns are present.

## Adding a new strategy

1. Create `strategies/<slug>.py` with a `build(dfs, link_type)` function.
2. Import it in `strategies/__init__.py` and add a mapping in the `strategies` dict inside
   `strategy_for()`.

Most new scenarios will need their own `build()` that hardcodes the schema's field names and blocking
logic. Use `brp_dedupe.py` as a reference implementation. The minimum viable implementation is:

```python
def build(dfs, link_type="dedupe_only"):
    return None, None, None, None, {}
```

which delegates entirely to `build_fallback_comparisons()`. Add explicit comparisons and blocking
rules when keyword-based heuristic detection is insufficient for the schema.
