# splink strategies

Per-scenario splink comparison and blocking configurations for the accuracy benchmarks.

## Overview

Each strategy module exposes a single `build()` function:

```python
def build(toml_data: dict, dfs: list, link_type: str = "dedupe_only") \
    -> tuple[list, list, str | None, str | None]:
    ...
```

Return value: `(comparisons, blocking_rules, em_col, surname_col)`

| Return value | Type | Description |
|---|---|---|
| `comparisons` | `list` | splink comparison objects (e.g. `LevenshteinAtThresholds`, `ExactMatch`) |
| `blocking_rules` | `list` | splink `block_on(...)` rules |
| `em_col` | `str \| None` | Column used for the first EM training pass (typically firstname) |
| `surname_col` | `str \| None` | Column used for the second EM training pass (typically surname) |

Returning `(None, None, None, None)` signals the calling `run.py` to fall back to heuristic
auto-detection instead.

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
| _(any other slug)_ | `default.py` | Delegates to `utils.build_from_mapping()` |

## default strategy

`default.py` delegates directly to `utils.build_from_mapping()`, which reads `comparison_type`,
`blocking_key`, and `role` annotations from `mapping.toml` to construct comparisons and blocking rules.
When the TOML has no `comparison_type` annotations, `build_from_mapping()` returns `(None, None, None, None)`
and the calling `run.py` applies keyword-based heuristic detection.

## Blocking key types

The TOML-driven path supports two `blocking_key` values used by `utils.build_from_mapping()`:

| Value | Generated column | Logic |
|---|---|---|
| `soundex_initial_year` | `_bk_soundex_init_year` | `soundex(surname) + ":" + firstname[:1] + ":" + dob[:4]`. Falls back to `surname[:4]` when `jellyfish` is not installed. |
| `year_month` | `_bk_dob_ym` | `dob[:7]` (YYYY-MM) |

For link modes (`link_only`, `link_and_dedupe`), `build_from_mapping()` additionally appends direct
`block_on(surname_col)` and `block_on(dob_col)` rules to improve blocking recall across sources.

## Adding a new strategy

1. Create `strategies/<slug>.py` with a `build(toml_data, dfs, link_type)` function.
2. Import it in `strategies/__init__.py` and add a mapping in the `strategies` dict inside
   `strategy_for()`.

Most scenarios can simply re-export `default.build`:

```python
from .default import build
__all__ = ["build"]
```

Add overrides only when the default TOML-driven logic needs scenario-specific adjustments (e.g.
custom comparison thresholds, additional blocking rules, or field renames not covered by the TOML).
