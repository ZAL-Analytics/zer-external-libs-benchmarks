"""Default splink accuracy strategy,triggers fallback heuristic."""


def build(dfs, link_type="dedupe_only"):
    """Return (None, None, None, None, {}) to trigger build_fallback_comparisons."""
    return None, None, None, None, {}
