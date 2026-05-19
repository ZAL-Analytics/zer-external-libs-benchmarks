"""Default splink accuracy strategy — delegates to the shared utils.build_from_mapping."""

import sys
from pathlib import Path

_SPLINK_DIR = Path(__file__).resolve().parents[1]
if str(_SPLINK_DIR) not in sys.path:
    sys.path.insert(0, str(_SPLINK_DIR))

from utils import build_from_mapping


def build(toml_data, dfs, link_type="dedupe_only"):
    """Return (comparisons, blocking_rules, em_col, surname_col) using the shared TOML-driven logic."""
    return build_from_mapping(toml_data, dfs, link_type)
