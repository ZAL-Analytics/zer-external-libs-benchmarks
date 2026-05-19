"""Splink accuracy strategy for brp/dedupe.

BRP records use full first names (no initials).  The default TOML-driven
blocking and comparisons are well-calibrated for this scenario.
Starting point: delegates to default.  Add overrides here as needed.
"""

from .default import build

__all__ = ["build"]
