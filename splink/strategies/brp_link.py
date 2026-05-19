"""Splink accuracy strategy for brp/link.

Both sources are BRP (full first names, no initials).  The default TOML-driven
blocking and comparisons work well here.
Starting point: delegates to default.  Add overrides here as needed.
"""

from .default import build

__all__ = ["build"]
