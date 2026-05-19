"""Splink accuracy strategy for brp/link_and_dedupe.

Both sources are BRP (full first names, no initials).  Combined link-and-dedupe
mode with default blocking is well-calibrated.
Starting point: delegates to default.  Add overrides here as needed.
"""

from .default import build

__all__ = ["build"]
