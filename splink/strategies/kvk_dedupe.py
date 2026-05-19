"""Splink accuracy strategy for kvk/dedupe.

KvK company-contact records include postcode, woonplaats, and business
identifiers — no initials issue.  Default blocking is well-calibrated.
Starting point: delegates to default.  Add overrides here as needed.
"""

from .default import build

__all__ = ["build"]
