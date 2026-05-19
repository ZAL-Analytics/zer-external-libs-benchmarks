"""Splink accuracy strategy for brp_kvk_hks/link_and_dedupe.

Three-source scenario: BRP × KvK × HKS.  HKS records contain ~11 % initials
in voornamen — same root cause as brp_hks/link.  The default year_month
blocking key generates too many false candidate pairs, causing EM to
miscalibrate.

Fix: use only soundex_initial_year blocking (drop year_month).
See brp_hks_link.py for a full explanation.
"""

from .brp_hks_link import build

__all__ = ["build"]
