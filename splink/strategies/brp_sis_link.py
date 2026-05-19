"""Splink accuracy strategy for brp_sis/link.

Same root cause as brp_hks/link: SIS records contain ~17 % initials in
voornamen, and the year_month blocking key generates too many false candidate
pairs.  See brp_hks_link.py for a full analysis.

Fix: use only soundex_initial_year blocking (drop year_month).
"""

from .brp_hks_link import build

__all__ = ["build"]
