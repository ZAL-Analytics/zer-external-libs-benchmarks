"""Splink accuracy strategy for brp_kvk_hks/link_and_dedupe,delegates to brp_hks_link.

Three-source scenario: BRP × KvK × HKS.  HKS records contain ~11% initials in
voornamen,same root cause as brp_hks/link.  See brp_hks_link.py.
"""

from .brp_hks_link import build

__all__ = ["build"]
