"""Splink accuracy strategy for brp_sis/link — delegates to brp_hks_link.

SIS records contain ~17% initials in voornamen — same root cause as HKS.
See brp_hks_link.py for a full explanation.
"""

from .brp_hks_link import build

__all__ = ["build"]
