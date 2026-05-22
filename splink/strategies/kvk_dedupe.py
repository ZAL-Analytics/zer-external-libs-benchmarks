"""Splink accuracy strategy for kvk/dedupe,delegates to brp_dedupe.

KvK company-contact records use the same BRP field names (voornamen, achternaam,
geboortedatum, postcode) so the BRP blocking/comparison rules apply directly.
"""

from .brp_dedupe import build

__all__ = ["build"]
