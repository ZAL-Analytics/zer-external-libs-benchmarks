"""Splink accuracy strategy for brp/dedupe (and brp/link, brp/link_and_dedupe, kvk/dedupe).

BRP records use full first names (no initials).  Two blocking rules:
- soundex_initial_year: surname phonetic + first_initial + birth_year (precise AND-style)
- year_month: birth year-month (looser OR-style, safe with full first names)
"""

import sys
from pathlib import Path

_SPLINK_DIR = Path(__file__).resolve().parents[1]
if str(_SPLINK_DIR) not in sys.path:
    sys.path.insert(0, str(_SPLINK_DIR))

try:
    import jellyfish
except ImportError:
    jellyfish = None

_SURNAME_COL       = "achternaam"
_FIRSTNAME_COL     = "voornamen"
_DOB_COL           = "geboortedatum"
_POSTCODE_COL      = "postcode"
_WOONPLAATS_COL    = "woonplaats"
_GEBOORTELAND_COL  = "geboorteland"
_NATIONALITEIT_COL = "nationaliteit"


def build(dfs, link_type="dedupe_only"):
    """Return (comparisons, blocking_rules, em_col, surname_col, renames)."""
    import splink.comparison_library as cl
    from splink import block_on

    ref_df = dfs[0]

    for df in dfs:
        if all(c in df.columns for c in [_SURNAME_COL, _FIRSTNAME_COL, _DOB_COL]):
            if jellyfish:
                df["_bk_soundex_init_year"] = (
                    df[_SURNAME_COL].apply(lambda x: jellyfish.soundex(x) if x else "")
                    + ":" + df[_FIRSTNAME_COL].str[:1].fillna("")
                    + ":" + df[_DOB_COL].str[:4].fillna("")
                )
            else:
                df["_bk_soundex_init_year"] = (
                    df[_SURNAME_COL].str[:4].fillna("")
                    + ":" + df[_FIRSTNAME_COL].str[:1].fillna("")
                    + ":" + df[_DOB_COL].str[:4].fillna("")
                )
        if _DOB_COL in df.columns:
            df["_bk_dob_ym"] = df[_DOB_COL].str[:7].fillna("")

    comparisons = []
    for col in [_FIRSTNAME_COL, _SURNAME_COL]:
        if col in ref_df.columns:
            try:
                comparisons.append(cl.JaroWinklerAtThresholds(col, [0.88, 0.7]))
            except AttributeError:
                comparisons.append(cl.LevenshteinAtThresholds(col, [1, 2, 3]))
    for col in [_DOB_COL, _POSTCODE_COL, _WOONPLAATS_COL, _GEBOORTELAND_COL, _NATIONALITEIT_COL]:
        if col in ref_df.columns:
            comparisons.append(cl.ExactMatch(col))

    if not comparisons:
        return None, None, None, None, {}

    blocking_rules = []
    if "_bk_soundex_init_year" in ref_df.columns:
        blocking_rules.append(block_on("_bk_soundex_init_year"))
    if "_bk_dob_ym" in ref_df.columns:
        blocking_rules.append(block_on("_bk_dob_ym"))
    if not blocking_rules:
        blocking_rules = [block_on(_FIRSTNAME_COL)] if _FIRSTNAME_COL in ref_df.columns else []

    em_col      = _FIRSTNAME_COL if _FIRSTNAME_COL in ref_df.columns else None
    surname_col = _SURNAME_COL   if _SURNAME_COL   in ref_df.columns else None

    return comparisons, blocking_rules, em_col, surname_col, {}
