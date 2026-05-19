"""Splink accuracy strategy for brp_kvk/link.

BRP (Dutch civil registry) × KvK (Dutch commercial registry — company contacts).

Root cause of default poor performance: the default build_from_mapping uses both a
soundex_initial_year blocking key AND a year_month DOB key.  The year_month key generates
enormous numbers of false candidate pairs (all records sharing a birth year-month).  KvK
records additionally may carry abbreviated voornamen (e.g. "J.H." instead of "Jan Hendrik"),
weakening Jaro-Winkler discrimination on first names and compounding EM overfit.

Fix: drop the year_month key; instead add postcode exact-match blocking.  Dutch postcodes
are street-level (4 digits + 2 letters ≈ 15-30 households), so two records sharing a
postcode are almost certainly at the same address.  Both BRP and KvK carry postcode, which
makes this a uniquely strong OR-rule for this scenario that is not available for BRP×HKS.

The soundex_initial_year key is kept as the primary blocking rule; postcode is the catch-all
for pairs where surname Soundex or first-initial differ due to name abbreviation in KvK.

Link-mode fallbacks (exact achternaam, exact geboortedatum) from the default strategy are
intentionally omitted — exact achternaam is covered by soundex_initial_year at lower noise,
and exact geboortedatum alone produces too many false candidates to be useful here.
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


def build(toml_data, dfs, link_type="dedupe_only"):
    """Build splink comparisons and blocking rules for brp_kvk/link."""
    mappings = toml_data.get("field_mappings", [])
    if not mappings or not any(m.get("comparison_type") for m in mappings):
        return None, None, None, None

    import splink.comparison_library as cl
    from splink import block_on

    surname_col   = next((m["a_field"] for m in mappings if m.get("role") == "surname"), None)
    firstname_col = next((m["a_field"] for m in mappings if m.get("role") == "firstname"), None)
    dob_col       = next((m["a_field"] for m in mappings if m.get("role") == "dob"), None)
    postcode_col  = next((m["a_field"] for m in mappings if m.get("a_field") == "postcode"), None)

    for df in dfs:
        if surname_col and firstname_col and dob_col:
            if all(c in df.columns for c in [surname_col, firstname_col, dob_col]):
                if jellyfish:
                    df["_bk_soundex_init_year"] = (
                        df[surname_col].apply(lambda x: jellyfish.soundex(x) if x else "")
                        + ":" + df[firstname_col].str[:1].fillna("")
                        + ":" + df[dob_col].str[:4].fillna("")
                    )
                else:
                    df["_bk_soundex_init_year"] = (
                        df[surname_col].str[:4].fillna("")
                        + ":" + df[firstname_col].str[:1].fillna("")
                        + ":" + df[dob_col].str[:4].fillna("")
                    )

    ref_df = dfs[0]
    comparisons = []
    for m in mappings:
        col   = m.get("a_field", "")
        ctype = m.get("comparison_type", "exact")
        if col not in ref_df.columns:
            continue
        if ctype == "jaro_winkler":
            try:
                comparisons.append(cl.JaroWinklerAtThresholds(col, [0.88, 0.7]))
            except AttributeError:
                comparisons.append(cl.LevenshteinAtThresholds(col, [1, 2, 3]))
        elif ctype == "date":
            comparisons.append(cl.ExactMatch(col))
        else:
            comparisons.append(cl.ExactMatch(col))

    blocking_rules = []
    if "_bk_soundex_init_year" in ref_df.columns:
        blocking_rules.append(block_on("_bk_soundex_init_year"))
    if postcode_col and postcode_col in ref_df.columns:
        blocking_rules.append(block_on(postcode_col))

    if not blocking_rules and mappings:
        first_col = mappings[0].get("a_field")
        if first_col and first_col in ref_df.columns:
            blocking_rules = [block_on(first_col)]

    return comparisons, blocking_rules, firstname_col, surname_col
