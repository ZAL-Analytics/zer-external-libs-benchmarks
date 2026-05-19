"""Splink accuracy strategy for brp_hks/link (and brp_kvk_hks/link_and_dedupe).

Root cause: the default build_from_mapping adds both a soundex_initial_year
blocking key AND a year_month DOB key.  The year_month key is a loose OR
rule that generates huge numbers of false candidate pairs (all records sharing
a birth year-month).  HKS records contain ~11 % initials in voornamen, so
Jaro-Winkler on first names cannot discriminate "J. Jansen" from any other
"J. Jansen" born in the same month, causing EM to overfit.

Fix: use only the soundex_initial_year key (AND-style: surname phonetic +
first_initial + birth_year).  A full first name's initial always matches an
abbreviated initial, so true-match recall is preserved while same-month false
pairs are eliminated.
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
    """Build splink comparisons and blocking rules, using only soundex_initial_year blocking."""
    mappings = toml_data.get("field_mappings", [])
    if not mappings or not any(m.get("comparison_type") for m in mappings):
        return None, None, None, None

    import splink.comparison_library as cl
    from splink import block_on

    surname_col   = next((m["a_field"] for m in mappings if m.get("role") == "surname"), None)
    firstname_col = next((m["a_field"] for m in mappings if m.get("role") == "firstname"), None)
    dob_col       = next((m["a_field"] for m in mappings if m.get("role") == "dob"), None)

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

    # Only soundex_initial_year — drop year_month to cut down false candidates.
    blocking_rules = []
    if "_bk_soundex_init_year" in ref_df.columns:
        blocking_rules.append(block_on("_bk_soundex_init_year"))

    if not blocking_rules and mappings:
        first_col = mappings[0].get("a_field")
        if first_col and first_col in ref_df.columns:
            blocking_rules = [block_on(first_col)]

    return comparisons, blocking_rules, firstname_col, surname_col
