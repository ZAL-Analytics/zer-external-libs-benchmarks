"""Shared splink helper functions used by dedupe, link_only, and link_and_dedupe benchmarks."""

from pathlib import Path

try:
    import jellyfish
except ImportError:
    jellyfish = None


def load_toml(first_dataset_path):
    """Load mapping.toml from parent.parent of the dataset. Returns {} if missing/unreadable."""
    toml_path = Path(first_dataset_path).parent.parent / "mapping.toml"
    if not toml_path.exists():
        return {}
    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return {}
        with open(toml_path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def build_from_mapping(toml_data, dfs, link_type="dedupe_only"):
    """Build splink comparisons and blocking rules from mapping.toml annotations.

    Mutates dfs in-place to add pre-computed blocking key columns.
    Returns (comparisons, blocking_rules, em_col, surname_col) or (None,)*4 when
    the TOML has no comparison_type annotations (falls back to heuristics).
    em_col is the firstname-role field used for EM training.
    For link modes, adds direct surname/DOB blocking rules to improve blocking recall.
    """
    mappings = toml_data.get("field_mappings", [])
    if not mappings or not any(m.get("comparison_type") for m in mappings):
        return None, None, None, None

    import splink.comparison_library as cl
    from splink import block_on

    surname_col   = next((m["a_field"] for m in mappings if m.get("role") == "surname"), None)
    firstname_col = next((m["a_field"] for m in mappings if m.get("role") == "firstname"), None)
    dob_col       = next((m["a_field"] for m in mappings if m.get("role") == "dob"), None)

    for df in dfs:
        for m in mappings:
            bk  = m.get("blocking_key")
            col = m.get("a_field", "")
            if bk == "soundex_initial_year":
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
            elif bk == "year_month":
                if col in df.columns:
                    df["_bk_dob_ym"] = df[col].str[:7].fillna("")

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
    for m in mappings:
        bk = m.get("blocking_key")
        if bk == "soundex_initial_year" and "_bk_soundex_init_year" in ref_df.columns:
            blocking_rules.append(block_on("_bk_soundex_init_year"))
        elif bk == "year_month" and "_bk_dob_ym" in ref_df.columns:
            blocking_rules.append(block_on("_bk_dob_ym"))

    if not blocking_rules and mappings:
        first_col = mappings[0].get("a_field")
        if first_col and first_col in ref_df.columns:
            blocking_rules = [block_on(first_col)]

    if link_type in ("link_only", "link_and_dedupe"):
        if surname_col and surname_col in ref_df.columns:
            blocking_rules.append(block_on(surname_col))
        if dob_col and dob_col in ref_df.columns:
            blocking_rules.append(block_on(dob_col))

    return comparisons, blocking_rules, firstname_col, surname_col


def add_blocking_keys(df, name_cols, date_cols, addr_cols, id_cols):
    """Pre-compute zer-equivalent blocking key columns (heuristic fallback)."""
    surname_col = name_cols[-1] if name_cols else None
    first_col   = name_cols[0]  if name_cols else None
    dob_col     = date_cols[0]  if date_cols else None
    addr_col    = addr_cols[0]  if addr_cols else None

    if jellyfish and surname_col and dob_col:
        df["_bk_name_dob"] = (
            df[surname_col].apply(lambda x: jellyfish.soundex(x) if x else "")
            + "_" + df[dob_col].str[:4]
        )
    if dob_col:
        df["_bk_dob_ym"] = df[dob_col].str[:7]
    if addr_col and first_col:
        df["_bk_addr_init"] = (
            df[addr_col].str.split().str[0].str[:1].fillna("")
            + "_" + df[first_col].str[:1].fillna("")
        )
    for id_col in id_cols[:2]:
        safe = id_col.replace(" ", "_")
        df[f"_bk_id4_{safe}"] = df[id_col].str[-4:].fillna("")
    return df
