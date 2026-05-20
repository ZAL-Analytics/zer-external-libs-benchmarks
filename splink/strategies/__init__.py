"""Per-scenario splink accuracy strategies.

Each module exposes a ``build(dfs, link_type)`` function that returns
``(comparisons, blocking_rules, em_col, surname_col, renames)``.
If comparisons is None, the caller falls back to build_fallback_comparisons().

To add a new strategy:
1. Create ``strategies/<dataset_name>.py`` with a ``build()`` function.
2. Add a match arm in ``strategy_for`` below.
"""

from . import default                    as _default
from . import brp_dedupe                 as _brp_dedupe
from . import brp_link                   as _brp_link
from . import brp_link_and_dedupe        as _brp_link_and_dedupe
from . import brp_kvk_link               as _brp_kvk_link
from . import brp_hks_link               as _brp_hks_link
from . import brp_sis_link               as _brp_sis_link
from . import brp_kvk_hks_link_and_dedupe as _brp_kvk_hks_link_and_dedupe
from . import kvk_dedupe                 as _kvk_dedupe


def strategy_for(dataset_name: str):
    """Return the build function for ``dataset_name``.

    The returned callable has signature
    ``build(dfs, link_type) -> (comparisons, blocking_rules, em_col, surname_col, renames)``.
    """
    strategies = {
        "brp_dedupe":                  _brp_dedupe.build,
        "micro_brp_dedupe":            _brp_dedupe.build,

        "brp_link":                    _brp_link.build,
        "micro_brp_link":              _brp_link.build,

        "brp_link_and_dedupe":         _brp_link_and_dedupe.build,
        "micro_brp_link_and_dedupe":   _brp_link_and_dedupe.build,

        "brp_kvk_link":                _brp_kvk_link.build,

        "brp_hks_link":                _brp_hks_link.build,

        "brp_sis_link":                _brp_sis_link.build,
        "micro_brp_sis_link":          _brp_sis_link.build,

        "brp_kvk_hks_link_and_dedupe": _brp_kvk_hks_link_and_dedupe.build,

        "kvk_dedupe":                  _kvk_dedupe.build,
    }
    return strategies.get(dataset_name, _default.build)
