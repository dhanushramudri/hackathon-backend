import pandas as pd

from app.core.adapter import get_adapter
from app.engines.coe_skill_engine import GENERIC_SKILL_COES
from app.engines.coe_taxonomy import resolve_coe_label

_cache: dict[str, str] | None = None
_cache_fingerprint: tuple | None = None

def _canonicalize(raw_coe: str) -> str:
    # Delegates to coe_taxonomy, which builds its alias map from
    # COE_SKILL_MAP (the same source this function used to read directly) --
    # kept as a thin wrapper here so callers/tests that import _canonicalize
    # from this module keep working unchanged.
    return resolve_coe_label(raw_coe) or raw_coe.strip()

def _fingerprint(skills_df: pd.DataFrame) -> tuple:
    return (len(skills_df), int(pd.util.hash_pandas_object(skills_df, index=False).sum()))

def get_employee_primary_coe_map() -> dict[str, str]:
    global _cache, _cache_fingerprint
    adapter = get_adapter()
    skills = adapter.get_skills()
    fingerprint = _fingerprint(skills)
    if _cache is not None and fingerprint == _cache_fingerprint:
        return _cache

    observed = skills[(skills["skill_source"] == "observed") & (~skills["coe"].isin(GENERIC_SKILL_COES))]
    result: dict[str, str] = {}
    if not observed.empty:
        mode_coe = observed.groupby("employee_id")["coe"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        result = {emp_id: _canonicalize(coe) for emp_id, coe in mode_coe.items() if coe is not None}

    _cache = result
    _cache_fingerprint = fingerprint
    return result