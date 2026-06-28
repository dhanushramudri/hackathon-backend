import pandas as pd

from app.core.adapter import get_adapter
from app.engines.coe_skill_engine import COE_SKILL_MAP, GENERIC_SKILL_COES

_REVERSE_COE_MAP: dict[str, str] = {
    skill_coe.strip().lower(): canonical for canonical, mapping in COE_SKILL_MAP.items() for skill_coe in mapping["skill_coes"]
}

_cache: dict[str, str] | None = None
_cache_fingerprint: tuple | None = None

def _canonicalize(raw_coe: str) -> str:
    mapped = _REVERSE_COE_MAP.get(raw_coe.strip().lower())
    if mapped:
        return mapped
    cleaned = raw_coe.strip()
    return cleaned.title() if cleaned.islower() else cleaned

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
        mode_coe = observed.groupby("employee_id")["coe"].agg(lambda s: s.mode().iat[0])
        result = {emp_id: _canonicalize(coe) for emp_id, coe in mode_coe.items()}

    _cache = result
    _cache_fingerprint = fingerprint
    return result
