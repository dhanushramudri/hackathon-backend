"""Maps the synthetic skills/competency dataset onto real employees.

Confirmed live: 05_Skill_Details_clean.csv (300,218 rows) and
06_Competency_Details_clean.csv (1,577 rows) both cover exactly the same
1,050 synthetic employee_ids (EMP1, EMP2, ... -- 100% overlap between the
two files), and NONE of those IDs match a real employee_id (real IDs are
JM-prefixed/numeric/JML-/EXT-, confirmed zero overlap either way). So the
real employee roster has no real skills/competency data at all, and the
skills/competency data has no real employee link -- two disconnected real
datasets, neither fabricated on its own, that were simply never joined.

There is no real join key between them (no shared ID, no name). This module
does NOT invent one. Instead, each real employee is assigned the synthetic
profile of a synthetic employee who shares their real designation (job_name)
and, where derivable, their real CoE -- a plausible stand-in, not a claim
about that specific person's actual skills. The real CoE comes from
experience_engine.build_employee_experience_profiles()'s tech_coe_breakdown
(derived from real project allocations), NOT from employee_coe.py's
get_employee_primary_coe_map() -- that function reads the same broken
skills table by employee_id and is itself returning empty results for every
real employee right now (separate, pre-existing bug, not fixed here).

Every result carries is_synthetic=True and match_type -- callers MUST
surface that, never presenting this as a real skill assessment."""
import hashlib

import pandas as pd

# _cached_query, not get_adapter().get_skills()/get_competencies(): those
# public methods now return the REMAPPED (real-ID) table produced by this
# very module (see build_real_employee_skills_table/_competency_table below,
# wired in from app.core.adapter.LocalAdapter.get_skills/get_competencies)
# -- this module needs the ORIGINAL fake-ID table to build that remap in the
# first place, so it reads straight from the shared DuckDB cache instead.
from app.core.adapter import _cached_query, get_adapter
from app.engines.coe_taxonomy import normalize_coe_label
from app.engines.experience_engine import build_employee_experience_profiles

_cache: dict[str, dict] | None = None
_cache_fingerprint: tuple | None = None
_table_cache: dict[str, pd.DataFrame] = {}
_table_cache_fingerprint: tuple | None = None


def _raw_skills() -> pd.DataFrame:
    return _cached_query("skills").copy()


def _raw_competencies() -> pd.DataFrame:
    return _cached_query("competencies").copy()


def _fingerprint(skills: pd.DataFrame, employees: pd.DataFrame) -> tuple:
    return (
        len(skills), int(pd.util.hash_pandas_object(skills["employee_id"], index=False).sum()),
        len(employees), int(pd.util.hash_pandas_object(employees["employee_id"], index=False).sum()),
    )


def _real_employee_coe(emp_id: str, experience_profiles: dict) -> str | None:
    profile = experience_profiles.get(emp_id)
    breakdown = (profile or {}).get("tech_coe_breakdown") or {}
    if not breakdown:
        return None
    top_label = max(breakdown.items(), key=lambda kv: kv[1])[0]
    return normalize_coe_label(top_label)


def _stable_pick(pool: list[str], real_employee_id: str) -> str:
    """Deterministic per real employee (same result every reload) but
    spread across the pool instead of every employee sharing a designation
    collapsing onto pool[0]."""
    idx = int(hashlib.sha1(real_employee_id.encode()).hexdigest(), 16) % len(pool)
    return pool[idx]


# A small, hand-reviewed list of designation spellings that ARE genuinely
# the same real role (not just similar-looking words) -- e.g. "Technical"
# vs "Technology" Solutions Architect is confirmed elsewhere in this app
# (role_hierarchy.py) to be inconsistent data entry for one real grade, not
# two different functions. Deliberately NOT a generic fuzzy/token-overlap
# matcher: an earlier version of this file tried that and it matched "IT
# Solution Consultant" (an internal IT-services role) to "Solutions
# Consultant" (a client-delivery role) purely because they share the word
# "consultant" -- confirmed wrong (different department, different function)
# and reverted. Only add an entry here after confirming the two titles are
# genuinely the same role, the same way role_hierarchy.py's LEVELS groups
# same-level UK/India naming-family pairs.
_DESIGNATION_ALIASES: dict[str, str] = {
    "technology solutions architect": "technical solutions architect",
}


def _resolve_designation_alias(designation_key: str, synthetic_designation_keys: set[str]) -> str | None:
    alias = _DESIGNATION_ALIASES.get(designation_key)
    if alias and alias in synthetic_designation_keys:
        return alias
    return None


def get_employee_skill_mapping(with_coe_refinement: bool = False) -> dict[str, dict]:
    """{real_employee_id: {synthetic_employee_id, match_type, designation, coe}}

    match_type is "designation_and_coe" (only when with_coe_refinement=True
    -- see below), "designation_only" (job_name matched exactly), or
    "designation_alias" (title differs only by a confirmed same-role
    spelling variant, see _DESIGNATION_ALIASES). A real employee absent from
    this dict has no exact or confirmed-alias match at all (e.g. "Chief
    Financial Officer", "IT Solution Consultant") -- deliberately left
    unmapped rather than guessing from superficial word overlap, since this
    dataset only covers delivery/technical roles and a wrong guess (e.g.
    treating an internal IT-services role as equivalent to a client-delivery
    "Solutions Consultant" just because both contain "consultant") would
    misrepresent that employee's actual function, not just their skills.

    with_coe_refinement=True additionally narrows the match using each real
    employee's actual CoE (derived from experience_engine.
    build_employee_experience_profiles()'s real allocation history) --
    default OFF. That function is a genuinely expensive (~70s+ on real data
    volume, confirmed) cold computation that's otherwise unrelated to this
    mapping; this function is called via adapter.get_skills(), which dozens
    of call sites across the app reach through completely unrelated code
    paths (GET /employees, free pool, leave, recommendations...) -- forcing
    all of them to eat that cost on whichever one happens to run first per
    server lifetime is a real regression (confirmed: caused genuine request
    timeouts/socket hang-ups on /employees and /forecast/top-candidates-for-
    role in practice), for a refinement that only improves ~1/3 of already-
    mapped employees from "designation match" to "designation+CoE match".
    Callers who specifically want the finer-grained match (and are already
    paying for experience_profiles anyway, e.g. recommendation_service.py)
    can opt in explicitly."""
    global _cache, _cache_fingerprint
    skills = _raw_skills()
    employees = get_adapter().get_employees()
    fingerprint = _fingerprint(skills, employees) + (with_coe_refinement,)
    if _cache is not None and fingerprint == _cache_fingerprint:
        return _cache

    synthetic_profiles = (
        skills.dropna(subset=["employee_id", "designation"])
        .groupby("employee_id")
        .agg(designation=("designation", "first"), coe=("coe", "first"))
        .reset_index()
    )
    synthetic_profiles["designation_key"] = synthetic_profiles["designation"].str.strip().str.lower()
    synthetic_profiles["coe_key"] = synthetic_profiles["coe"].apply(normalize_coe_label)

    pool_by_designation: dict[str, list[str]] = {}
    pool_by_designation_coe: dict[tuple[str, str], list[str]] = {}
    for row in synthetic_profiles.itertuples(index=False):
        pool_by_designation.setdefault(row.designation_key, []).append(row.employee_id)
        if row.coe_key:
            pool_by_designation_coe.setdefault((row.designation_key, row.coe_key), []).append(row.employee_id)

    experience_profiles = build_employee_experience_profiles() if with_coe_refinement else {}
    result: dict[str, dict] = {}
    synthetic_designation_keys = set(pool_by_designation.keys())
    for emp in employees.itertuples(index=False):
        emp_id = emp.employee_id
        if not pd.notna(emp.job_name):
            continue
        designation_key = emp.job_name.strip().lower()
        if not designation_key:
            continue

        real_coe = _real_employee_coe(emp_id, experience_profiles) if with_coe_refinement else None
        pool = pool_by_designation_coe.get((designation_key, real_coe)) if real_coe else None
        match_type = "designation_and_coe"
        if not pool:
            pool = pool_by_designation.get(designation_key)
            match_type = "designation_only"
        if not pool:
            alias_key = _resolve_designation_alias(designation_key, synthetic_designation_keys)
            if alias_key:
                pool = pool_by_designation[alias_key]
                match_type = "designation_alias"
        if not pool:
            continue

        result[emp_id] = {
            "synthetic_employee_id": _stable_pick(pool, emp_id),
            "match_type": match_type,
            "designation": emp.job_name,
            "coe": real_coe,
        }

    _cache = result
    _cache_fingerprint = fingerprint
    return result


def get_real_employee_skills(real_employee_id: str) -> dict | None:
    """Real per-employee skill rows via the mapping above -- always
    is_synthetic=True. Returns None if this employee has no mapped profile
    at all (no synthetic profile shares their real designation)."""
    entry = get_employee_skill_mapping().get(real_employee_id)
    if entry is None:
        return None
    skills = _raw_skills()
    rows = skills[skills["employee_id"] == entry["synthetic_employee_id"]]
    return {
        "is_synthetic": True,
        "match_type": entry["match_type"],
        "designation": entry["designation"],
        "coe": entry["coe"],
        "synthetic_source_employee_id": entry["synthetic_employee_id"],
        "skills": rows[["coe_skill", "skill", "subskill", "experience", "score"]].to_dict("records"),
    }


def get_real_employee_competency(real_employee_id: str) -> dict | None:
    """Real per-employee competency rows via the SAME mapping (same
    synthetic_employee_id as skills, so both come from one consistent
    synthetic person, not two different ones) -- always is_synthetic=True."""
    entry = get_employee_skill_mapping().get(real_employee_id)
    if entry is None:
        return None
    competencies = _raw_competencies()
    rows = competencies[competencies["employee_id"] == entry["synthetic_employee_id"]]
    return {
        "is_synthetic": True,
        "match_type": entry["match_type"],
        "designation": entry["designation"],
        "coe": entry["coe"],
        "synthetic_source_employee_id": entry["synthetic_employee_id"],
        "competencies": rows[["competency_sheet", "competency_question", "response", "score"]].to_dict("records"),
    }


def _fingerprint_mapping(mapping: dict[str, dict]) -> tuple:
    return (len(mapping), hash(tuple(sorted((k, v["synthetic_employee_id"]) for k, v in mapping.items()))))


def _remap_table(raw: pd.DataFrame, mapping: dict[str, dict]) -> pd.DataFrame:
    """Vectorized version of what get_real_employee_skills/_competency do
    one employee at a time -- explodes the mapping (real_employee_id ->
    synthetic_employee_id) against the raw fake-ID table so every real,
    mapped employee gets their assigned synthetic profile's full row set,
    re-labeled with their own real employee_id. Real employees who share a
    synthetic donor (unavoidable: 925 real employees, ~1,050 synthetic
    profiles, matched by designation not 1:1 identity) each get their own
    copy of those rows -- intentional, not a duplicate-data bug. Unmapped
    real employees contribute no rows at all, same as today's silent-miss
    behavior for them, except now it's because there's truly nothing to
    assign, not because of an ID mismatch bug."""
    if not mapping:
        return raw.iloc[0:0].assign(is_synthetic=pd.Series(dtype=bool), match_type=pd.Series(dtype=str))
    map_df = pd.DataFrame(
        [{"employee_id": real_id, "_synthetic_id": v["synthetic_employee_id"], "match_type": v["match_type"]} for real_id, v in mapping.items()]
    )
    merged = map_df.merge(raw, left_on="_synthetic_id", right_on="employee_id", how="inner", suffixes=("", "_synthetic"))
    merged["is_synthetic"] = True
    merged = merged.drop(columns=["employee_id_synthetic"]).rename(columns={"_synthetic_id": "synthetic_source_employee_id"})
    return merged


def build_real_employee_skills_table() -> pd.DataFrame:
    """The full skills table, real-employee_id-keyed -- what
    LocalAdapter.get_skills() actually returns now, so every existing
    consumer (scoring.build_employee_skill_index, embedding_engine.
    build_employee_embedding_index, semantic_match_service, etc.) gets real
    IDs with zero changes at each of those call sites."""
    global _table_cache, _table_cache_fingerprint
    mapping = get_employee_skill_mapping()
    fingerprint = _fingerprint_mapping(mapping)
    if _table_cache.get("skills") is not None and _table_cache_fingerprint == fingerprint:
        return _table_cache["skills"]
    table = _remap_table(_raw_skills(), mapping)
    _table_cache["skills"] = table
    _table_cache_fingerprint = fingerprint
    return table


def build_real_employee_competency_table() -> pd.DataFrame:
    """Same real-ID remap as build_real_employee_skills_table, for
    LocalAdapter.get_competencies()."""
    global _table_cache, _table_cache_fingerprint
    mapping = get_employee_skill_mapping()
    fingerprint = _fingerprint_mapping(mapping)
    if _table_cache.get("competencies") is not None and _table_cache_fingerprint == fingerprint:
        return _table_cache["competencies"]
    table = _remap_table(_raw_competencies(), mapping)
    _table_cache["competencies"] = table
    _table_cache_fingerprint = fingerprint
    return table
