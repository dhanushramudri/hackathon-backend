"""Employee project-history / track-record engine.

Builds a per-employee profile of real (COMPLETE/ACTIVE) past project experience
from allocations + projects, and matches it against a new deal's category to
produce a track-record signal used as a tie-breaker (and optional advanced
weight) in recommendation ranking -- distinct from skill/competency/availability.
"""

import pandas as pd

from app.core.adapter import get_adapter

# Only these statuses represent real delivered (or currently being delivered)
# work -- PROPOSE/DEAL WON/DEAL LOST/SOW PENDING/SCOPING APPROVAL/CLOSED are
# pipeline/sales states, not evidence the employee actually did the work.
REAL_STATUSES = frozenset({"COMPLETE", "ACTIVE"})

# Internal work is discretionary, not client delivery -- consistent with how
# availability_as_of() in recommendation_service.py already excludes it from
# "busy" capacity. It still counts toward total_projects (breadth/tenure) but
# never toward category-relevance matching.
INTERNAL_PROJECT_TYPE = "Internal Project"

UNTAGGED_VALUES = frozenset({"", "nan", "not_mapped", "none"})

# Recency decay -- a project that ended 5 years ago shouldn't count identically
# to one that ended last quarter. ACTIVE projects (still ongoing) always get
# full weight since the employee is delivering in that category right now.
RECENT_DAYS = 548   # ~18 months
MID_DAYS = 1095     # ~3 years
RECENT_WEIGHT = 1.0
MID_WEIGHT = 0.5
OLD_WEIGHT = 0.25

TOP_CATEGORIES_N = 5

# "Number of projects" is a standalone breadth/seniority signal, independent of
# whether any of those projects match the deal's category -- a veteran with 25
# projects across many categories still reads as more proven than someone with 2,
# even before category relevance is considered. Cap chosen from the real data
# distribution (median ~7 real projects org-wide, so 20 already reads as "very
# experienced" without being dominated by the small number of 100+-project outliers).
PROJECT_COUNT_NORMALIZATION_CAP = 20


def _split_multi(value) -> list[str]:
    """Split a semicolon-joined multi-value field (e.g. tech_coe) into clean labels."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = [p.strip() for p in str(value).split(";")]
    return [p for p in parts if p and p.lower() not in UNTAGGED_VALUES]


def _recency_weight(status: str, project_end_date, today: pd.Timestamp) -> float:
    if status == "ACTIVE":
        return RECENT_WEIGHT
    if pd.isna(project_end_date):
        return MID_WEIGHT  # COMPLETE but no end date on file -- treat as moderately recent, not unknown-penalized
    days_since_end = (today - pd.Timestamp(project_end_date)).days
    if days_since_end <= RECENT_DAYS:
        return RECENT_WEIGHT
    if days_since_end <= MID_DAYS:
        return MID_WEIGHT
    return OLD_WEIGHT


_profile_cache: dict[str, dict] | None = None
_profile_fingerprint: tuple | None = None


def _fingerprint(allocations: pd.DataFrame, projects: pd.DataFrame) -> tuple:
    return (
        len(allocations),
        len(projects),
        int(pd.util.hash_pandas_object(allocations["project_id"], index=False).sum()),
        int(pd.util.hash_pandas_object(projects["project_code"], index=False).sum()),
    )


def build_employee_experience_profiles() -> dict[str, dict]:
    """One profile per employee who has ≥1 real (COMPLETE/ACTIVE) project.

    Each profile:
      total_projects: int -- distinct real project_code count (any type, breadth signal)
      distinct_clients: int -- distinct client_id across client-facing real projects
      proposition_breakdown: dict[str, float] -- recency-weighted count per proposition_coe label
      tech_coe_breakdown: dict[str, float] -- recency-weighted count per tech_coe label
    Employees with zero real project history are simply absent from the dict --
    callers must treat a missing employee_id as "no_history", not zero/penalized.
    """
    global _profile_cache, _profile_fingerprint
    adapter = get_adapter()
    allocations = adapter.get_allocations()
    projects = adapter.get_projects()

    fingerprint = _fingerprint(allocations, projects)
    if _profile_cache is not None and fingerprint == _profile_fingerprint:
        return _profile_cache

    proj_cols = [
        "project_code", "project_status", "type_of_project", "client_id",
        "proposition_coe", "tech_coe", "project_end_date",
    ]
    merged = allocations[["employee_id", "project_id"]].merge(
        projects[proj_cols], left_on="project_id", right_on="project_code", how="inner"
    )
    real = merged[merged["project_status"].isin(REAL_STATUSES)]
    # Dedup: an employee can have multiple allocation rows against the same project
    # (role/percentage changes mid-project) -- experience is per-project, not per-allocation-row.
    real = real.drop_duplicates(subset=["employee_id", "project_code"])

    today = pd.Timestamp.now().normalize()
    real = real.assign(
        recency_weight=[
            _recency_weight(status, end_date, today)
            for status, end_date in zip(real["project_status"], real["project_end_date"])
        ]
    )

    is_client_facing = real["type_of_project"] != INTERNAL_PROJECT_TYPE

    profiles: dict[str, dict] = {}
    for emp_id, group in real.groupby("employee_id"):
        client_facing = group[is_client_facing.loc[group.index]]

        proposition_breakdown: dict[str, float] = {}
        tech_coe_breakdown: dict[str, float] = {}
        for row in client_facing.itertuples(index=False):
            w = row.recency_weight
            for label in _split_multi(row.proposition_coe):
                proposition_breakdown[label] = proposition_breakdown.get(label, 0.0) + w
            for label in _split_multi(row.tech_coe):
                tech_coe_breakdown[label] = tech_coe_breakdown.get(label, 0.0) + w

        distinct_clients = client_facing.loc[client_facing["client_id"].notna(), "client_id"].nunique()

        profiles[emp_id] = {
            "total_projects": int(group["project_code"].nunique()),
            "distinct_clients": int(distinct_clients),
            "proposition_breakdown": proposition_breakdown,
            "tech_coe_breakdown": tech_coe_breakdown,
        }

    _profile_cache = profiles
    _profile_fingerprint = fingerprint
    return profiles


def _top_categories(breakdown: dict[str, float], n: int = TOP_CATEGORIES_N) -> list[dict]:
    ranked = sorted(breakdown.items(), key=lambda kv: -kv[1])[:n]
    return [{"category": k, "count": round(v, 2)} for k, v in ranked]


def _project_count_score(total_projects: int) -> float:
    return round(min(total_projects / PROJECT_COUNT_NORMALIZATION_CAP, 1.0), 3)


def match_experience(
    profile: dict | None,
    requested_solution: str | None,
    requested_tech_coes: list[str] | None = None,
) -> dict:
    """Match one employee's experience profile against a deal's requested category.

    Matching scope (confirmed): the deal's `Solution` field (proposition_coe
    vocabulary) is the primary match target. When absent, falls back to
    tech_coe overlap as a weaker, explicitly-labeled signal.

    Returns a dict always safe to consume regardless of profile/requested_*
    being None/empty -- callers never need to special-case missing data.
    """
    has_category = bool(requested_solution) or bool(requested_tech_coes)
    if not has_category:
        total = profile["total_projects"] if profile else 0
        return {
            "relevant_project_count": 0.0,
            "relevant_project_ratio": 0.0,
            "total_projects": total,
            "distinct_clients": profile["distinct_clients"] if profile else 0,
            "experience_confidence": "no_requirement",
            "top_categories": _top_categories(profile["proposition_breakdown"]) if profile else [],
            "project_count_score": _project_count_score(total),
        }

    if profile is None or profile["total_projects"] == 0:
        return {
            "relevant_project_count": 0.0,
            "relevant_project_ratio": 0.0,
            "total_projects": 0,
            "distinct_clients": 0,
            "experience_confidence": "no_history",
            "top_categories": [],
            "project_count_score": 0.0,
        }

    total = profile["total_projects"]
    relevant = 0.0
    confidence = "no_match"

    if requested_solution:
        relevant = profile["proposition_breakdown"].get(requested_solution, 0.0)
        if relevant > 0:
            confidence = "observed"

    if relevant == 0.0 and requested_tech_coes:
        tech_hit = sum(profile["tech_coe_breakdown"].get(c, 0.0) for c in requested_tech_coes)
        if tech_hit > 0:
            relevant = tech_hit
            confidence = "related_only"

    return {
        "relevant_project_count": round(relevant, 2),
        "relevant_project_ratio": round(relevant / total, 3) if total else 0.0,
        "total_projects": total,
        "distinct_clients": profile["distinct_clients"],
        "experience_confidence": confidence,
        "top_categories": _top_categories(profile["proposition_breakdown"]),
        "project_count_score": _project_count_score(total),
    }


def get_employee_project_history(employee_id: str, category: str | None = None) -> list[dict]:
    """Raw list of one employee's real (COMPLETE/ACTIVE) past projects, for
    on-demand drilldown (clicking "13 projects" / "6 clients" / a category chip)
    -- not part of the cached bulk profile since this is a single-employee,
    on-click lookup, not something computed for every candidate in a ranking pass.

    category, when given, filters to projects whose proposition_coe includes it
    (matches the same multi-value split used in build_employee_experience_profiles).
    """
    adapter = get_adapter()
    allocations = adapter.get_allocations()
    projects = adapter.get_projects()

    proj_cols = [
        "project_code", "project_status", "type_of_project", "client_id",
        "proposition_coe", "tech_coe", "project_start_date", "project_end_date",
    ]
    emp_allocs = allocations.loc[allocations["employee_id"] == employee_id, ["employee_id", "project_id"]]
    merged = emp_allocs.merge(projects[proj_cols], left_on="project_id", right_on="project_code", how="inner")
    real = merged[merged["project_status"].isin(REAL_STATUSES)].drop_duplicates(subset=["project_code"])

    if category:
        real = real[real["proposition_coe"].apply(lambda v: category in _split_multi(v))]

    real = real.sort_values("project_end_date", ascending=False, na_position="last")

    def _fmt_date(v):
        return v.strftime("%Y-%m-%d") if pd.notna(v) else None

    return [
        {
            "project_code": row.project_code,
            "client_id": row.client_id if pd.notna(row.client_id) else None,
            "proposition_coe": _split_multi(row.proposition_coe),
            "tech_coe": _split_multi(row.tech_coe),
            "status": row.project_status,
            "type_of_project": row.type_of_project,
            "start_date": _fmt_date(row.project_start_date),
            "end_date": _fmt_date(row.project_end_date),
        }
        for row in real.itertuples(index=False)
    ]
