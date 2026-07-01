import re

import pandas as pd

from app.core.adapter import get_adapter
from app.engines import scoring
from app.engines.designation_ladder import adjacent_designations
from app.engines.employee_coe import get_employee_primary_coe_map
from app.engines.resource_code_decoder import decode_resource_code
from app.engines.skillset_classifier import classify_skillset, classify_skillset_with_proof

TOP_N = 15
MAX_FALLBACK_CANDIDATES = 5

class RowIndexOutOfRange(Exception):

    def __init__(self, row_index: int, max_index: int):
        self.row_index = row_index
        self.max_index = max_index
        super().__init__(f"row_index {row_index} out of range (0-{max_index})")

def _fmt_date(value) -> str | None:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else None

INTERNAL_PROJECT_TYPE = "Internal Project"

def availability_as_of(allocations: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.Series:
    active_then = allocations[
        (allocations["is_allocation_active"] == 1)
        & (allocations["allocated_start_date"] <= as_of_date)
        & (allocations["allocated_end_date"] >= as_of_date)
    ]
    # Internal-project work is discretionary ("contribute when you have time"), not a
    # hard commitment -- it must never count as "busy" when checking real capacity for
    # a new client engagement, redeployment, or AI semantic match. Every caller of this
    # function (get_recommendations, get_redeploy_candidates_as_of, semantic match) goes
    # through here, so the fix is centralized.
    projects = get_adapter().get_projects()[["project_code", "type_of_project"]].rename(
        columns={"project_code": "project_id"}
    )
    active_then = active_then.merge(projects, on="project_id", how="left")
    client_only = active_then[active_then["type_of_project"] != INTERNAL_PROJECT_TYPE]
    busy_pct = client_only.groupby("employee_id")["allocation_by_percentage"].sum()
    return busy_pct

def _match_tier(candidate: dict, requested_designations: list[str] | None) -> str | None:
    if not requested_designations:
        return None
    if candidate["skill_confidence"] not in ("no_match", "no_requirement"):
        return "skill_match"
    if candidate["job_name"] in requested_designations:
        return "same_grade_fallback"
    adjacent = {
        d for req in requested_designations for d, _offset in adjacent_designations(req)
    }
    if candidate["job_name"] in adjacent:
        return "adjacent_level_fallback"
    return None

def _build_fallback_candidates(ranked: list[dict], requested_designations: list[str], requested_coe_categories: list[str]) -> dict:
    coe_wanted = {c.strip().lower() for c in requested_coe_categories}

    def _sort_key(c: dict) -> tuple:
        coe_match = 1 if (c.get("coe") or "").strip().lower() in coe_wanted else 0
        return (-c["skill_score"], -coe_match, -c["available_pct"], -c["competency_score"])

    adjacent = {
        d for req in requested_designations for d, _offset in adjacent_designations(req)
    } - set(requested_designations)

    same_grade = sorted(
        [c for c in ranked if c["job_name"] in requested_designations], key=_sort_key
    )[:MAX_FALLBACK_CANDIDATES]
    adjacent_level = sorted(
        [c for c in ranked if c["job_name"] in adjacent], key=_sort_key
    )[:MAX_FALLBACK_CANDIDATES]

    for c in same_grade:
        c["match_tier"] = "same_grade_fallback"
    for c in adjacent_level:
        c["match_tier"] = "adjacent_level_fallback"

    return {
        "requested_designations": requested_designations,
        "same_grade": same_grade,
        "adjacent_level": adjacent_level,
    }

EARLIEST_AVAILABILITY_SEARCH_DAYS = 180

def find_earliest_availability(
    employee_id: str, allocations: pd.DataFrame, after_date: pd.Timestamp, requested_pct: float
) -> dict | None:
    window_end = after_date + pd.Timedelta(days=EARLIEST_AVAILABILITY_SEARCH_DAYS)
    own_active = allocations[
        (allocations["employee_id"] == employee_id)
        & (allocations["is_allocation_active"] == 1)
        & (allocations["allocated_end_date"] > after_date)
        & (allocations["allocated_end_date"] <= window_end)
    ]
    candidate_end_dates = sorted(own_active["allocated_end_date"].dropna().unique())

    for end_date in candidate_end_dates:
        check_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        busy = float(availability_as_of(allocations, check_date).get(employee_id, 0.0))
        available_pct = max(0.0, 100.0 - busy)
        if available_pct >= requested_pct:
            return {
                "earliest_available_date": check_date.strftime("%Y-%m-%d"),
                "proof": (
                    f"Current allocation ends {pd.Timestamp(end_date).strftime('%Y-%m-%d')}, "
                    f"freeing up {round(available_pct, 1)}% capacity from {check_date.strftime('%Y-%m-%d')}."
                ),
            }
    return None

def get_recommendations(
    skillset_text: str,
    likely_start_date: str,
    requested_pct_raw: str = "100",
    top_n: int = TOP_N,
    *,
    requested_designations: list[str] | None = None,
    requested_coe_categories: list[str] | None = None,
    compute_earliest_availability: bool = True,
    employees: pd.DataFrame | None = None,
    competencies: pd.DataFrame | None = None,
    allocations: pd.DataFrame | None = None,
    pipeline_skillset: pd.DataFrame | None = None,
    skills: pd.DataFrame | None = None,
    skill_index: dict | None = None,
    employee_coe_map: dict | None = None,
) -> dict:
    adapter = get_adapter()
    employees = adapter.get_employees() if employees is None else employees
    competencies = adapter.get_competencies() if competencies is None else competencies
    allocations = adapter.get_allocations() if allocations is None else allocations
    pipeline_skillset = adapter.get_pipeline_skillset() if pipeline_skillset is None else pipeline_skillset

    as_of_date = pd.to_datetime(likely_start_date)
    requested_pct = scoring.parse_requested_pct(requested_pct_raw)
    required_phrases = scoring.tokenize_skillset(skillset_text)
    required_phrases = scoring.enrich_required_phrases(required_phrases, pipeline_skillset)
    if skill_index is None:
        skills = adapter.get_skills() if skills is None else skills
        skill_index = scoring.build_employee_skill_index(skills)
    if employee_coe_map is None:
        employee_coe_map = get_employee_primary_coe_map()
    busy_pct = availability_as_of(allocations, as_of_date)

    active_employees = employees[employees["account_status"] == 1]
    job_name_by_id = active_employees.set_index("employee_id")["job_name"].to_dict()
    competency_index = scoring.build_employee_competency_index(competencies)
    default_competency = {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"}

    results = []
    for emp_id in active_employees["employee_id"]:
        job_name = job_name_by_id.get(emp_id)
        skill_result = scoring.score_skill_match(required_phrases, skill_index.get(emp_id, {}))
        competency_entry = competency_index.get(emp_id, default_competency)
        competency_score = competency_entry["score"]
        competency_confidence = competency_entry["confidence"]
        available_pct = max(0.0, 100.0 - float(busy_pct.get(emp_id, 0.0)))
        availability_score = min(available_pct / 100.0, 1.0)
        composite = scoring.composite_score(skill_result["score"], competency_score, availability_score)
        bucket_value = scoring.bucket(skill_result["score"], skill_result["confidence"])
        meets_requested_capacity = bool(available_pct >= requested_pct)

        results.append(
            {
                "employee_id": emp_id,
                "job_name": job_name,
                "coe": employee_coe_map.get(emp_id),
                "composite_score": composite,
                "bucket": bucket_value,
                "staffing_signal": scoring.staffing_signal(bucket_value),
                "explanation": scoring.explain_candidate(
                    employee_id=emp_id,
                    job_name=job_name,
                    bucket_value=bucket_value,
                    skill_result=skill_result,
                    competency_score=competency_score,
                    available_pct=available_pct,
                    requested_pct=requested_pct,
                    meets_requested_capacity=meets_requested_capacity,
                    competency_confidence=competency_confidence,
                ),
                "skill_score": skill_result["score"],
                "matched_skills": skill_result["matched"],
                "missing_skills": skill_result["missing"],
                "skill_confidence": skill_result["confidence"],
                "competency_score": competency_score,
                "competency_confidence": competency_confidence,
                "available_pct": round(available_pct, 1),
                "meets_requested_capacity": meets_requested_capacity,
            }
        )

    ranked = sorted(results, key=lambda r: r["composite_score"], reverse=True)
    candidates_meeting_capacity = [r for r in ranked if r["meets_requested_capacity"]]
    pool = candidates_meeting_capacity or ranked
    top = pool[:top_n]

    for c in top:
        c["match_tier"] = _match_tier(c, requested_designations)
        c["earliest_available_date"] = None
        c["earliest_available_proof"] = None

    best_fit_if_delayed: list[dict] = []
    if compute_earliest_availability:
        shown_ids = {c["employee_id"] for c in top}
        for c in ranked[:10]:
            if len(best_fit_if_delayed) >= 3:
                break
            if c["meets_requested_capacity"] or c["employee_id"] in shown_ids:
                continue
            found = find_earliest_availability(c["employee_id"], allocations, as_of_date, requested_pct)
            if found:
                best_fit_if_delayed.append(
                    {
                        **c,
                        "match_tier": _match_tier(c, requested_designations),
                        "earliest_available_date": found["earliest_available_date"],
                        "earliest_available_proof": found["proof"],
                    }
                )

    top_signal = top[0]["staffing_signal"] if top else "hire"
    hire_vs_redeploy = top_signal == "hire"
    has_skillset = bool(required_phrases)
    # top_n is a fixed display cap (TOP_N), not a measure of how many people genuinely
    # match -- without these, "Candidates (15/15)" looks identical whether 15 people
    # skill-matched at 100% or zero phrases were ever specified and every "candidate" is
    # really just the most available, unranked-by-skill person (bucket="not_assessed").
    # Surfacing the real pool size and match count lets the UI say so honestly.
    real_match_count = sum(1 for r in top if r["bucket"] != "not_assessed")
    genuine_skill_match_count = sum(
        1 for r in top if r["skill_confidence"] not in ("no_match", "no_requirement")
    )

    fallback_candidates = None
    if requested_designations and has_skillset and genuine_skill_match_count == 0:
        fallback_candidates = _build_fallback_candidates(
            ranked, requested_designations, requested_coe_categories or []
        )

    return {
        "request": {
            "skillset_text": skillset_text,
            "required_phrases": required_phrases,
            "likely_start_date": likely_start_date,
            "requested_pct": requested_pct,
        },
        "candidates": top,
        "hire_vs_redeploy_flag": hire_vs_redeploy,
        "top_candidate_signal": top_signal,
        "has_skillset": has_skillset,
        "total_employees_considered": int(len(active_employees)),
        "candidate_pool_size": int(len(pool)),
        "candidates_with_real_skill_match": real_match_count,
        "genuine_skill_match_count": genuine_skill_match_count,
        "fallback_candidates": fallback_candidates,
        "best_fit_if_delayed": best_fit_if_delayed,
    }

def get_recommendations_for_pipeline_row(
    row_index: int, pipeline: pd.DataFrame | None = None, top_n: int = TOP_N, **prefetched
) -> dict:
    adapter = get_adapter()
    pipeline = adapter.get_pipeline_forecast() if pipeline is None else pipeline
    if row_index < 0 or row_index >= len(pipeline):
        raise RowIndexOutOfRange(row_index, len(pipeline) - 1)

    row = pipeline.iloc[row_index]
    requested_designations = decode_resource_code(row.get("resources_requested"))
    requested_coe_categories, skillset_classification_proof = classify_skillset_with_proof(row.get("skillset"))
    result = get_recommendations(
        skillset_text=row.get("skillset", ""),
        likely_start_date=str(row.get("likely_start_date")),
        requested_pct_raw=row.get("requested_pct", "100"),
        top_n=top_n,
        requested_designations=requested_designations,
        requested_coe_categories=requested_coe_categories,
        **prefetched,
    )
    cluster = row.get("cluster")
    deal_id = row.get("deal_id")
    result["pipeline_row"] = {
        "row_index": row_index,
        "deal_id": int(deal_id) if pd.notna(deal_id) else None,
        "cluster": int(cluster) if pd.notna(cluster) else None,
        "client": row.get("client"),
        "client_priority": row.get("client_priority"),
        "em": row.get("em"),
        "solution": row.get("solution"),
        "resources_requested": row.get("resources_requested"),
        "requested_pct": row.get("requested_pct"),
        "sow_signed": row.get("sow_signed"),
        "status": row.get("status"),
        "priority": row.get("priority"),
        "likely_start_date": _fmt_date(row.get("likely_start_date")),
        "request_received": _fmt_date(row.get("request_received")),
        "original_requested_start_date": _fmt_date(row.get("original_requested_start_date")),
        "start_date_confirmed": row.get("start_date_confirmed"),
        "number_of_weeks": row.get("number_of_weeks") if pd.notna(row.get("number_of_weeks")) else None,
        "request_type": row.get("request_type"),
        "deal_stage_hubspot": row.get("deal_stage_hubspot"),
        "comments": row.get("comments"),
        "skillset_coe_categories": requested_coe_categories,
        "skillset_classification_proof": skillset_classification_proof,
        "requested_designations": requested_designations,
    }

    if pd.notna(deal_id):
        siblings = pipeline[pipeline["deal_id"] == deal_id].sort_index()
        result["deal_composition"] = [
            {
                "row_index": int(idx),
                "resources_requested": sib.get("resources_requested"),
                "requested_pct": sib.get("requested_pct"),
                "skillset": sib.get("skillset"),
                "is_current": int(idx) == row_index,
            }
            for idx, sib in siblings.iterrows()
        ]
    else:
        result["deal_composition"] = []

    return result

_open_rows_cache: list[dict] | None = None
_open_rows_fingerprint: tuple | None = None

def _open_pipeline_rows_enriched() -> list[dict]:
    """Open (not-yet-resourced) pipeline rows with required_phrases/skill_areas
    precomputed once. tokenize_skillset + enrich_required_phrases + classify_skillset
    are each a real DataFrame scan -- redoing them per employee (245 open rows x 412
    free-pool people) measured at ~500s for the full pool. Precomputing once here and
    caching (same fingerprint pattern as scoring.build_employee_skill_index) drops that
    to a single pass; every per-employee match call below reuses this list."""
    global _open_rows_cache, _open_rows_fingerprint
    adapter = get_adapter()
    pipeline = adapter.get_pipeline_forecast()
    fingerprint = (len(pipeline), int(pd.util.hash_pandas_object(pipeline["status"], index=False).sum()))
    if _open_rows_cache is not None and fingerprint == _open_rows_fingerprint:
        return _open_rows_cache

    pipeline_skillset = adapter.get_pipeline_skillset()
    is_open = ~pipeline["status"].fillna("").str.strip().str.lower().eq("resourced")
    open_rows = pipeline[is_open]

    enriched = []
    for idx, row in open_rows.iterrows():
        required_phrases = scoring.tokenize_skillset(row.get("skillset", ""))
        required_phrases = scoring.enrich_required_phrases(required_phrases, pipeline_skillset)
        if not required_phrases:
            continue
        enriched.append(
            {
                "row_index": int(idx),
                "client": row.get("client") if pd.notna(row.get("client")) else None,
                "resources_requested": row.get("resources_requested"),
                "requested_pct": row.get("requested_pct"),
                "likely_start_date": _fmt_date(row.get("likely_start_date")),
                "status": row.get("status"),
                "priority": row.get("priority"),
                "required_phrases": required_phrases,
                # Pre-split once here instead of inside score_skill_match -- that function
                # re.split()s every phrase on every call, which is fine for one employee
                # but is the dominant cost when scoring hundreds of employees x hundreds
                # of rows (the batch summary below).
                "phrase_tokens": [[t for t in re.split(r"\W+", p.lower()) if len(t) > 2] for p in required_phrases],
                "skill_areas": classify_skillset(row.get("skillset", "")),
            }
        )
    _open_rows_cache = enriched
    _open_rows_fingerprint = fingerprint
    return enriched

def get_redeploy_matches_for_employee(employee_id: str, top_n: int = 20) -> list[dict]:
    """Reverse direction of get_recommendations: for one specific employee, every open
    pipeline deal they could redeploy into, ranked by the same composite_score (skill +
    competency + availability) used everywhere else in the app -- not skill alone, so a
    candidate who's a perfect skill match but already busy or weak on competency doesn't
    outrank someone who's a genuinely better overall fit."""
    adapter = get_adapter()
    skill_index = scoring.build_employee_skill_index(adapter.get_skills())
    employee_tokens = skill_index.get(employee_id, {})
    if not employee_tokens:
        return []

    competency_index = scoring.build_employee_competency_index(adapter.get_competencies())
    competency_entry = competency_index.get(employee_id, {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
    allocations = adapter.get_allocations()

    matches = []
    for row in _open_pipeline_rows_enriched():
        skill_result = scoring.score_skill_match(row["required_phrases"], employee_tokens)
        if skill_result["score"] <= 0:
            continue
        as_of_date = pd.to_datetime(row["likely_start_date"]) if row["likely_start_date"] else pd.Timestamp.now().normalize()
        busy_pct = float(availability_as_of(allocations, as_of_date).get(employee_id, 0.0))
        available_pct = max(0.0, 100.0 - busy_pct)
        availability_score = min(available_pct / 100.0, 1.0)
        composite = scoring.composite_score(skill_result["score"], competency_entry["score"], availability_score)
        matches.append(
            {
                "row_index": row["row_index"],
                "client": row["client"],
                "resources_requested": row["resources_requested"],
                "requested_pct": row["requested_pct"],
                "likely_start_date": row["likely_start_date"],
                "status": row["status"],
                "priority": row["priority"],
                "skill_areas": row["skill_areas"],
                "skill_score": skill_result["score"],
                "matched_skills": skill_result["matched"],
                "missing_skills": skill_result["missing"],
                "skill_confidence": skill_result["confidence"],
                "competency_score": competency_entry["score"],
                "competency_confidence": competency_entry["confidence"],
                "available_pct": round(available_pct, 1),
                "composite_score": composite,
                "bucket": scoring.bucket(skill_result["score"], skill_result["confidence"]),
            }
        )
    matches.sort(key=lambda m: -m["composite_score"])
    return matches[:top_n]

def _fast_skill_score(phrase_tokens_list: list[list[str]], employee_tokens: dict[str, dict]) -> float:
    """Score-only twin of scoring.score_skill_match for the batch summary path -- skips
    building matched/missing lists and re-tokenizing phrases (both already precomputed
    in _open_pipeline_rows_enriched), since the summary only needs the number."""
    if not phrase_tokens_list:
        return 0.5
    weights = []
    for tokens in phrase_tokens_list:
        best_weight = 0.0
        for t in tokens:
            entry = employee_tokens.get(t)
            if entry is not None and entry["weight"] > best_weight:
                best_weight = entry["weight"]
        if best_weight > 0:
            weights.append(best_weight)
    if not weights:
        return 0.0
    return min(sum(weights) / len(phrase_tokens_list), 1.0)

def get_redeploy_summary_for_employees(employee_ids: list[str]) -> dict[str, dict]:
    """Cheap batch variant for the Free Pool table's 'Projects Recommended' column --
    one top match + count per employee, for potentially hundreds of employees at once.
    Uses a single as-of-today availability snapshot shared across everyone (one groupby
    over the whole org, O(1) lookup per employee) instead of get_redeploy_matches_for_employee's
    precise per-deal as-of-likely-start-date check -- exact for "today", a reasonable
    trade for a table-wide preview; the modal drilldown still uses the precise version."""
    adapter = get_adapter()
    skill_index = scoring.build_employee_skill_index(adapter.get_skills())
    competency_index = scoring.build_employee_competency_index(adapter.get_competencies())
    allocations = adapter.get_allocations()
    busy_pct_today = availability_as_of(allocations, pd.Timestamp.now().normalize())
    open_rows = _open_pipeline_rows_enriched()

    summary: dict[str, dict] = {}
    for emp_id in employee_ids:
        employee_tokens = skill_index.get(emp_id, {})
        if not employee_tokens:
            summary[emp_id] = {"recommended_project_count": 0, "top_match": None}
            continue
        competency_entry = competency_index.get(emp_id, {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
        available_pct = max(0.0, 100.0 - float(busy_pct_today.get(emp_id, 0.0)))
        availability_score = min(available_pct / 100.0, 1.0)

        best = None
        count = 0
        for row in open_rows:
            skill_score = _fast_skill_score(row["phrase_tokens"], employee_tokens)
            if skill_score <= 0:
                continue
            count += 1
            composite = scoring.composite_score(skill_score, competency_entry["score"], availability_score)
            if best is None or composite > best["composite_score"]:
                best = {
                    "row_index": row["row_index"],
                    "client": row["client"],
                    "resources_requested": row["resources_requested"],
                    "skill_areas": row["skill_areas"],
                    "skill_score": skill_score,
                    "composite_score": composite,
                }
        summary[emp_id] = {"recommended_project_count": count, "top_match": best}
    return summary

def _safe_str(val) -> str | None:
    """Return None if val is NaN/None/empty, else the stripped string."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s or None


_PRIORITY_RANK: dict[str, int] = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
_STATUS_RANK: dict[str, int] = {"not resourced": 0, "part resourced": 1, "resourced": 2}
_LATE_NOTICE_THRESHOLD_DAYS = 14


def _is_late_notice(likely_start_date, request_received) -> bool | None:
    """Mirrors the computation in the pipeline router — requests with fewer than
    14 days between the received date and the target start are flagged late-notice."""
    try:
        if not pd.notna(likely_start_date) or not pd.notna(request_received):
            return None
        days = (pd.Timestamp(likely_start_date) - pd.Timestamp(request_received)).days
        return bool(days < _LATE_NOTICE_THRESHOLD_DAYS)
    except Exception:
        return None


def list_deals() -> list[dict]:
    """Groups all pipeline rows by deal_id for the project-based recommendation view.
    Rows without a deal_id form singleton groups so nothing is dropped.
    Returns one entry per deal (or solo row) with full metadata for the left panel,
    including every field needed to replicate the By-Role filter set."""
    from collections import defaultdict
    adapter = get_adapter()
    pipeline = adapter.get_pipeline_forecast()

    groups: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for idx, row in pipeline.iterrows():
        deal_id = row.get("deal_id")
        key = f"deal_{int(deal_id)}" if pd.notna(deal_id) else f"solo_{int(idx)}"
        groups[key].append((int(idx), row))

    deals: list[dict] = []
    for deal_key, row_list in groups.items():
        start_dates = [
            row.get("likely_start_date")
            for _, row in row_list
            if pd.notna(row.get("likely_start_date"))
        ]
        earliest_start = _fmt_date(min(start_dates)) if start_dates else None

        priority_vals = [_safe_str(row.get("priority")) for _, row in row_list]
        priority_vals = [p for p in priority_vals if p]
        best_priority: str | None = None
        best_rank = 999
        for p in priority_vals:
            rank = _PRIORITY_RANK.get(p.lower(), 99)
            if rank < best_rank:
                best_rank = rank
                best_priority = p

        status_vals = [_safe_str(row.get("status")) for _, row in row_list]
        status_vals = [s for s in status_vals if s]
        worst_status: str | None = None
        worst_rank = 999
        for s in status_vals:
            rank = _STATUS_RANK.get(s.lower(), 99)
            if rank < worst_rank:
                worst_rank = rank
                worst_status = s

        sow_signed = any(
            (_safe_str(row.get("sow_signed")) or "").lower() == "yes"
            for _, row in row_list
        )
        # A deal is late-notice if ANY role in it has a late-notice start
        is_late = any(
            _is_late_notice(row.get("likely_start_date"), row.get("request_received"))
            for _, row in row_list
        )
        # start_date_confirmed: "Yes" if any role is confirmed
        start_confirmed = any(
            (_safe_str(row.get("start_date_confirmed")) or "").lower() == "yes"
            for _, row in row_list
        )

        # Use the first role's values for fields that are deal-level by nature
        first_row = row_list[0][1]

        roles = [
            {
                "row_index": idx,
                "resources_requested": _safe_str(row.get("resources_requested")),
                "requested_pct": _safe_str(row.get("requested_pct")),
                "skillset": _safe_str(row.get("skillset")),
                "status": _safe_str(row.get("status")),
                "priority": _safe_str(row.get("priority")),
                "likely_start_date": _fmt_date(row.get("likely_start_date")),
                "client_priority": _safe_str(row.get("client_priority")),
                "request_type": _safe_str(row.get("request_type")),
                "deal_stage_hubspot": _safe_str(row.get("deal_stage_hubspot")),
                "start_date_confirmed": _safe_str(row.get("start_date_confirmed")),
                "is_late_notice": _is_late_notice(row.get("likely_start_date"), row.get("request_received")),
            }
            for idx, row in row_list
        ]

        deals.append({
            "deal_key": deal_key,
            "row_indices": [idx for idx, _ in row_list],
            "client": _safe_str(first_row.get("client")),
            "cluster": int(first_row.get("cluster")) if pd.notna(first_row.get("cluster")) else None,
            "solution": _safe_str(first_row.get("solution")),
            "role_count": len(row_list),
            "roles": roles,
            "earliest_start": earliest_start,
            "priority": best_priority,
            "status": worst_status,
            "sow_signed": sow_signed,
            "is_late_notice": is_late,
            "start_date_confirmed": "Yes" if start_confirmed else "No",
            # Deal-level filter fields — taken from the first role (consistent within a deal)
            "client_priority": _safe_str(first_row.get("client_priority")),
            "request_type": _safe_str(first_row.get("request_type")),
            "deal_stage_hubspot": _safe_str(first_row.get("deal_stage_hubspot")),
        })

    deals.sort(key=lambda d: (
        d["earliest_start"] or "9999",
        _PRIORITY_RANK.get((d["priority"] or "").strip().lower(), 99),
    ))
    return deals


def get_project_team_recommendation(row_indices: list[int], top_n: int = 15) -> dict:
    """Greedy conflict-aware team assignment for a set of pipeline roles.

    Processes roles from most-constrained (fewest candidates that meet requested capacity)
    to least-constrained, so that hard-to-fill roles get first pick of the talent pool.
    Tracks remaining capacity per employee and deducts after each assignment so one person
    cannot be double-booked across roles in the same deal.

    Returns per-role assignments and a deal-level coverage summary."""
    if not row_indices:
        return {
            "roles": [],
            "coverage_summary": {"total": 0, "assigned": 0, "hire_signal": 0, "conflict": 0},
        }

    adapter = get_adapter()
    pipeline = adapter.get_pipeline_forecast()

    # Prefetch once — each get_recommendations_for_pipeline_row call reuses this
    prefetched: dict = {
        "employees": adapter.get_employees(),
        "competencies": adapter.get_competencies(),
        "allocations": adapter.get_allocations(),
        "pipeline_skillset": adapter.get_pipeline_skillset(),
        "skill_index": scoring.build_employee_skill_index(adapter.get_skills()),
        "employee_coe_map": get_employee_primary_coe_map(),
        "compute_earliest_availability": False,
    }

    # Fetch all candidates for each role with a large pool so constraint ordering is accurate
    role_data: list[dict] = []
    for row_index in row_indices:
        if row_index < 0 or row_index >= len(pipeline):
            continue
        result = get_recommendations_for_pipeline_row(
            row_index, pipeline=pipeline, top_n=2000, **prefetched
        )
        role_data.append({
            "row_index": row_index,
            "pipeline_row": result["pipeline_row"],
            "all_candidates": result["candidates"],
            "requested_pct": result["request"]["requested_pct"],
            "hire_vs_redeploy_flag": result["hire_vs_redeploy_flag"],
            "has_skillset": result["has_skillset"],
            "fallback_candidates": result.get("fallback_candidates"),
        })

    # Sort roles from most-constrained (fewest capacity-meeting candidates) to least
    def _viable_count(rd: dict) -> int:
        return sum(1 for c in rd["all_candidates"] if c["meets_requested_capacity"])

    constraint_order = sorted(range(len(role_data)), key=lambda i: _viable_count(role_data[i]))

    # Bootstrap remaining_capacity from the first time we see each employee
    remaining_capacity: dict[str, float] = {}
    for rd in role_data:
        for c in rd["all_candidates"]:
            if c["employee_id"] not in remaining_capacity:
                remaining_capacity[c["employee_id"]] = c["available_pct"]

    assigned_map: dict[int, dict | None] = {}
    status_map: dict[int, str] = {}

    for i in constraint_order:
        rd = role_data[i]
        row_index = rd["row_index"]
        req_pct = rd["requested_pct"]

        # Pick the best candidate (already ranked by composite_score desc) with enough capacity
        best: dict | None = None
        for c in rd["all_candidates"]:
            if remaining_capacity.get(c["employee_id"], 0.0) >= req_pct:
                best = c
                break

        if best is not None:
            remaining_capacity[best["employee_id"]] = remaining_capacity[best["employee_id"]] - req_pct
            assigned_map[row_index] = best
            status_map[row_index] = "assigned"
        elif rd["hire_vs_redeploy_flag"]:
            assigned_map[row_index] = None
            status_map[row_index] = "hire_signal"
        else:
            # Internal candidates exist but all are capacity-exhausted by sibling roles
            assigned_map[row_index] = None
            status_map[row_index] = "conflict"

    # Build output: preserve original row order; include top-N candidates for the UI
    output_roles: list[dict] = []
    for rd in role_data:
        row_index = rd["row_index"]
        assigned = assigned_map.get(row_index)
        assigned_id = assigned["employee_id"] if assigned else None

        # Alternatives: best candidates that still have enough remaining capacity, excluding
        # the assigned one (informational — the RM decides whether to swap)
        alternatives = [
            c for c in rd["all_candidates"]
            if c["employee_id"] != assigned_id
            and remaining_capacity.get(c["employee_id"], 0.0) >= rd["requested_pct"]
        ][:5]

        output_roles.append({
            "row_index": row_index,
            "pipeline_row": rd["pipeline_row"],
            "requested_pct": rd["requested_pct"],
            "has_skillset": rd["has_skillset"],
            "hire_vs_redeploy_flag": rd["hire_vs_redeploy_flag"],
            "status": status_map.get(row_index, "conflict"),
            "assigned": assigned,
            "alternatives": alternatives,
            "candidates": rd["all_candidates"][:top_n],
            "fallback_candidates": rd["fallback_candidates"],
        })

    # Restore original row_indices order for the response
    idx_order = {ri: pos for pos, ri in enumerate(row_indices)}
    output_roles.sort(key=lambda r: idx_order.get(r["row_index"], 999))

    total = len(output_roles)
    assigned_count = sum(1 for r in output_roles if r["status"] == "assigned")
    hire_count = sum(1 for r in output_roles if r["status"] == "hire_signal")
    conflict_count = sum(1 for r in output_roles if r["status"] == "conflict")

    return {
        "roles": output_roles,
        "coverage_summary": {
            "total": total,
            "assigned": assigned_count,
            "hire_signal": hire_count,
            "conflict": conflict_count,
        },
    }


def get_coverage_summary() -> dict:
    adapter = get_adapter()
    skill_index = scoring.build_employee_skill_index(adapter.get_skills())
    prefetched = {
        "employees": adapter.get_employees(),
        "competencies": adapter.get_competencies(),
        "allocations": adapter.get_allocations(),
        "pipeline_skillset": adapter.get_pipeline_skillset(),
        "skill_index": skill_index,
        "employee_coe_map": get_employee_primary_coe_map(),
        "compute_earliest_availability": False,
    }
    pipeline = adapter.get_pipeline_forecast()

    rows = []
    for row_index in range(len(pipeline)):
        result = get_recommendations_for_pipeline_row(row_index, pipeline=pipeline, **prefetched)
        top = result["candidates"][0] if result["candidates"] else None
        has_skillset = len(result["request"]["required_phrases"]) > 0
        rows.append(
            {
                "row_index": row_index,
                "client": result["pipeline_row"]["client"],
                "resources_requested": result["pipeline_row"]["resources_requested"],
                "top_candidate_signal": result["top_candidate_signal"] if has_skillset else None,
                "top_bucket": (top["bucket"] if top else "gap") if has_skillset else None,
                "has_skillset": has_skillset,
            }
        )

    total = len(rows)
    no_skillset_count = sum(1 for r in rows if not r["has_skillset"])
    hire_count = sum(1 for r in rows if r["top_candidate_signal"] == "hire")
    redeploy_count = sum(1 for r in rows if r["top_candidate_signal"] == "redeploy")
    training_count = sum(1 for r in rows if r["top_candidate_signal"] == "redeploy_with_training")

    return {
        "total_demand_rows": total,
        "no_skillset_specified_count": no_skillset_count,
        "redeploy_ready_count": redeploy_count,
        "redeploy_with_training_count": training_count,
        "hire_signal_count": hire_count,
        "hire_signal_pct": round(100.0 * hire_count / total, 1) if total else 0.0,
        "rows": rows,
    }
