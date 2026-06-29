import math

import pandas as pd

from app.core.adapter import get_adapter
from app.engines import scoring
from app.engines.designation_ladder import adjacent_designations
from app.engines.employee_coe import get_employee_primary_coe_map
from app.engines.role_mix_engine import get_role_mix_by_category, get_role_mix_by_coes
from app.services.allocation_report_service import UNDER_UTILIZED_THRESHOLD
from app.services.free_pool_service import get_free_pool_by_designation
from app.services.rate_card_service import get_hourly_rate
from app.services.recommendation_service import availability_as_of

STANDARD_MONTHLY_HOURS = 160
MIN_AVAILABLE_PCT_TO_SURFACE = 100 - UNDER_UTILIZED_THRESHOLD
RECOMMENDED_DATE_SEARCH_DAYS = 180

def get_redeploy_candidates_as_of(designation: str, as_of_date: pd.Timestamp, employees: pd.DataFrame, allocations: pd.DataFrame) -> list[dict]:
    busy_pct = availability_as_of(allocations, as_of_date)
    active_in_role = employees[(employees["account_status"] == 1) & (employees["job_name"] == designation)]

    candidates = []
    for _, emp in active_in_role.iterrows():
        busy = float(busy_pct.get(emp["employee_id"], 0.0))
        available_pct = max(0.0, 100.0 - busy)
        if busy > 0 and available_pct < MIN_AVAILABLE_PCT_TO_SURFACE:
            continue
        candidates.append(
            {
                "employee_id": emp["employee_id"],
                "job_name": designation,
                "department_name": emp.get("department_name"),
                "location": emp.get("location"),
                "reason": "fully_free" if busy == 0 else "under_utilized",
                "project_id": None,
                "current_allocation_pct": round(busy, 1),
                "available_pct_as_of": round(available_pct, 1),
            }
        )
    candidates.sort(key=lambda c: -c["available_pct_as_of"])
    return candidates

def _tag_coe(candidates: list[dict], employee_coe_map: dict[str, str]) -> None:
    for c in candidates:
        c["coe"] = employee_coe_map.get(c["employee_id"])

def _score_candidates(candidates: list[dict], required_skills: list[str], skill_index: dict | None, common_tokens: frozenset) -> None:
    if skill_index is None:
        return
    for c in candidates:
        skill_result = scoring.score_skill_match(required_skills, skill_index.get(c["employee_id"], {}), common_tokens)
        c["skill_score"] = skill_result["score"]
        c["matched_skills"] = skill_result["matched"]
        c["missing_skills"] = skill_result["missing"]
        c["skill_confidence"] = skill_result["confidence"]
    candidates.sort(key=lambda c: -c["skill_score"])

def _find_recommended_start_date(
    designation: str,
    requested_date: pd.Timestamp,
    needed_headcount: int,
    employees: pd.DataFrame,
    allocations: pd.DataFrame,
    required_skills: list[str],
    skill_index: dict | None,
    common_tokens: frozenset,
) -> dict | None:
    ladder = [designation] + [d for d, _ in adjacent_designations(designation)]
    relevant_ids = set(
        employees[(employees["account_status"] == 1) & (employees["job_name"].isin(ladder))]["employee_id"]
    )
    window_end = requested_date + pd.Timedelta(days=RECOMMENDED_DATE_SEARCH_DAYS)
    future_ends = (
        allocations[
            allocations["employee_id"].isin(relevant_ids)
            & (allocations["is_allocation_active"] == 1)
            & (allocations["allocated_end_date"] > requested_date)
            & (allocations["allocated_end_date"] <= window_end)
        ]["allocated_end_date"]
        .dropna()
        .sort_values()
        .unique()
    )

    for end_date in future_ends:
        check_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        fill: list[dict] = []
        for d in ladder:
            pool = get_redeploy_candidates_as_of(d, check_date, employees, allocations)
            _score_candidates(pool, required_skills, skill_index, common_tokens)
            if d != designation:
                if skill_index is None:
                    continue
                pool = [c for c in pool if c.get("skill_score", 0) >= scoring.ELIGIBLE_THRESHOLD]
            for c in pool:
                c["source_designation"] = d
                c["level_offset"] = 0 if d == designation else next(o for dd, o in adjacent_designations(designation) if dd == d)
            fill.extend(pool)
        if len(fill) >= needed_headcount:
            return {
                "recommended_start_date": check_date.strftime("%Y-%m-%d"),
                "proof": (
                    f"{len(fill)} of {needed_headcount} needed {designation} role(s) covered by real allocations "
                    f"ending {pd.Timestamp(end_date).strftime('%Y-%m-%d')} or earlier."
                ),
                "available_then": fill,
            }
    return None

def _resolve_role_mix(spec: dict) -> dict:
    if spec.get("role_mix_overrides"):
        return {
            "role_mix": spec["role_mix_overrides"],
            "sample_size": None,
            "source": "manual_override",
            "matched_project_codes": [],
        }
    if spec.get("category"):
        return get_role_mix_by_category(spec["category"])
    return get_role_mix_by_coes(spec.get("coes") or [], spec.get("type_of_project"))

def get_new_project_forecast(specs: list[dict]) -> dict:
    today = pd.Timestamp.now().normalize()
    today_key = today.strftime("%Y-%m-%d")
    pool_by_designation = get_free_pool_by_designation()
    employee_coe_map = get_employee_primary_coe_map()
    employees_df: pd.DataFrame | None = None
    allocations_df: pd.DataFrame | None = None

    all_required_skills = sorted({s.lower() for spec in specs for s in (spec.get("required_skills") or [])})
    skill_index = None
    common_tokens: frozenset = frozenset()
    if all_required_skills:
        skill_index = scoring.build_employee_skill_index(get_adapter().get_skills())
        common_tokens = scoring.common_skill_tokens(skill_index)

    total_need: dict[tuple[str, str], float] = {}
    duration_weeks_by_date: dict[str, int | None] = {}
    role_mix_sources = []
    for spec in specs:
        result = _resolve_role_mix(spec)
        role_mix_sources.append(
            {
                "spec": spec,
                "source": result["source"],
                "sample_size": result.get("sample_size"),
                "matched_project_codes": result.get("matched_project_codes", []),
            }
        )
        date_key = spec.get("start_date") or today_key
        duration_weeks_by_date.setdefault(date_key, spec.get("duration_weeks"))
        for designation, fte in result["role_mix"].items():
            key = (designation, date_key)
            total_need[key] = total_need.get(key, 0) + fte * spec["count"]

    def _ensure_employee_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
        nonlocal employees_df, allocations_df
        if employees_df is None:
            adapter = get_adapter()
            employees_df = adapter.get_employees()
            allocations_df = adapter.get_allocations()
        return employees_df, allocations_df

    breakdown = []
    for (designation, date_key), needed_fte in sorted(total_need.items(), key=lambda x: -x[1]):
        needed_headcount = math.ceil(needed_fte)

        if date_key == today_key:
            candidates = [dict(c) for c in pool_by_designation.get(designation, [])]
        else:
            emp_df, alloc_df = _ensure_employee_tables()
            candidates = get_redeploy_candidates_as_of(designation, pd.to_datetime(date_key), emp_df, alloc_df)
        _score_candidates(candidates, all_required_skills, skill_index, common_tokens)
        _tag_coe(candidates, employee_coe_map)

        # Holding the exact designation only means availability, not skill fit -- without this
        # gate, asking for a skill nobody has still reports every free person in that title as
        # "covered" (shortfall 0), because shortfall_at_level used to come from raw headcount.
        # The adjacent-level fallback below already requires ELIGIBLE_THRESHOLD; same-level
        # candidates need the identical check for the same reason.
        if skill_index is not None:
            qualifying_candidates = [c for c in candidates if c.get("skill_score", 0) >= scoring.ELIGIBLE_THRESHOLD]
        else:
            qualifying_candidates = candidates

        shortfall_at_level = max(0, needed_headcount - len(qualifying_candidates))
        adjacent_level_candidates: list[dict] = []
        adjacent_fill_count = 0
        if shortfall_at_level > 0:
            for adj_designation, offset in adjacent_designations(designation):
                if date_key == today_key:
                    pool = [dict(c) for c in pool_by_designation.get(adj_designation, [])]
                else:
                    emp_df, alloc_df = _ensure_employee_tables()
                    pool = get_redeploy_candidates_as_of(adj_designation, pd.to_datetime(date_key), emp_df, alloc_df)
                _score_candidates(pool, all_required_skills, skill_index, common_tokens)
                for c in pool:
                    c["source_designation"] = adj_designation
                    c["level_offset"] = offset
                adjacent_level_candidates.extend(pool)
            _tag_coe(adjacent_level_candidates, employee_coe_map)
            adjacent_level_candidates.sort(key=lambda c: (-c.get("skill_score", -1), abs(c["level_offset"])))
            if skill_index is not None:
                qualifying = [c for c in adjacent_level_candidates if c.get("skill_score", 0) >= scoring.ELIGIBLE_THRESHOLD]
                adjacent_fill_count = min(len(qualifying), shortfall_at_level)

        shortfall = max(0, shortfall_at_level - adjacent_fill_count)

        recommended_start_date = None
        recommended_start_date_proof = None
        recommended_available_then: list[dict] = []
        if shortfall > 0:
            emp_df, alloc_df = _ensure_employee_tables()
            found = _find_recommended_start_date(
                designation, pd.to_datetime(date_key), needed_headcount, emp_df, alloc_df, all_required_skills, skill_index, common_tokens
            )
            if found:
                recommended_start_date = found["recommended_start_date"]
                recommended_start_date_proof = found["proof"]
                recommended_available_then = found["available_then"]
                _tag_coe(recommended_available_then, employee_coe_map)

        hourly_rate = get_hourly_rate(designation)
        shortfall_value_usd = round(shortfall * (hourly_rate or 0) * STANDARD_MONTHLY_HOURS, 0)
        full_role_monthly_value_usd = round(needed_headcount * (hourly_rate or 0) * STANDARD_MONTHLY_HOURS, 0)
        achievable_monthly_value_usd = full_role_monthly_value_usd - shortfall_value_usd
        breakdown.append(
            {
                "designation": designation,
                "start_date": date_key,
                "duration_weeks": duration_weeks_by_date.get(date_key),
                "needed_fte": round(needed_fte, 2),
                "needed_headcount": needed_headcount,
                "available_for_redeploy": len(candidates),
                "qualifying_for_redeploy": len(qualifying_candidates),
                "redeploy_candidates": candidates,
                "adjacent_level_candidates": adjacent_level_candidates,
                "adjacent_fill_count": adjacent_fill_count,
                "shortfall": shortfall,
                "shortfall_value_usd": shortfall_value_usd,
                "full_role_monthly_value_usd": full_role_monthly_value_usd,
                "achievable_monthly_value_usd": achievable_monthly_value_usd,
                "hire_signal": shortfall > 0,
                "recommended_start_date": recommended_start_date,
                "recommended_start_date_proof": recommended_start_date_proof,
                "recommended_available_then": recommended_available_then,
            }
        )

    total_full_role_value_usd = sum(b["full_role_monthly_value_usd"] for b in breakdown)
    total_achievable_value_usd = sum(b["achievable_monthly_value_usd"] for b in breakdown)
    pct_achievable_with_current_headcount = (
        round(100 * total_achievable_value_usd / total_full_role_value_usd, 1)
        if total_full_role_value_usd > 0
        else None
    )

    return {
        "specs": specs,
        "role_mix_sources": role_mix_sources,
        "required_skills": all_required_skills,
        "breakdown": breakdown,
        "total_shortfall_headcount": sum(b["shortfall"] for b in breakdown),
        "total_shortfall_value_usd": sum(b["shortfall_value_usd"] for b in breakdown),
        "total_full_role_value_usd": total_full_role_value_usd,
        "total_achievable_value_usd": total_achievable_value_usd,
        "pct_achievable_with_current_headcount": pct_achievable_with_current_headcount,
    }
