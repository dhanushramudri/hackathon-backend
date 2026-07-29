import math

import pandas as pd

from app.core.adapter import get_adapter
from app.engines import availability_hold, embedding_engine, experience_engine, scoring
from app.engines.designation_ladder import LEADERSHIP_DESIGNATIONS, adjacent_designations
from app.engines.employee_coe import get_employee_primary_coe_map
from app.engines.role_mix_engine import CANONICAL_COE_MAP, DOCX_CATEGORY_MAP, get_role_mix_by_category, get_role_mix_by_coes
from app.services.allocation_report_service import UNDER_UTILIZED_THRESHOLD
from app.services.free_pool_service import get_free_pool_by_designation
from app.services.rate_card_service import get_hourly_rate
from app.services.recommendation_service import availability_as_of

STANDARD_MONTHLY_HOURS = 160
MIN_AVAILABLE_PCT_TO_SURFACE = 100 - UNDER_UTILIZED_THRESHOLD
RECOMMENDED_DATE_SEARCH_DAYS = 180

# Same 5-parameter ranking flexibility as get_recommendations (see
# scoring.composite_score_v2 / BASE_WEIGHTS) -- one selection for the whole
# forecast run (all specs in the same request share one Advanced Filters
# panel), since role-mix candidates are already pooled/scored per designation
# rather than per spec.
DEFAULT_FORECAST_INCLUDE: dict[str, bool] = {
    "skill": True, "competency": True, "availability": True,
    "category_match": False, "project_count": False,
}

def get_redeploy_candidates_as_of(designation: str, as_of_date: pd.Timestamp, employees: pd.DataFrame, allocations: pd.DataFrame) -> list[dict]:
    busy_pct = availability_as_of(allocations, as_of_date)
    active_in_role = employees[(employees["account_status"] == 1) & (employees["job_name"] == designation)]
    # Same hold/doubt signal surfaced everywhere else a candidate is shown (see
    # app/engines/availability_hold.py) -- someone who looks free today may still
    # be tied to a project the Health monitor expects to run long.
    hold_flags = availability_hold.get_employee_hold_flags()

    candidates = []
    for _, emp in active_in_role.iterrows():
        emp_id = emp["employee_id"]
        busy = float(busy_pct.get(emp_id, 0.0))
        available_pct = max(0.0, 100.0 - busy)
        if busy > 0 and available_pct < MIN_AVAILABLE_PCT_TO_SURFACE:
            continue
        hold_info = hold_flags.get(emp_id)
        candidates.append(
            {
                "employee_id": emp_id,
                "job_name": designation,
                "department_name": emp.get("department_name"),
                "location": emp.get("location"),
                "reason": "fully_free" if busy == 0 else "under_utilized",
                "project_id": None,
                "current_allocation_pct": round(busy, 1),
                "available_pct_as_of": round(available_pct, 1),
                "on_hold": hold_info is not None,
                "hold_projects": hold_info["projects"] if hold_info else [],
            }
        )
    candidates.sort(key=lambda c: -c["available_pct_as_of"])
    return candidates

def _tag_coe(candidates: list[dict], employee_coe_map: dict[str, str]) -> None:
    for c in candidates:
        c["coe"] = employee_coe_map.get(c["employee_id"])

def _score_candidates(
    candidates: list[dict], required_skills: list[str], skill_index: dict | None, competency_index: dict | None = None,
    emp_embedding_index: dict | None = None,
    # Same 5-parameter ranking flexibility as get_recommendations -- see
    # scoring.composite_score_v2. One selection per forecast run (see
    # DEFAULT_FORECAST_INCLUDE above).
    include: dict[str, bool] | None = None,
    experience_profiles: dict | None = None,
    requested_tech_coes: list[str] | None = None,
) -> None:
    if skill_index is None:
        return
    include = include or DEFAULT_FORECAST_INCLUDE
    # Semantic layer — embed required_skills once, batch cosine-sim (same 65/35 blend as /recommendations)
    semantic_scores: dict[str, float] = {}
    if emp_embedding_index is not None and required_skills:
        job_vec = embedding_engine.embed_jobspec(" | ".join(required_skills))
        if job_vec is not None:
            pool_ids = {c["employee_id"] for c in candidates}
            semantic_scores = embedding_engine.batch_cosine_similarity(
                job_vec, {k: v for k, v in emp_embedding_index.items() if k in pool_ids}
            )
    for c in candidates:
        word_result = scoring.score_skill_match(required_skills, skill_index.get(c["employee_id"], {}))
        sem_score = semantic_scores.get(c["employee_id"])
        if sem_score is not None and required_skills:
            blended = 0.65 * sem_score + 0.35 * word_result["score"]
            confidence = word_result["confidence"]
            if confidence == "no_match" and sem_score >= 0.35:
                confidence = "semantic_match"
            skill_result = {"score": round(min(blended, 1.0), 3), "matched": word_result["matched"], "missing": word_result["missing"], "confidence": confidence}
        else:
            skill_result = word_result
        c["skill_score"] = skill_result["score"]
        c["matched_skills"] = skill_result["matched"]
        c["missing_skills"] = skill_result["missing"]
        c["skill_confidence"] = skill_result["confidence"]
        if competency_index is not None:
            comp_entry = competency_index.get(c["employee_id"], {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
            avail_score = min((c.get("available_pct_as_of") or c.get("idle_capacity_pct") or 0.0) / 100.0, 1.0)
            c["competency_score"] = comp_entry["score"]
            experience = experience_engine.match_experience(
                (experience_profiles or {}).get(c["employee_id"]), None, requested_tech_coes or []
            )
            c["relevant_project_count"] = experience["relevant_project_count"]
            c["relevant_project_ratio"] = experience["relevant_project_ratio"]
            c["total_projects"] = experience["total_projects"]
            c["distinct_clients"] = experience["distinct_clients"]
            c["experience_confidence"] = experience["experience_confidence"]
            c["top_categories"] = experience["top_categories"]
            c["project_count_score"] = experience["project_count_score"]
            c["composite_score"] = scoring.composite_score_v2(
                skill_result["score"], comp_entry["score"], avail_score,
                experience["relevant_project_ratio"], experience["project_count_score"],
                include=include,
            )
    if competency_index is not None:
        candidates.sort(key=lambda c: -c.get("composite_score", 0))
    else:
        candidates.sort(key=lambda c: -c["skill_score"])

def _find_recommended_start_date(
    designation: str,
    requested_date: pd.Timestamp,
    needed_headcount: int,
    employees: pd.DataFrame,
    allocations: pd.DataFrame,
    required_skills: list[str],
    skill_index: dict | None,
    competency_index: dict | None = None,
    emp_embedding_index: dict | None = None,
    include: dict[str, bool] | None = None,
    experience_profiles: dict | None = None,
    requested_tech_coes: list[str] | None = None,
    # Same date_key-scoped claimed-employee set as get_new_project_forecast's
    # main loop -- someone already committed to a concurrent role at this same
    # requested_date shouldn't also be projected as this role's future fill.
    claimed_ids: set[str] | None = None,
) -> dict | None:
    claimed_ids = claimed_ids or set()
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
            pool = [c for c in get_redeploy_candidates_as_of(d, check_date, employees, allocations) if c["employee_id"] not in claimed_ids]
            _score_candidates(
                pool, required_skills, skill_index, competency_index, emp_embedding_index,
                include=include, experience_profiles=experience_profiles, requested_tech_coes=requested_tech_coes,
            )
            if d != designation:
                if skill_index is None:
                    continue
                if designation not in LEADERSHIP_DESIGNATIONS:
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

def get_new_project_forecast(specs: list[dict], include: dict[str, bool] | None = None) -> dict:
    include = include or DEFAULT_FORECAST_INCLUDE
    today = pd.Timestamp.now().normalize()
    today_key = today.strftime("%Y-%m-%d")
    pool_by_designation = get_free_pool_by_designation()
    employee_coe_map = get_employee_primary_coe_map()
    employees_df: pd.DataFrame | None = None
    allocations_df: pd.DataFrame | None = None
    # Experience/track-record layer (category_match / project_count), built once
    # for the whole call -- not per candidate/spec. Each spec's `coes` selection
    # (expanded through the same CANONICAL_COE_MAP get_role_mix_by_coes() uses)
    # becomes the requested_tech_coes for every designation it contributes need
    # to -- role-mix candidates are pooled per designation, not per spec, so this
    # is request-level granularity same as all_required_skills below.
    experience_profiles = experience_engine.build_employee_experience_profiles()
    tech_coes_by_key: dict[tuple[str, str], set[str]] = {}

    all_required_skills = sorted({s.lower() for spec in specs for s in (spec.get("required_skills") or [])})
    skill_index = None
    competency_index = None
    emp_embedding_index = None
    if all_required_skills:
        adapter = get_adapter()
        _skills_df = adapter.get_skills()
        skill_index = scoring.build_employee_skill_index(_skills_df)
        competency_index = scoring.build_employee_competency_index(adapter.get_competencies())
        emp_embedding_index = embedding_engine.build_employee_embedding_index(_skills_df)

    total_need: dict[tuple[str, str], float] = {}
    duration_weeks_by_date: dict[str, int | None] = {}
    role_mix_sources = []
    excluded_rare_roles: dict[str, dict] = {}
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
        spec_tech_coes = {alias for coe in (spec.get("coes") or []) for alias in CANONICAL_COE_MAP.get(coe, [coe])}
        # "Quick-fill from a project category" specs never carry `coes` (frontend
        # sends coes=undefined whenever category is set -- see toForecastSpec in
        # forecast/new-project/page.tsx), so without this fallback every
        # category-based spec silently got zero category_match/project_count
        # signal no matter what Advanced Filters selected. DOCX_CATEGORY_MAP's
        # tech_coe_any is already raw tech_coe vocabulary -- no further mapping
        # needed (unlike CANONICAL_COE_MAP above, which bridges canonical CoE
        # names to that same vocabulary for the `coes` case).
        if not spec_tech_coes and spec.get("category"):
            spec_tech_coes = set(DOCX_CATEGORY_MAP.get(spec["category"], {}).get("tech_coe_any", []))
        # role_mix carries every designation ever seen on a past project, even ones that
        # showed up on a single one-off engagement (prevalence_pct in the low single
        # digits). Rounding even a 5% historical fte need up to "you must hire 1 of
        # these" for a brand-new project massively overstates the real ask -- only the
        # empirically common roles (the same >=40% prevalence bar the role-mix preview
        # already uses) count toward real headcount need here. Rare ones are tracked
        # and surfaced separately instead of silently dropped.
        common_by_designation = {r["designation"]: r["common"] for r in result.get("roles", [])}
        for designation, fte in result["role_mix"].items():
            if not common_by_designation.get(designation, True):
                prior = excluded_rare_roles.get(designation)
                prevalence = next((r["prevalence_pct"] for r in result["roles"] if r["designation"] == designation), None)
                excluded_rare_roles[designation] = {
                    "designation": designation,
                    "prevalence_pct": prevalence,
                    "fte": round((prior["fte"] if prior else 0) + fte * spec["count"], 2),
                }
                continue
            key = (designation, date_key)
            total_need[key] = total_need.get(key, 0) + fte * spec["count"]
            if spec_tech_coes:
                tech_coes_by_key.setdefault(key, set()).update(spec_tech_coes)

    def _ensure_employee_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
        nonlocal employees_df, allocations_df
        if employees_df is None:
            adapter = get_adapter()
            employees_df = adapter.get_employees()
            allocations_df = adapter.get_allocations()
        return employees_df, allocations_df

    # Designations sharing an adjacency relation (e.g. "Software Engineer" <->
    # "Senior Software Engineer") can otherwise each independently draw the same
    # physical people as an adjacent-level fill for two different roles' shortfalls
    # in the same forecast run -- silently double-counting real capacity when
    # multiple specs need adjacent titles at the same date_key. Tracked per
    # date_key (not globally) so a person claimed for a role starting today
    # doesn't wrongly block a different role starting months later, when they'd
    # be free again.
    claimed_ids_by_date: dict[str, set[str]] = {}

    breakdown = []
    for (designation, date_key), needed_fte in sorted(total_need.items(), key=lambda x: -x[1]):
        needed_headcount = math.ceil(needed_fte)
        requested_tech_coes = list(tech_coes_by_key.get((designation, date_key), []))
        claimed = claimed_ids_by_date.setdefault(date_key, set())

        if date_key == today_key:
            candidates = [dict(c) for c in pool_by_designation.get(designation, []) if c["employee_id"] not in claimed]
        else:
            emp_df, alloc_df = _ensure_employee_tables()
            candidates = [
                c for c in get_redeploy_candidates_as_of(designation, pd.to_datetime(date_key), emp_df, alloc_df)
                if c["employee_id"] not in claimed
            ]
        _score_candidates(
            candidates, all_required_skills, skill_index, competency_index, emp_embedding_index,
            include=include, experience_profiles=experience_profiles, requested_tech_coes=requested_tech_coes,
        )
        _tag_coe(candidates, employee_coe_map)

        # Holding the exact designation only means availability, not skill fit -- without this
        # gate, asking for a skill nobody has still reports every free person in that title as
        # "covered" (shortfall 0), because shortfall_at_level used to come from raw headcount.
        # The adjacent-level fallback below already requires ELIGIBLE_THRESHOLD; same-level
        # candidates need the identical check for the same reason. Leadership designations
        # (Manager/Principal/Associate Partner/Partner) are exempt -- they're oversight roles,
        # not hands-on ICs, so a missing technical skill shouldn't read as "need to hire one".
        if skill_index is not None and designation not in LEADERSHIP_DESIGNATIONS:
            qualifying_candidates = [c for c in candidates if c.get("skill_score", 0) >= scoring.ELIGIBLE_THRESHOLD]
        else:
            qualifying_candidates = candidates

        # Candidates are already best-first (sorted by composite/skill score in
        # _score_candidates, or by availability if unscored) -- claim exactly the
        # ones this designation's own need would actually draw on, so a sibling
        # designation's adjacent-level fallback below can't also count them.
        own_fill_count = min(len(qualifying_candidates), needed_headcount)
        claimed.update(c["employee_id"] for c in qualifying_candidates[:own_fill_count])

        shortfall_at_level = max(0, needed_headcount - len(qualifying_candidates))
        adjacent_level_candidates: list[dict] = []
        adjacent_fill_count = 0
        if shortfall_at_level > 0:
            for adj_designation, offset in adjacent_designations(designation):
                if date_key == today_key:
                    pool = [dict(c) for c in pool_by_designation.get(adj_designation, []) if c["employee_id"] not in claimed]
                else:
                    emp_df, alloc_df = _ensure_employee_tables()
                    pool = [
                        c for c in get_redeploy_candidates_as_of(adj_designation, pd.to_datetime(date_key), emp_df, alloc_df)
                        if c["employee_id"] not in claimed
                    ]
                _score_candidates(
                    pool, all_required_skills, skill_index, competency_index, emp_embedding_index,
                    include=include, experience_profiles=experience_profiles, requested_tech_coes=requested_tech_coes,
                )
                for c in pool:
                    c["source_designation"] = adj_designation
                    c["level_offset"] = offset
                adjacent_level_candidates.extend(pool)
            _tag_coe(adjacent_level_candidates, employee_coe_map)
            adjacent_level_candidates.sort(key=lambda c: (-c.get("skill_score", -1), abs(c["level_offset"])))
            if skill_index is not None and designation not in LEADERSHIP_DESIGNATIONS:
                qualifying = [c for c in adjacent_level_candidates if c.get("skill_score", 0) >= scoring.ELIGIBLE_THRESHOLD]
                adjacent_fill_count = min(len(qualifying), shortfall_at_level)
                claimed.update(c["employee_id"] for c in qualifying[:adjacent_fill_count])
            else:
                adjacent_fill_count = min(len(adjacent_level_candidates), shortfall_at_level)
                claimed.update(c["employee_id"] for c in adjacent_level_candidates[:adjacent_fill_count])

        shortfall = max(0, shortfall_at_level - adjacent_fill_count)

        recommended_start_date = None
        recommended_start_date_proof = None
        recommended_available_then: list[dict] = []
        if shortfall > 0:
            emp_df, alloc_df = _ensure_employee_tables()
            found = _find_recommended_start_date(
                designation, pd.to_datetime(date_key), needed_headcount, emp_df, alloc_df, all_required_skills, skill_index, competency_index, emp_embedding_index,
                include=include, experience_profiles=experience_profiles, requested_tech_coes=requested_tech_coes,
                claimed_ids=claimed,
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
        "excluded_rare_roles": sorted(excluded_rare_roles.values(), key=lambda r: -(r["prevalence_pct"] or 0)),
        "total_shortfall_headcount": sum(b["shortfall"] for b in breakdown),
        "total_shortfall_value_usd": sum(b["shortfall_value_usd"] for b in breakdown),
        "total_full_role_value_usd": total_full_role_value_usd,
        "total_achievable_value_usd": total_achievable_value_usd,
        "pct_achievable_with_current_headcount": pct_achievable_with_current_headcount,
    }
