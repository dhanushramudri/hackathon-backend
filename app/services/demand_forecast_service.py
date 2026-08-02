import math

import pandas as pd

from app.core.adapter import get_adapter
from app.engines import availability_hold, embedding_engine, experience_engine, scoring
from app.engines.role_hierarchy import LEADERSHIP_DESIGNATIONS, adjacent_designations, same_level_peers
from app.engines.employee_coe import get_employee_primary_coe_map
from app.engines.revenue_engine import (
    DELIVERY_TEMPLATE,
    DND_TEMPLATE,
    DND_TYPICAL_DURATION_WEEKS,
    compute_duration_buckets,
    dnd_revenue_range_for_duration,
    get_revenue_benchmarks_by_coe,
)
from app.engines.role_mix_engine import (
    CANONICAL_COE_MAP,
    DOCX_CATEGORY_MAP,
    get_role_mix_by_category,
    get_role_mix_by_coes,
)
from app.services.allocation_report_service import UNDER_UTILIZED_THRESHOLD
from app.services.rate_card_service import get_hourly_rate
from app.services.recommendation_service import availability_as_of, get_recommendations

STANDARD_MONTHLY_HOURS = 160
MIN_AVAILABLE_PCT_TO_SURFACE = 100 - UNDER_UTILIZED_THRESHOLD

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

def get_top_candidates_for_role(designation: str, as_of_date: str, limit: int = 10) -> list[dict]:
    """Ranked-by-availability shortlist for one designation, as of a date --
    the same real signal get_redeploy_candidates_as_of already uses elsewhere,
    exposed standalone so the Wizard's Resource Allocation step can pre-select
    a real "top guy" per budgeted role instead of leaving every row blank."""
    adapter = get_adapter()
    candidates = get_redeploy_candidates_as_of(
        designation, pd.to_datetime(as_of_date), adapter.get_employees(), adapter.get_allocations()
    )
    return candidates[:limit]

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

    # Cross-role tier prefetch (see below) -- built lazily, once, only if some
    # designation actually still has a shortfall after its own + adjacent-ladder
    # tiers, since it's real extra cost (org-wide skill/embedding scoring via
    # the same engine get_recommendations() uses) that most forecast runs with
    # a healthy bench will never need.
    cross_role_prefetch: dict | None = None

    def _ensure_cross_role_prefetch() -> dict:
        nonlocal cross_role_prefetch
        if cross_role_prefetch is None:
            adapter = get_adapter()
            emp_df, alloc_df = _ensure_employee_tables()
            _skills = adapter.get_skills()
            cross_role_prefetch = {
                "employees": emp_df,
                "competencies": adapter.get_competencies(),
                "allocations": alloc_df,
                "pipeline_skillset": adapter.get_pipeline_skillset(),
                "skills": _skills,
                "skill_index": skill_index or scoring.build_employee_skill_index(_skills),
                "employee_coe_map": employee_coe_map,
                "emp_embedding_index": emp_embedding_index or embedding_engine.build_employee_embedding_index(_skills),
                "experience_profiles": experience_profiles,
                "compute_earliest_availability": False,
                "compute_other_options": False,
            }
        return cross_role_prefetch

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

        # Always availability-as-of the role's own needed_by date (same real
        # mechanism the main Recommendations engine uses) -- there used to be
        # a separate, date-agnostic free-pool shortcut for "needed today"
        # roles that ignored whether a candidate's current allocation(s)
        # actually free them up by the date in question. That let someone
        # over-allocated today, whose commitment doesn't end until well after
        # the role's needed_by date, be scored as if 100% free right now and
        # silently counted toward "covered" -- see get_redeploy_candidates_as_of's
        # own MIN_AVAILABLE_PCT_TO_SURFACE gate, which is the check that's
        # supposed to catch exactly that case.
        emp_df, alloc_df = _ensure_employee_tables()
        # Holding a same-level cross-family title (e.g. "Senior Consultant" for a
        # "Solutions Consultant" request) is the same real coverage signal as
        # holding the literal requested title -- these are the same org level,
        # just named differently in the UK/USA vs India naming conventions.
        same_level_titles = [designation, *same_level_peers(designation)]
        seen_ids: set[str] = set()
        candidates = []
        for title in same_level_titles:
            for c in get_redeploy_candidates_as_of(title, pd.to_datetime(date_key), emp_df, alloc_df):
                if c["employee_id"] in claimed or c["employee_id"] in seen_ids:
                    continue
                seen_ids.add(c["employee_id"])
                candidates.append(c)
        _score_candidates(
            candidates, all_required_skills, skill_index, competency_index, emp_embedding_index,
            include=include, experience_profiles=experience_profiles, requested_tech_coes=requested_tech_coes,
        )
        _tag_coe(candidates, employee_coe_map)

        # Holding the exact requested designation is itself the real headcount-coverage
        # signal here -- a real gap on one specific required skill is trainable, not a
        # hire signal, for a headcount forecast (unlike picking one best-fit candidate,
        # where the Recommendations engine's own skill gate still applies). skill_score
        # still drives composite ranking/sort order via _score_candidates, so a stronger
        # skill match is still surfaced first -- it's just no longer a hard in/out gate
        # for whether someone counts toward the need. (A prior version gated this on
        # skill_score >= ELIGIBLE_THRESHOLD, splitting same-title candidates into a
        # confusing "qualifies" vs "holds the title but doesn't meet the skillset"
        # display -- removed per explicit product decision.)
        qualifying_candidates = candidates

        # Tag every candidate with whether it's actually in qualifying_candidates
        # (not re-derivable from skill_score alone client-side -- leadership
        # designations are exempt from the threshold above, so a frontend
        # reimplementing "skill_score >= 0.6" would wrongly split a Manager
        # with a low skill_score into "doesn't qualify" even though they do).
        qualifying_ids = {c["employee_id"] for c in qualifying_candidates}
        for c in candidates:
            c["meets_requested_skillset"] = c["employee_id"] in qualifying_ids

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
                qualifying_adjacent_ids = {c["employee_id"] for c in qualifying}
            else:
                adjacent_fill_count = min(len(adjacent_level_candidates), shortfall_at_level)
                claimed.update(c["employee_id"] for c in adjacent_level_candidates[:adjacent_fill_count])
                qualifying_adjacent_ids = {c["employee_id"] for c in adjacent_level_candidates}
            # Same reasoning as the same-title tag above -- leadership
            # designations aren't skill-gated here either.
            for c in adjacent_level_candidates:
                c["meets_requested_skillset"] = c["employee_id"] in qualifying_adjacent_ids

        shortfall_after_adjacent = max(0, shortfall_at_level - adjacent_fill_count)

        # Cross-role tier: title-ladder adjacency (above) only catches a different
        # seniority of the SAME track (Software Engineer <-> Senior Software
        # Engineer). "We're short a Solutions Enabler, but is there a Senior
        # Software Engineer whose actual skills fit?" is a different track
        # entirely -- so this runs the exact same org-wide skill/competency/
        # availability search the main Recommendations page uses (get_recommendations,
        # bucketed eligible/trainable/gap via scoring.bucket()), unrestricted by
        # designation.
        #
        # These are surfaced as real, useful context (their skill records
        # genuinely match), but do NOT reduce shortfall or the hire signal --
        # a skill-tag match alone doesn't mean pulling a Consultant off their
        # day job to hands-on-deliver a technical build is a realistic plan.
        # Same-title + adjacent-title (above) is the only pool the shortfall
        # math trusts as a real redeployment option; cross-role match is shown
        # separately so an RM can judge case-by-case whether it's worth
        # pursuing, not have it silently assumed into "covered."
        # `trainable` results are a real skill gap, not headcount, and belong
        # in the training/upskill plan below regardless of shortfall.
        cross_role_candidates: list[dict] = []
        training_candidates: list[dict] = []
        if all_required_skills:
            prefetch = _ensure_cross_role_prefetch()
            cross_role_result = get_recommendations(
                # Comma-separated -- scoring.tokenize_skillset splits on "," / ";" only,
                # never on " | ". Joining with " | " (a prior bug) made every requested
                # skill collapse into one giant undividable phrase, so missing_skills
                # came back permanently empty and matched_skills was always one blob
                # covering all of them -- no real per-skill signal ever reached the
                # Cross-Role Match / Trainable tabs.
                skillset_text=", ".join(all_required_skills),
                likely_start_date=date_key,
                requested_pct_raw="100",
                top_n=200,
                **prefetch,
            )
            ladder_ids = {c["employee_id"] for c in candidates} | {c["employee_id"] for c in adjacent_level_candidates}
            for c in cross_role_result["candidates"]:
                if c["employee_id"] in claimed or c["employee_id"] in ladder_ids:
                    continue
                if c["bucket"] == "eligible":
                    cross_role_candidates.append(c)
                elif c["bucket"] == "trainable":
                    training_candidates.append(c)

        shortfall = shortfall_after_adjacent

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
                "cross_role_candidates": cross_role_candidates,
                "cross_role_match_count": len(cross_role_candidates),
                "training_candidates": training_candidates,
                "shortfall": shortfall,
                "shortfall_value_usd": shortfall_value_usd,
                "full_role_monthly_value_usd": full_role_monthly_value_usd,
                "achievable_monthly_value_usd": achievable_monthly_value_usd,
                "hire_signal": shortfall > 0,
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

# Priority weight per rank in a priority_coes list -- halves each step so the
# top-priority CoE (Data Engineering by default, reflecting where JMAN is
# strongest/most proven) claims roughly half the revenue target, the next
# claims half of what's left, and so on. The last CoE in the list absorbs
# whatever weight remains instead of being halved again, so weights always
# sum to exactly 1.0 regardless of list length.
def _priority_weights(n: int) -> list[float]:
    if n <= 0:
        return []
    weights = [0.5 ** (i + 1) for i in range(n - 1)]
    weights.append(1.0 - sum(weights))
    return weights

# Below this fraction of a CoE's own average project size, a "project" would
# be a sliver too small to be a real, staffable engagement -- skipped rather
# than reporting a nonsensical "0.1 projects".
MIN_PROJECT_SHARE_FRACTION = 0.15

def get_revenue_target_forecast(
    target_revenue_usd: float,
    priority_coes: list[str] | None = None,
    start_date: str | None = None,
    duration_weeks: int | None = None,
    type_of_project: str | None = None,
    include: dict[str, bool] | None = None,
    duration_mix: dict[str, float] | None = None,
    # No real D&D-to-delivery conversion rate exists in this org's data (see
    # the D&D block below), so this is a user-supplied assumption, not a
    # fitted figure -- 100% (1 D&D per delivery project, the old fixed
    # behavior) unless the RM overrides it via the editable win-rate input.
    dnd_win_rate_pct: float | None = None,
    # Optional -- when set, adds a plain timeline feasibility check (real
    # typical project duration vs. real weeks remaining until this date). Not
    # a hiring-lead-time projection -- there's no real data in this org to
    # fit one, so a staffing gap is surfaced as a fact (see forecast.breakdown
    # /pct_achievable_with_current_headcount), never a fabricated "ready by" date.
    target_date: str | None = None,
) -> dict:
    """Reverse the usual forecast question: given a revenue target, what project
    mix (how many of each CoE) gets us there, prioritized toward the CoEs we're
    strongest in -- and, feeding that mix through the same get_new_project_forecast
    pipeline (redeploy -> adjacent title -> cross-role skill match -> train ->
    hire), what resources does that mix actually require, what do we already
    have, and where's the real gap.

    Revenue figures and the role mix per project are grounded in real JMAN
    delivery-project economics (see app/engines/revenue_engine.py's
    DELIVERY_TEMPLATE) -- ~$35k revenue, ~5 weeks, 2 engineers + 1 Solutions
    Enabler + 1 Consultant -- not derived from the source data's real
    allocation hours (which pool multi-month, multi-phase engagements under
    one project_code and don't correspond to a single clean "project" unit).
    $35k is a RATE anchored to 5 weeks, not a flat per-project constant -- a
    longer requested duration_weeks scales the per-project revenue (and the
    D&D estimate below) proportionally.
    """
    # duration_mix (from the frontend's Short/Mid/Long slider) overrides a
    # plain duration_weeks input -- it's a weighted blend of the real
    # historical duration buckets rather than one flat number, but feeds into
    # exactly the same duration_weeks-scaled revenue math below.
    effective_duration_weeks = duration_weeks
    if duration_mix:
        buckets = compute_duration_buckets()["buckets"]
        if buckets:
            effective_duration_weeks = sum(
                duration_mix.get(b, 0) * buckets[b]["avg_weeks"] for b in ("short", "mid", "long") if b in buckets
            )
    duration_weeks = effective_duration_weeks

    benchmarks = get_revenue_benchmarks_by_coe(duration_weeks)
    if not benchmarks:
        return {
            "target_revenue_usd": target_revenue_usd,
            "priority_coes": [],
            "project_mix": [],
            "total_projected_revenue_usd": 0.0,
            "revenue_gap_usd": target_revenue_usd,
            "pct_of_target_covered": 0.0,
            "forecast": None,
            "design_and_discovery": None,
            "timeline": None,
            "revenue_hit_estimate": None,
            "error": "No historical project revenue data available to benchmark against.",
        }

    if not priority_coes:
        priority_coes = ["Data Engineering"] + sorted(
            (c for c in benchmarks if c != "Data Engineering"),
            key=lambda c: -benchmarks[c]["avg_revenue_per_project"],
        )
    priority_coes = [c for c in priority_coes if c in benchmarks]

    # Clamped to [1, 100] -- 0% would make "D&D needed" divide-by-zero/infinite,
    # and >100% isn't a meaningful win rate.
    dnd_win_rate = max(1.0, min(100.0, dnd_win_rate_pct if dnd_win_rate_pct is not None else 100.0)) / 100.0

    weights = _priority_weights(len(priority_coes))
    project_mix = []
    specs = []
    for coe, weight in zip(priority_coes, weights):
        b = benchmarks[coe]
        avg = b["avg_revenue_per_project"]
        target_share = target_revenue_usd * weight
        project_count = round(target_share / avg) if avg > 0 else 0
        if project_count <= 0 and avg > 0 and target_share >= avg * MIN_PROJECT_SHARE_FRACTION:
            project_count = 1
        projected_revenue = project_count * avg
        # How many D&D engagements it takes to land project_count real delivery
        # projects at the (user-set) win rate -- e.g. 50% win rate needs 2 D&Ds
        # per delivery project actually won.
        dnd_engagements_needed = math.ceil(project_count / dnd_win_rate) if project_count > 0 else 0
        project_mix.append({
            "coe": coe,
            "weight_pct": round(weight * 100, 1),
            "target_share_usd": round(target_share, 0),
            "project_count": project_count,
            "avg_revenue_per_project": avg,
            "projected_revenue_usd": projected_revenue,
            "sample_size": b["sample_size"],
            "dnd_engagements_needed": dnd_engagements_needed,
        })
        if project_count > 0:
            specs.append({
                "coes": [coe],
                "count": project_count,
                "start_date": start_date,
                "duration_weeks": duration_weeks or DELIVERY_TEMPLATE["duration_weeks"],
                "type_of_project": type_of_project,
                # Real delivery-team template (2 engineers + 1 Solutions
                # Enabler + 1 Consultant) instead of the empirical per-CoE
                # role mix, which is built from the same over-broad real
                # project_code groupings the revenue figure above replaced.
                "role_mix_overrides": DELIVERY_TEMPLATE["role_mix"],
            })

    forecast = get_new_project_forecast(specs, include=include) if specs else None
    total_projected_revenue = sum(m["projected_revenue_usd"] for m in project_mix)
    total_delivery_projects = sum(m["project_count"] for m in project_mix)
    total_dnd_engagements_needed = sum(m["dnd_engagements_needed"] for m in project_mix)

    # D&D is the real precursor phase most delivery projects go through before
    # a client commits -- surfaced as prerequisite context, scaled by the
    # editable win rate above (100% = old fixed 1:1 behavior) since no real
    # D&D-to-delivery conversion rate exists in this org's data to fit one.
    dnd_duration = DND_TYPICAL_DURATION_WEEKS
    dnd_revenue_low, dnd_revenue_high = dnd_revenue_range_for_duration(dnd_duration)
    design_and_discovery = (
        {
            "engagements_needed": total_dnd_engagements_needed,
            "win_rate_pct": round(dnd_win_rate * 100, 1),
            "duration_weeks": dnd_duration,
            "revenue_usd_low": dnd_revenue_low,
            "revenue_usd_high": dnd_revenue_high,
            "total_revenue_usd_low": total_dnd_engagements_needed * dnd_revenue_low,
            "total_revenue_usd_high": total_dnd_engagements_needed * dnd_revenue_high,
            "role_mix": DND_TEMPLATE["role_mix"],
            "note": (
                f"Based on a {round(dnd_win_rate * 100)}% D&D-to-delivery win rate (editable, not a fitted "
                f"figure -- no real conversion data exists) -- {total_delivery_projects} delivery project(s) "
                f"need {total_dnd_engagements_needed} D&D engagement(s) at that rate. Clients typically commit "
                "to a delivery project only after a paid D&D engagement -- not included in the revenue totals above."
            ),
        }
        if total_delivery_projects > 0
        else None
    )

    # Plain timeline feasibility check -- real typical project duration vs.
    # real weeks remaining until the target date. Deliberately NOT a hiring-
    # lead-time projection (see the param docstring above): a staffing gap is
    # reported as a fact via forecast.pct_achievable_with_current_headcount,
    # never smoothed into a fabricated "ready by" date.
    timeline = None
    if target_date and duration_weeks:
        today = pd.Timestamp.now().normalize()
        target_ts = pd.to_datetime(target_date).normalize()
        weeks_available = max((target_ts - today).days / 7, 0)
        timeline = {
            "target_date": target_ts.strftime("%Y-%m-%d"),
            "weeks_available": round(weeks_available, 1),
            "typical_project_weeks": round(duration_weeks, 1),
            "likely_fits": weeks_available >= duration_weeks,
        }

    # Plain calendar estimate for "when do we actually see this revenue" --
    # start_date_used (real user input, or today if not set) + the same
    # effective_duration_weeks already used for the revenue math above,
    # assuming every project in the mix starts together and runs in parallel
    # (not sequentially -- a fully-staffed team works them concurrently, it
    # doesn't queue them). This is NOT a hiring-lead-time projection: if
    # forecast still shows a shortfall, the date is surfaced with an explicit
    # "assumes_full_staffing" caveat rather than pushed out by a fabricated
    # hire-by date (no real hiring-lead-time data exists in this org -- see
    # the target_date param docstring above).
    revenue_hit_estimate = None
    if total_delivery_projects > 0:
        # Same fallback the revenue math itself uses (delivery_revenue_for_duration)
        # when no explicit duration_weeks/duration_mix was given -- the real
        # $35k/5-week anchor, not a fresh assumption invented here.
        effective_weeks_for_date = duration_weeks if duration_weeks and duration_weeks > 0 else DELIVERY_TEMPLATE["duration_weeks"]
        start_ts = pd.to_datetime(start_date).normalize() if start_date else pd.Timestamp.now().normalize()
        hit_date = start_ts + pd.Timedelta(weeks=effective_weeks_for_date)
        shortfall = forecast.get("total_shortfall_headcount", 0) if forecast else 0
        revenue_hit_estimate = {
            "start_date_used": start_ts.strftime("%Y-%m-%d"),
            "hit_date": hit_date.strftime("%Y-%m-%d"),
            "duration_weeks": round(effective_weeks_for_date, 1),
            "project_count": total_delivery_projects,
            "has_staffing_gap": shortfall > 0,
            "shortfall_headcount": shortfall,
        }

    return {
        "target_revenue_usd": target_revenue_usd,
        "priority_coes": priority_coes,
        "project_mix": project_mix,
        "total_projected_revenue_usd": total_projected_revenue,
        "revenue_gap_usd": round(target_revenue_usd - total_projected_revenue, 0),
        "pct_of_target_covered": round(100 * total_projected_revenue / target_revenue_usd, 1) if target_revenue_usd > 0 else None,
        "forecast": forecast,
        "design_and_discovery": design_and_discovery,
        "effective_duration_weeks": round(duration_weeks, 1) if duration_weeks else None,
        "timeline": timeline,
        "revenue_hit_estimate": revenue_hit_estimate,
    }
