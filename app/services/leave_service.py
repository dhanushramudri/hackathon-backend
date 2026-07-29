import pandas as pd

from app.core.adapter import get_adapter
from app.engines.coe_skill_engine import GENERIC_SKILL_COES
from app.engines.role_mix_engine import canonical_project_coe
from app.engines import embedding_engine
from app.engines import experience_engine
from app.engines import availability_hold
from app.engines.scoring import (
    DEFAULT_COMPETENCY_SCORE,
    bucket,
    build_employee_competency_index,
    build_employee_skill_index,
    composite_score_v2,
    score_skill_match,
    top_skill_phrases_for_employees,
)
from app.engines.employee_coe import get_employee_primary_coe_map
from app.services.free_pool_service import get_free_pool_by_designation
from app.services.recommendation_service import (
    SKILLSET_CATEGORY_TO_TECH_COE,
    _coe_affinity_rank,
    _COE_AFFINITY_NEUTRAL,
)

MAX_BACKFILL_SHOWN = 5
TOP_N_REQUIRED_SKILLS = 8
# Below this, a roster's skill rows are too sparse to trust as "this project needs
# these skills" -- a single person's idiosyncratic skill list isn't a project
# signature. Falls back to the on-leave person's own skills instead (see below).
MIN_ROSTER_FOR_PROJECT_SKILLS = 2
DEFAULT_NEAR_CAPACITY_TOLERANCE_PCT = 25.0

# Unknown/non-billable rates sort as "most expensive" so a missing rate never
# accidentally wins a cost tiebreak -- same convention as recommendation_service.
_FALLBACK_RATE = float("inf")

def _cost_key(c: dict) -> float:
    rate = c.get("hourly_rate_usd")
    return -(rate if rate is not None else _FALLBACK_RATE)

def _top_skill_phrases(skills_subset: pd.DataFrame, top_n: int) -> list[str]:
    return top_skill_phrases_for_employees(skills_subset, GENERIC_SKILL_COES, top_n)

def get_leave_impact(
    *,
    # All 5 ranking parameters are independently selectable in Advanced Filters --
    # skill/competency/availability default on, the other two default off. Same
    # defaults/semantics as get_recommendations (see scoring.composite_score_v2).
    include_skill: bool = True,
    include_competency: bool = True,
    include_availability: bool = True,
    include_category_match: bool = False,
    include_project_count: bool = False,
    include_coe_affinity: bool = True,
    # Tie-break only (see _cost_key below), off by default -- see
    # recommendation_service.COST_TIE_BAND_PCT precedent.
    include_cost_efficiency: bool = False,
    # Hard pool gate, not a score factor -- default (False) hides candidates who
    # can't actually absorb the vacated allocation_by_percentage right now.
    include_below_capacity: bool = False,
    near_capacity_tolerance_pct: float = DEFAULT_NEAR_CAPACITY_TOLERANCE_PCT,
) -> list[dict]:
    adapter = get_adapter()
    leaves = adapter.get_leaves()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()
    projects = adapter.get_projects()
    skills = adapter.get_skills()
    pool_by_designation = get_free_pool_by_designation()
    tech_coe_by_project = projects.set_index("project_code")["tech_coe"]
    skill_index = build_employee_skill_index(skills)
    competency_index = build_employee_competency_index(adapter.get_competencies())
    emp_embedding_index = embedding_engine.build_employee_embedding_index(skills)
    # Same hold/doubt signal surfaced on every other recommendation-adjacent view
    # (Recommendations candidates, Free Pool, Employees, Employee Profile) -- see
    # app/engines/availability_hold.py. A backfill candidate whose current project
    # is itself at risk of extending is not actually a safe pick.
    hold_flags = availability_hold.get_employee_hold_flags()
    # Experience/track-record layer (category_match / project_count), built once
    # for the whole call -- not per candidate. Leave has no deal "Solution" field,
    # so requested_solution is always None; match_experience() falls back to
    # tech_coe-only matching in that case (see experience_engine.match_experience).
    experience_profiles = experience_engine.build_employee_experience_profiles()
    employee_coe_map = get_employee_primary_coe_map()

    today = pd.Timestamp.now().normalize()
    relevant_leaves = leaves[leaves["leave_end_date"] >= today]
    active_alloc = allocations[allocations["is_allocation_active"] == 1]

    own_skill_phrases_cache: dict[str, list[str]] = {}
    project_skill_phrases_cache: dict[str, tuple[list[str], str]] = {}

    impacts = []
    for _, leave in relevant_leaves.iterrows():
        emp_id = leave["employee_id"]
        emp_row = employees[employees["employee_id"] == emp_id]
        job_name = emp_row["job_name"].iloc[0] if not emp_row.empty else None

        backfill_pool = [c for c in pool_by_designation.get(job_name, []) if c["employee_id"] != emp_id]

        affected = active_alloc[active_alloc["employee_id"] == emp_id]
        for _, alloc in affected.iterrows():
            project_id = alloc["project_id"]
            coe = canonical_project_coe(tech_coe_by_project.get(project_id))
            # Same bridge get_recommendations uses when a deal has no Solution value:
            # map the canonical CoE category into the real tech_coe vocabulary so the
            # experience engine can find a related-project signal. Leave has no deal
            # "Solution" field at all, so requested_solution is always None here --
            # match_experience() falls back to tech_coe-only matching in that case.
            requested_tech_coes = SKILLSET_CATEGORY_TO_TECH_COE.get(coe, []) if coe else []
            vacated_pct = float(alloc["allocation_by_percentage"]) if pd.notna(alloc["allocation_by_percentage"]) else 100.0

            if project_id not in project_skill_phrases_cache:
                roster_ids = active_alloc[
                    (active_alloc["project_id"] == project_id) & (active_alloc["employee_id"] != emp_id)
                ]["employee_id"].unique()
                required_phrases = (
                    _top_skill_phrases(skills[skills["employee_id"].isin(roster_ids)], TOP_N_REQUIRED_SKILLS)
                    if len(roster_ids) >= MIN_ROSTER_FOR_PROJECT_SKILLS
                    else []
                )
                project_skill_phrases_cache[project_id] = (required_phrases, "project_roster" if required_phrases else "")
            required_phrases, required_skill_source = project_skill_phrases_cache[project_id]

            if not required_phrases:
                if emp_id not in own_skill_phrases_cache:
                    own_skill_phrases_cache[emp_id] = _top_skill_phrases(
                        skills[skills["employee_id"] == emp_id], TOP_N_REQUIRED_SKILLS
                    )
                required_phrases = own_skill_phrases_cache[emp_id]
                required_skill_source = "own_skills" if required_phrases else "none"

            # Semantic layer — embed the required skillset once per project, then
            # blend 65% semantic + 35% word for each backfill candidate (same formula
            # as get_recommendations so all candidate surfaces are consistent).
            skillset_text = " | ".join(required_phrases)
            job_vec = embedding_engine.embed_jobspec(skillset_text) if skillset_text else None
            pool_ids = {c["employee_id"] for c in backfill_pool}
            semantic_scores: dict[str, float] = {}
            if emp_embedding_index is not None and job_vec is not None and pool_ids:
                semantic_scores = embedding_engine.batch_cosine_similarity(
                    job_vec, {k: v for k, v in emp_embedding_index.items() if k in pool_ids}
                )

            scored_pool = []
            for c in backfill_pool:
                cand_id = c["employee_id"]
                word_result = score_skill_match(required_phrases, skill_index.get(cand_id, {}))
                sem_score = semantic_scores.get(cand_id)
                if sem_score is not None and required_phrases:
                    blended = 0.65 * sem_score + 0.35 * word_result["score"]
                    confidence = word_result["confidence"]
                    if confidence == "no_match" and sem_score >= 0.35:
                        confidence = "semantic_match"
                    skill_result = {"score": round(min(blended, 1.0), 3), "matched": word_result["matched"], "missing": word_result["missing"], "confidence": confidence}
                else:
                    skill_result = word_result
                comp_entry = competency_index.get(cand_id, {"score": DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
                idle_capacity_pct = c.get("idle_capacity_pct") or 0.0
                avail_score = min(idle_capacity_pct / 100.0, 1.0)
                experience = experience_engine.match_experience(
                    experience_profiles.get(cand_id), None, requested_tech_coes
                )
                comp = composite_score_v2(
                    skill_result["score"], comp_entry["score"], avail_score,
                    experience["relevant_project_ratio"], experience["project_count_score"],
                    include={
                        "skill": include_skill,
                        "competency": include_competency,
                        "availability": include_availability,
                        "category_match": include_category_match,
                        "project_count": include_project_count,
                    },
                )
                coe_affinity_rank = (
                    _coe_affinity_rank(employee_coe_map.get(cand_id), [coe] if coe else None)
                    if include_coe_affinity else _COE_AFFINITY_NEUTRAL
                )
                hold_info = hold_flags.get(cand_id)
                meets_requested_capacity = bool(idle_capacity_pct >= vacated_pct)
                near_capacity = bool(idle_capacity_pct >= vacated_pct - near_capacity_tolerance_pct)
                scored_pool.append({
                    **c,
                    "skill_score": skill_result["score"],
                    "matched_skills": skill_result["matched"],
                    "missing_skills": skill_result["missing"],
                    "skill_confidence": skill_result["confidence"],
                    "skill_bucket": bucket(skill_result["score"], skill_result["confidence"]),
                    "competency_score": comp_entry["score"],
                    "competency_confidence": comp_entry["confidence"],
                    "relevant_project_count": experience["relevant_project_count"],
                    "relevant_project_ratio": experience["relevant_project_ratio"],
                    "total_projects": experience["total_projects"],
                    "distinct_clients": experience["distinct_clients"],
                    "experience_confidence": experience["experience_confidence"],
                    "top_categories": experience["top_categories"],
                    "project_count_score": experience["project_count_score"],
                    "coe_affinity_rank": coe_affinity_rank,
                    "coe_preferred": coe_affinity_rank == _coe_affinity_rank(coe, [coe]) if coe else False,
                    "composite_score": comp,
                    "meets_requested_capacity": meets_requested_capacity,
                    "near_capacity": near_capacity,
                    "on_hold": hold_info is not None,
                    "hold_projects": hold_info["projects"] if hold_info else [],
                })

            # Sort by composite score first, then CoE-affinity / cost-efficiency as
            # tie-breaks -- same tie-break semantics as get_recommendations, applied
            # here on a much smaller (same-designation) pool. Cost efficiency is a
            # near-no-op in practice: every candidate in a designation-scoped backfill
            # pool shares (roughly) the same rate-card band, so it mainly matters when
            # ties actually occur.
            scored_pool.sort(
                key=lambda c: (
                    c["composite_score"] or 0,
                    c["coe_affinity_rank"] if include_coe_affinity else 0,
                    _cost_key(c) if include_cost_efficiency else 0,
                ),
                reverse=True,
            )

            # Hard pool gate (not a score factor): default hides candidates who can't
            # actually absorb the vacated allocation_by_percentage right now. Same
            # near_capacity_tolerance_pct fallback chain as get_recommendations' pool
            # gate -- near-capacity candidates first, then exact/over-capacity, then
            # (only if include_below_capacity) everyone.
            candidates_near_capacity = [c for c in scored_pool if c["near_capacity"]]
            candidates_meeting_capacity = [c for c in scored_pool if c["meets_requested_capacity"]]
            gated_pool = (
                scored_pool if include_below_capacity
                else (candidates_near_capacity or candidates_meeting_capacity or scored_pool)
            )
            top_skill_score = gated_pool[0]["skill_score"] if gated_pool else None

            impacts.append(
                {
                    "employee_id": emp_id,
                    "job_name": job_name,
                    "leave_type": leave["leave_type"],
                    "leave_start_date": leave["leave_start_date"].strftime("%Y-%m-%d"),
                    "leave_end_date": leave["leave_end_date"].strftime("%Y-%m-%d"),
                    "is_currently_on_leave": bool(leave["leave_start_date"] <= today <= leave["leave_end_date"]),
                    "top_backfill_skill_score": top_skill_score,
                    "project_id": project_id,
                    "coe": coe,
                    "allocation_by_percentage": alloc["allocation_by_percentage"],
                    "backfill_candidates": gated_pool[:MAX_BACKFILL_SHOWN],
                    "backfill_available": len(gated_pool) > 0,
                    "backfill_pool_size": len(scored_pool),
                    "required_skills": required_phrases,
                    "required_skill_source": required_skill_source,
                }
            )

    return sorted(impacts, key=lambda i: (not i["is_currently_on_leave"], i["leave_start_date"]))
