import pandas as pd

from app.core.adapter import get_adapter
from app.engines.coe_skill_engine import GENERIC_SKILL_COES
from app.engines.role_mix_engine import canonical_project_coe
from app.engines import embedding_engine
from app.engines.scoring import (
    DEFAULT_COMPETENCY_SCORE,
    bucket,
    build_employee_competency_index,
    build_employee_skill_index,
    composite_score,
    score_skill_match,
    top_skill_phrases_for_employees,
)
from app.services.free_pool_service import get_free_pool_by_designation

MAX_BACKFILL_SHOWN = 5
TOP_N_REQUIRED_SKILLS = 8
# Below this, a roster's skill rows are too sparse to trust as "this project needs
# these skills" -- a single person's idiosyncratic skill list isn't a project
# signature. Falls back to the on-leave person's own skills instead (see below).
MIN_ROSTER_FOR_PROJECT_SKILLS = 2

def _top_skill_phrases(skills_subset: pd.DataFrame, top_n: int) -> list[str]:
    return top_skill_phrases_for_employees(skills_subset, GENERIC_SKILL_COES, top_n)

def get_leave_impact() -> list[dict]:
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
                emp_id = c["employee_id"]
                word_result = score_skill_match(required_phrases, skill_index.get(emp_id, {}))
                sem_score = semantic_scores.get(emp_id)
                if sem_score is not None and required_phrases:
                    blended = 0.65 * sem_score + 0.35 * word_result["score"]
                    confidence = word_result["confidence"]
                    if confidence == "no_match" and sem_score >= 0.35:
                        confidence = "semantic_match"
                    skill_result = {"score": round(min(blended, 1.0), 3), "matched": word_result["matched"], "missing": word_result["missing"], "confidence": confidence}
                else:
                    skill_result = word_result
                comp_entry = competency_index.get(emp_id, {"score": DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
                avail_score = min((c.get("idle_capacity_pct") or 0.0) / 100.0, 1.0)
                comp = composite_score(skill_result["score"], comp_entry["score"], avail_score)
                scored_pool.append({
                    **c,
                    "skill_score": skill_result["score"],
                    "matched_skills": skill_result["matched"],
                    "missing_skills": skill_result["missing"],
                    "skill_confidence": skill_result["confidence"],
                    "skill_bucket": bucket(skill_result["score"], skill_result["confidence"]),
                    "competency_score": comp_entry["score"],
                    "competency_confidence": comp_entry["confidence"],
                    "composite_score": comp,
                })
            scored_pool.sort(key=lambda c: -(c["composite_score"] or 0))
            top_skill_score = scored_pool[0]["skill_score"] if scored_pool else None

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
                    "coe": canonical_project_coe(tech_coe_by_project.get(project_id)),
                    "allocation_by_percentage": alloc["allocation_by_percentage"],
                    "backfill_candidates": scored_pool[:MAX_BACKFILL_SHOWN],
                    "backfill_available": len(scored_pool) > 0,
                    "required_skills": required_phrases,
                    "required_skill_source": required_skill_source,
                }
            )

    return sorted(impacts, key=lambda i: (not i["is_currently_on_leave"], i["leave_start_date"]))
