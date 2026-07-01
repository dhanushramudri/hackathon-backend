import re

import numpy as np
import pandas as pd

from app.core.adapter import get_adapter
from app.engines import scoring
from app.engines.designation_ladder import adjacent_designations
from app.engines.employee_coe import get_employee_primary_coe_map
from app.engines.resource_code_decoder import decode_resource_code
from app.engines.skillset_classifier import classify_skillset, classify_skillset_with_proof
from app.engines import embedding_engine

TOP_N = 15
MAX_FALLBACK_CANDIDATES = 5

# Roles that are never valid internal candidates for client delivery work.
# Confirmed by resource management — Finance, HR/People, Operations, IT support,
# Legal, Marketing, and executive/admin functions are excluded from all recommendations.
NON_DELIVERY_ROLES: frozenset[str] = frozenset({
    # Finance
    "Senior Finance Executive", "Finance Executive", "Finance Officer", "Finance Specialist",
    "Finance Assistant", "Finance Controller", "Associate Finance Controller",
    "Interim Group Financial Controller", "Head of Financial Control",
    "FP&A Business Partner", "FP&A Manager", "Head of FP&A",
    # HR / People / Talent
    "HRBP", "Associate HRBP", "Senior HRBP", "HR Administrator",
    "Head of HR", "Head of HR & Operations", "Head of Talent",
    "Senior HR Leader Consultant", "People Lead", "People Partner",
    "People and Internal Communications Associate", "Senior People Operations",
    "Talent Acquisition", "Talent Acquisition Partner",
    "TA Coordinator", "Senior TA Coordinator",
    # Operations / Admin / Resourcing
    "Operations Associate", "Operations Lead",
    "Resourcing Manager", "RM Specialist",
    "Administration", "Admin Manager", "Office Manager",
    "EA and Team Assistant", "Head of Delivery Governance",
    # IT support (internal IT, not delivery engineers)
    "IT Support Engineer", "IT Manager", "IT Infrastructure Enabler",
    # Legal
    "Legal Counsel", "Senior Legal Counsel", "Senior Legal Counsel & Head of Privacy",
    # Marketing
    "Marketing Manager",
    # Executive / board
    "Non-executive Director", "Partner",
})

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

# Blend weights: embedding captures semantic similarity, word matching provides
# verified skill overlap. 65/35 gives semantic the primary voice while keeping
# the proof-backed word signal meaningful.
_EMBEDDING_WEIGHT = 0.65
_WORD_WEIGHT = 0.35


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
    # Embedding layer — pre-built by the caller for multi-role efficiency.
    # When None, built here on demand (also cached internally by embedding_engine).
    emp_embedding_index: dict | None = None,
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

    # ── Embedding layer ───────────────────────────────────────────────────────
    # Build the employee embedding index on first call (cached inside the engine).
    # Compute one job-description embedding, then batch cosine-sim across all
    # employees in a single numpy matmul — total overhead ~1-2 ms.
    if emp_embedding_index is None:
        _skills_df = skills if skills is not None else adapter.get_skills()
        emp_embedding_index = embedding_engine.build_employee_embedding_index(_skills_df)

    job_vec = embedding_engine.embed_jobspec(skillset_text) if skillset_text else None
    semantic_scores: dict[str, float] = {}
    if emp_embedding_index is not None and job_vec is not None:
        semantic_scores = embedding_engine.batch_cosine_similarity(job_vec, emp_embedding_index)
    # ─────────────────────────────────────────────────────────────────────────

    active_employees = employees[
        (employees["account_status"] == 1)
        & (~employees["job_name"].isin(NON_DELIVERY_ROLES))
    ]
    job_name_by_id = active_employees.set_index("employee_id")["job_name"].to_dict()
    competency_index = scoring.build_employee_competency_index(competencies)
    default_competency = {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"}

    results = []
    for emp_id in active_employees["employee_id"]:
        job_name = job_name_by_id.get(emp_id)
        # Word-token matching — always runs; provides proof (matched/missing lists)
        word_result = scoring.score_skill_match(required_phrases, skill_index.get(emp_id, {}))

        # Semantic score — blended in when the embedding layer is available
        sem_score = semantic_scores.get(emp_id)
        if sem_score is not None and required_phrases:
            # 65% semantic + 35% word — semantic drives ranking, word anchors it to
            # verified skill records. When word score is 0 but semantic is strong, the
            # candidate still surfaces; matched_skills proves it if available.
            blended = _EMBEDDING_WEIGHT * sem_score + _WORD_WEIGHT * word_result["score"]
            # Confidence: prefer observed/imputed from word matching; fall back to
            # "semantic_match" when word matching found nothing but embedding did.
            confidence = word_result["confidence"]
            if confidence == "no_match" and sem_score >= 0.35:
                confidence = "semantic_match"
            skill_result = {
                "score": round(min(blended, 1.0), 3),
                "matched": word_result["matched"],
                "missing": word_result["missing"],
                "confidence": confidence,
            }
        else:
            skill_result = word_result

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

    # Sort by bucket first (eligible > trainable > gap), then confidence tier
    # (observed > imputed/semantic > no_match), then composite score.  Without
    # this, a 100%-available employee with only inferred/imputed skills can rank
    # above a Software Engineer with real observed (but weaker or partially busy)
    # skills purely because availability inflates their composite.
    _BUCKET_RANK = {"eligible": 3, "trainable": 2, "gap": 1, "not_assessed": 0}
    _CONFIDENCE_RANK = {"observed": 2, "imputed": 1, "semantic_match": 1, "no_match": 0, "no_requirement": 0}
    ranked = sorted(
        results,
        key=lambda r: (
            _BUCKET_RANK.get(r["bucket"], 0),
            _CONFIDENCE_RANK.get(r["skill_confidence"], 0),
            r["composite_score"],
        ),
        reverse=True,
    )
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
    # observed = directly assessed skill records (highest confidence)
    # imputed  = inferred from peers/defaults (lower confidence, shown separately)
    # semantic = AI embedding only, no skill records at all
    observed_skill_match_count = sum(
        1 for r in top if r["skill_confidence"] == "observed"
    )
    inferred_skill_match_count = sum(
        1 for r in top if r["skill_confidence"] == "imputed"
    )
    semantic_only_match_count = sum(
        1 for r in top if r["skill_confidence"] == "semantic_match"
    )
    # Legacy alias kept so existing callers don't break
    genuine_skill_match_count = observed_skill_match_count

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
        "observed_skill_match_count": observed_skill_match_count,
        "inferred_skill_match_count": inferred_skill_match_count,
        "semantic_only_match_count": semantic_only_match_count,
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
    _skillset_raw = row.get("skillset", "")
    _skillset = _skillset_raw if isinstance(_skillset_raw, str) else ""
    result = get_recommendations(
        skillset_text=_skillset,
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
    skills = adapter.get_skills()
    skill_index = scoring.build_employee_skill_index(skills)
    employee_tokens = skill_index.get(employee_id, {})

    # Semantic layer — get this employee's own embedding vector, then for each
    # pipeline row embed its skillset text (cached in embed_jobspec) and compute
    # cosine similarity. Same 65/35 blend as get_recommendations, reversed direction.
    emp_embedding_index = embedding_engine.build_employee_embedding_index(skills)
    emp_vec = emp_embedding_index.get(employee_id) if emp_embedding_index else None

    competency_index = scoring.build_employee_competency_index(adapter.get_competencies())
    competency_entry = competency_index.get(employee_id, {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
    allocations = adapter.get_allocations()

    matches = []
    for row in _open_pipeline_rows_enriched():
        word_result = scoring.score_skill_match(row["required_phrases"], employee_tokens)

        # Blend semantic into the word score for this pipeline row
        skill_result = word_result
        if emp_vec is not None and row["required_phrases"]:
            row_vec = embedding_engine.embed_jobspec(" | ".join(row["required_phrases"]))
            if row_vec is not None:
                sem_score = float(np.clip(np.dot(emp_vec, row_vec), 0.0, 1.0))
                blended = 0.65 * sem_score + 0.35 * word_result["score"]
                confidence = word_result["confidence"]
                if confidence == "no_match" and sem_score >= 0.35:
                    confidence = "semantic_match"
                skill_result = {"score": round(min(blended, 1.0), 3), "matched": word_result["matched"], "missing": word_result["missing"], "confidence": confidence}

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
    trade for a table-wide preview; the modal drilldown still uses the precise version.
    Uses the same 65/35 semantic+word blend as get_recommendations (reversed direction:
    employee vec → pipeline row vec, same as get_redeploy_matches_for_employee)."""
    adapter = get_adapter()
    skills = adapter.get_skills()
    skill_index = scoring.build_employee_skill_index(skills)
    competency_index = scoring.build_employee_competency_index(adapter.get_competencies())
    emp_embedding_index = embedding_engine.build_employee_embedding_index(skills)
    allocations = adapter.get_allocations()
    busy_pct_today = availability_as_of(allocations, pd.Timestamp.now().normalize())
    open_rows = _open_pipeline_rows_enriched()

    summary: dict[str, dict] = {}
    for emp_id in employee_ids:
        employee_tokens = skill_index.get(emp_id, {})
        if not employee_tokens:
            summary[emp_id] = {"recommended_project_count": 0, "top_match": None}
            continue
        emp_vec = emp_embedding_index.get(emp_id) if emp_embedding_index else None
        competency_entry = competency_index.get(emp_id, {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
        available_pct = max(0.0, 100.0 - float(busy_pct_today.get(emp_id, 0.0)))
        availability_score = min(available_pct / 100.0, 1.0)

        best = None
        count = 0
        for row in open_rows:
            word_score = _fast_skill_score(row["phrase_tokens"], employee_tokens)
            if emp_vec is not None and row["required_phrases"]:
                row_vec = embedding_engine.embed_jobspec(" | ".join(row["required_phrases"]))
                if row_vec is not None:
                    sem_score = float(np.clip(np.dot(emp_vec, row_vec), 0.0, 1.0))
                    skill_score = round(min(0.65 * sem_score + 0.35 * word_score, 1.0), 3)
                else:
                    skill_score = word_score
            else:
                skill_score = word_score
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

    # Prefetch once — each get_recommendations_for_pipeline_row call reuses this.
    # The embedding index is built here so all roles share the same cached vectors.
    _skills = adapter.get_skills()
    prefetched: dict = {
        "employees": adapter.get_employees(),
        "competencies": adapter.get_competencies(),
        "allocations": adapter.get_allocations(),
        "pipeline_skillset": adapter.get_pipeline_skillset(),
        "skills": _skills,
        "skill_index": scoring.build_employee_skill_index(_skills),
        "employee_coe_map": get_employee_primary_coe_map(),
        "emp_embedding_index": embedding_engine.build_employee_embedding_index(_skills),
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


_coverage_cache: dict | None = None
_coverage_fingerprint: tuple | None = None


def get_coverage_summary() -> dict:
    """Fast coverage summary using word-token scoring only — no embedding model call.

    The full get_recommendations_for_pipeline_row path runs sentence-transformer
    inference for every pipeline row (293 × ~5s = many minutes) and blocks the
    entire uvicorn worker while doing so. This version uses only the pre-built
    skill-token index (pure pandas/numpy, <1s total) which is accurate enough
    for the aggregate redeploy/hire-signal counts shown in the UI banner.
    Results are cached until the underlying data changes.
    """
    global _coverage_cache, _coverage_fingerprint

    adapter = get_adapter()
    pipeline = adapter.get_pipeline_forecast()
    skills = adapter.get_skills()
    pipeline_skillset = adapter.get_pipeline_skillset()

    fp = (
        len(pipeline),
        int(pd.util.hash_pandas_object(pipeline["status"], index=False).sum()),
        len(skills),
    )
    if _coverage_cache is not None and _coverage_fingerprint == fp:
        return _coverage_cache

    skill_index = scoring.build_employee_skill_index(skills)
    employees = adapter.get_employees()
    active_employees = employees[
        (employees["account_status"] == 1)
        & (~employees["job_name"].isin(NON_DELIVERY_ROLES))
    ]
    active_ids = set(active_employees["employee_id"].tolist())

    allocations = adapter.get_allocations()
    as_of_today = pd.Timestamp.now().normalize()
    busy_pct = availability_as_of(allocations, as_of_today)

    rows = []
    for row_index, row in pipeline.iterrows():
        _skillset_raw = row.get("skillset", "")
        _skillset = _skillset_raw if isinstance(_skillset_raw, str) else ""
        required_phrases = scoring.tokenize_skillset(_skillset)
        required_phrases = scoring.enrich_required_phrases(required_phrases, pipeline_skillset)
        has_skillset = bool(required_phrases)

        if not has_skillset:
            rows.append({
                "row_index": int(row_index),
                "client": row.get("client"),
                "resources_requested": row.get("resources_requested"),
                "top_candidate_signal": None,
                "top_bucket": None,
                "has_skillset": False,
            })
            continue

        # Score every active delivery employee with word-token only (no embeddings)
        best_signal = "hire"
        best_bucket = "gap"
        for emp_id in active_ids:
            avail = max(0.0, 100.0 - float(busy_pct.get(emp_id, 0.0)))
            if avail < 100.0:
                continue  # fast approximation: only consider fully-free employees
            word_result = scoring.score_skill_match(required_phrases, skill_index.get(emp_id, {}))
            b = scoring.bucket(word_result["score"], word_result["confidence"])
            if b == "eligible":
                best_bucket = "eligible"
                best_signal = "redeploy"
                break
            if b == "trainable" and best_bucket != "eligible":
                best_bucket = "trainable"
                best_signal = "redeploy_with_training"

        rows.append({
            "row_index": int(row_index),
            "client": row.get("client"),
            "resources_requested": row.get("resources_requested"),
            "top_candidate_signal": best_signal,
            "top_bucket": best_bucket,
            "has_skillset": True,
        })

    total = len(rows)
    no_skillset_count = sum(1 for r in rows if not r["has_skillset"])
    hire_count = sum(1 for r in rows if r["top_candidate_signal"] == "hire")
    redeploy_count = sum(1 for r in rows if r["top_candidate_signal"] == "redeploy")
    training_count = sum(1 for r in rows if r["top_candidate_signal"] == "redeploy_with_training")

    result = {
        "total_demand_rows": total,
        "no_skillset_specified_count": no_skillset_count,
        "redeploy_ready_count": redeploy_count,
        "redeploy_with_training_count": training_count,
        "hire_signal_count": hire_count,
        "hire_signal_pct": round(100.0 * hire_count / total, 1) if total else 0.0,
        "rows": rows,
    }
    _coverage_cache = result
    _coverage_fingerprint = fp
    return result
