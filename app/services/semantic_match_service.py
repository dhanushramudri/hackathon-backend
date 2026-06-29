import json
import re

import pandas as pd

from app.ai import llm
from app.ai.providers.base import QuotaExceededError
from app.core.adapter import get_adapter
from app.engines import scoring
from app.services.employee_profile_service import skills_for
from app.services.recommendation_service import RowIndexOutOfRange, availability_as_of

POOL_SIZE = 25
MAX_SKILLS_PER_CANDIDATE = 40

SYSTEM_PROMPT = """You are a strict skill-matching auditor for JMAN's Resource Management
Group. You will be given a required skillset (free text) and a list of candidates, each
with their real, recorded workplace skills/subskills. Your job: identify which candidates,
if any, plausibly satisfy the requirement even if the wording differs (synonyms, broader
or narrower phrasing of the same underlying skill) -- not just literal keyword overlap.

Hard rule: you may ONLY cite a skill or subskill that appears character-for-character in
that candidate's list below. Never invent, paraphrase into a new term, or attribute a
skill from one candidate to another. If a candidate has nothing plausibly relevant, leave
them out entirely. If no candidate has anything plausibly relevant, say so honestly --
do not stretch a weak match to avoid an empty result.

Respond with ONLY a JSON object, no other text:
{"matches": [{"employee_id": "...", "matched_requirement": "<which part of the requirement this addresses>", "evidence_skill": "<exact skill text from that candidate's list, or null>", "evidence_subskill": "<exact subskill text from that candidate's list, or null>", "confidence": "high"|"medium", "rationale": "<one sentence>"}], "no_match_found": true|false}"""

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def _build_candidate_pool(likely_start_date: str) -> list[dict]:
    adapter = get_adapter()
    employees = adapter.get_employees()
    competencies = adapter.get_competencies()
    allocations = adapter.get_allocations()

    active_employees = employees[employees["account_status"] == 1]
    as_of_date = pd.to_datetime(likely_start_date)
    busy_pct = availability_as_of(allocations, as_of_date)
    competency_index = scoring.build_employee_competency_index(competencies)
    default_competency = {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"}

    pool = []
    for _, emp in active_employees.iterrows():
        emp_id = emp["employee_id"]
        job_name = emp.get("job_name")
        available_pct = max(0.0, 100.0 - float(busy_pct.get(emp_id, 0.0)))
        competency_score = competency_index.get(emp_id, default_competency)["score"]
        rank_score = (available_pct / 100.0) * 0.5 + competency_score * 0.5
        pool.append(
            {
                "employee_id": emp_id,
                "job_name": job_name if pd.notna(job_name) else None,
                "rank_score": rank_score,
            }
        )

    pool.sort(key=lambda p: -p["rank_score"])
    return pool[:POOL_SIZE]

def _format_candidate_skills(employee_id: str, skills_df: pd.DataFrame) -> list[dict]:
    rows = skills_for(employee_id, skills_df)
    seen = set()
    out = []
    for r in rows:
        if not (r["skill"] or r["subskill"]):
            continue
        key = (r["skill"] or "", r["subskill"] or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= MAX_SKILLS_PER_CANDIDATE:
            break
    return out

def _parse_llm_response(content: str | None) -> dict:
    if not content:
        return {"matches": [], "no_match_found": True}
    cleaned = _JSON_FENCE_RE.sub("", content.strip())
    try:
        parsed = json.loads(cleaned)
        return {"matches": parsed.get("matches") or [], "no_match_found": bool(parsed.get("no_match_found", False))}
    except (json.JSONDecodeError, AttributeError):
        return {"matches": [], "no_match_found": True}

def _verify_match(claim: dict, pool_skills: dict[str, list[dict]]) -> dict | None:
    employee_id = claim.get("employee_id")
    if not employee_id or employee_id not in pool_skills:
        return None

    matched_requirement = (claim.get("matched_requirement") or "").strip()
    if not matched_requirement:
        return None

    evidence_skill = (claim.get("evidence_skill") or "").strip()
    evidence_subskill = (claim.get("evidence_subskill") or "").strip()
    if not evidence_skill and not evidence_subskill:
        return None

    for row in pool_skills[employee_id]:
        skill_match = bool(evidence_skill) and bool(row["skill"]) and evidence_skill.lower() == row["skill"].strip().lower()
        subskill_match = (
            bool(evidence_subskill) and bool(row["subskill"]) and evidence_subskill.lower() == row["subskill"].strip().lower()
        )
        if skill_match or subskill_match:
            confidence = claim.get("confidence") if claim.get("confidence") in ("high", "medium") else "medium"
            return {
                "employee_id": employee_id,
                "matched_requirement": matched_requirement,
                "skill": row["skill"],
                "subskill": row["subskill"],
                "score": row["score"],
                "skill_source": row["skill_source"],
                "confidence": confidence,
                "rationale": (claim.get("rationale") or "").strip() or None,
            }
    return None

def get_semantic_match_suggestions(row_index: int) -> dict:
    adapter = get_adapter()
    pipeline = adapter.get_pipeline_forecast()
    if row_index < 0 or row_index >= len(pipeline):
        raise RowIndexOutOfRange(row_index, len(pipeline) - 1)

    row = pipeline.iloc[row_index]
    skillset_text = row.get("skillset")
    if not skillset_text or pd.isna(skillset_text):
        return {
            "available": True,
            "requirement": None,
            "matches": [],
            "candidates_considered": 0,
            "no_match_found": True,
        }

    provider = llm.get_provider()
    if provider is None:
        return {"available": False, "reason": "No AI provider configured."}

    pool = _build_candidate_pool(str(row.get("likely_start_date")))
    skills_df = adapter.get_skills()
    pool_skills = {c["employee_id"]: _format_candidate_skills(c["employee_id"], skills_df) for c in pool}
    job_name_by_id = {c["employee_id"]: c["job_name"] for c in pool}

    candidate_lines = []
    for emp_id, rows in pool_skills.items():
        phrases = [f"{r['skill']}: {r['subskill']}" if r["subskill"] else r["skill"] for r in rows if r["skill"] or r["subskill"]]
        candidate_lines.append(f"{emp_id} ({job_name_by_id.get(emp_id) or 'Unknown role'}): {', '.join(phrases)}")

    user_message = f"Required skillset: {skillset_text}\n\nCandidates:\n" + "\n".join(candidate_lines)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]

    try:
        turn = provider.generate_with_tools(messages, [], temperature=0.0, max_tokens=1500)
    except QuotaExceededError:
        return {"available": False, "reason": "AI quota exceeded -- try again later."}

    if turn is None:
        return {"available": False, "reason": "AI provider call failed."}

    parsed = _parse_llm_response(turn.get("content"))
    verified = []
    seen_pairs = set()
    for claim in parsed["matches"]:
        verified_match = _verify_match(claim, pool_skills)
        if verified_match is None:
            continue
        pair_key = (verified_match["employee_id"], verified_match["skill"], verified_match["subskill"])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        verified.append(verified_match)

    return {
        "available": True,
        "requirement": skillset_text,
        "matches": verified,
        "candidates_considered": len(pool),
        "no_match_found": len(verified) == 0,
    }
