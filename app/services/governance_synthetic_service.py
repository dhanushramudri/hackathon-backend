"""AI-generated risk suggestions for the Cluster Governance view -- filling
the "Risks & Key Projects" gap for projects that have no real risk logged
yet. Uses this app's own LLM provider chain (same Azure OpenAI/Gemini/Claude
failover as sow_extraction_service.py), never hardcoded canned text.

Confirmed live: 08_WSR_Report_clean.csv's `comment` field is 100% null
across all 72,584 real rows -- JDWH currently only grants the RAG status
colors (scope/schedule/quality/csat/team_status), not narrative comments.
A separate real `risk` column (see map_wsr_table's risk_note field) IS
populated for 243 of 72,584 real rows.

Grounding rules, same posture as sow_extraction_service.py:
  - When a real risk_note exists, it is shown VERBATIM, with NO AI
    involvement at all -- real WSR content is shown as real WSR content,
    full stop. (An earlier version also asked the AI for a mitigation step
    grounded in the real note; dropped after real output showed the AI
    inventing filler -- e.g. "review the real risk note on file" -- when the
    note described something no longer actionable, like a discontinued pod.
    Better to show nothing than a plausible-sounding non-answer.)
  - Only when NO real risk_note exists is the AI used at all, generating
    BOTH the risk description and the mitigation step from the WSR RAG
    colors alone. The prompt explicitly forbids inventing project-specific
    detail that isn't in the (empty) input -- the honest output is a plain
    statement of which areas are flagged, not a fabricated narrative.
  - AI results are cached per (project, flagged-areas fingerprint) so the
    same real state is never re-sent to the AI on every page load.
"""
import hashlib
import json

from app.ai import llm
from app.core.config import APP_STATE_DIR

AI_RISK_CACHE_PATH = APP_STATE_DIR / "governance_ai_risk_cache.json"

RISK_TYPE_LABELS = {
    "scope_status": "Scope",
    "schedule_status": "Schedule",
    "quality_status": "Quality",
    "csat_status": "CSAT",
    "team_status": "Team",
}

_FULL_SUGGESTION_PROMPT = """You are a delivery governance analyst at a consultancy, preparing a weekly risk register. A project's Weekly Status Report currently only exposes RAG status colors -- no narrative comment is available yet from the data warehouse.

Flagged areas (Amber or Red) this week: {flagged}

Write:
1. risk_description: one honest sentence stating which of these areas are flagged and what that generally implies for delivery risk -- do NOT invent a client name, a specific cause, or any detail not given above; stay general to what a {flagged} flag typically signals.
2. mitigation_steps: one concrete, generic-but-relevant next step a delivery lead could take to investigate/address a {flagged} flag.

Respond with ONLY a JSON object, no markdown fences: {{"risk_description": "...", "mitigation_steps": "..."}}"""

_DELIVERY_COMMENT_PROMPT = """You are a delivery governance analyst summarizing a project's weekly status for a leadership review. Use ONLY the real facts below -- do not invent a client name, a cause, or any detail not stated here.

DevOps: {devops_summary}
Delivery signals: {milestone_facts}

Write ONE plain sentence (under 30 words) summarizing the overall delivery picture, grounded strictly in the facts above.

Respond with ONLY a JSON object, no markdown fences: {{"comment": "..."}}"""

_CLUSTER_SUMMARY_PROMPT = """You are opening a cluster's weekly governance call for a consultancy's Resource Manager. Write 3-4 short bullet points they'll read aloud to start the discussion -- terse and specific, in the same style as real examples from this team's own past decks:
"Only Heroux Devtek needs escalation (client SoW sign-off); with Boston MFO's revenue sign-off the one item to watch"
"Cluster 5 stays healthy -- nearly all projects green, no red; both ambers are client-driven, not delivery"
"No projects scheduled to kick start this week; Project Eagle and iGlobal Integration both complete on schedule"

Real facts for Cluster {cluster_number} -- {cluster_name} this week ({project_count} projects total):
- WSR status -- Red: {red_names} ({red} total). Amber: {amber_names} ({amber} total). Green: {green} projects. No report on file: {no_report} projects.
- Real logged risks: {logged_risks}
- Flagged but not yet logged as a real risk: {flagged_risks}
- {spotlight_count} project(s) picked for this week's detailed end-to-end review: {spotlight_names}
- Kicking off this week: {kickoff_names}
- Ending or extending this week: {ending_names}

Write 3-4 short bullet points (each one sentence, high level) that a Resource Manager can read aloud to open the discussion. Refer to projects by NAME (as given above), never by an internal code -- names are what get said out loud on a call. Do NOT invent a project, a client name, a cause, or any detail not given above. Prioritize the most severe first (Red WSR, real logged risks); if the cluster is broadly healthy, say so plainly in one point rather than manufacturing concern across several.

Respond with ONLY a JSON object, no markdown fences: {{"points": ["...", "...", "..."]}}"""


def summarize_cluster(cluster_number: int, cluster_name: str, facts: dict) -> list[str] | None:
    """The AI-generated bullet-point highlights opening a cluster's
    governance page -- grounded strictly in the same real aggregates already
    shown in the Risks/Spotlight/Kick-off/Ending/WSR sections below it on
    the page (see governance_service.get_cluster_dashboard), never any new/
    invented data. Project NAMES are passed in, not codes -- meaningless
    when read aloud. Cached per (cluster, fingerprint of those real
    aggregates) so it only regenerates when something in the cluster's real
    state actually changed since the last load."""
    fingerprint_basis = "|".join(
        [
            str(facts.get("project_count")),
            f"{facts.get('red_names')}-{facts.get('amber_names')}-{facts.get('green')}-{facts.get('no_report')}",
            facts.get("logged_risks", ""),
            facts.get("flagged_risks", ""),
            facts.get("spotlight_names", ""),
            facts.get("kickoff_names", ""),
            facts.get("ending_names", ""),
        ]
    )
    cache_key = f"cluster::{cluster_number}::points::{_fingerprint(fingerprint_basis)}"
    cache = _load_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.get("points")
    ai_result = _call_ai(
        _CLUSTER_SUMMARY_PROMPT.format(
            cluster_number=cluster_number,
            cluster_name=cluster_name,
            project_count=facts.get("project_count", 0),
            red=facts.get("red", 0), amber=facts.get("amber", 0), green=facts.get("green", 0), no_report=facts.get("no_report", 0),
            red_names=facts.get("red_names") or "none",
            amber_names=facts.get("amber_names") or "none",
            logged_risks=facts.get("logged_risks") or "none",
            flagged_risks=facts.get("flagged_risks") or "none",
            spotlight_count=facts.get("spotlight_count", 0),
            spotlight_names=facts.get("spotlight_names") or "none",
            kickoff_names=facts.get("kickoff_names") or "none scheduled",
            ending_names=facts.get("ending_names") or "none scheduled",
        )
    )
    points = (ai_result or {}).get("points")
    points = [str(p).strip() for p in points if str(p).strip()] if isinstance(points, list) else None
    cache[cache_key] = {"points": points}
    _save_cache(cache)
    return points


def _fingerprint(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode()).hexdigest()[:20]


def _load_cache() -> dict:
    if not AI_RISK_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(AI_RISK_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict) -> None:
    AI_RISK_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _call_ai(prompt: str) -> dict | None:
    providers = llm.get_providers()
    if not providers:
        return None
    for provider in providers:
        try:
            turn = provider.generate_with_tools([{"role": "user", "content": prompt}], [], max_tokens=300)
        except Exception:
            continue
        content = (turn or {}).get("content")
        if not content:
            continue
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            continue
    return None


def summarize_delivery_comment(project_code: str, devops_summary: str, milestone_facts: list[str]) -> str | None:
    """One AI sentence for the Spotlight table's Comments column, grounded
    strictly in the real DevOps summary + real Milestone Visibility facts
    (timesheet activity, allocation ramp changes, WSR trend -- see
    governance_service.compute_delivery_signals) -- never freeform, never
    inventing anything beyond what's passed in. Cached the same way as
    synthesize_risk's suggestions."""
    facts_str = "; ".join(milestone_facts) if milestone_facts else "none on record"
    cache_key = f"{project_code}::comment::{_fingerprint(devops_summary, facts_str)}"
    cache = _load_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.get("comment")
    ai_result = _call_ai(_DELIVERY_COMMENT_PROMPT.format(devops_summary=devops_summary, milestone_facts=facts_str))
    comment = (ai_result or {}).get("comment")
    cache[cache_key] = {"comment": comment}
    _save_cache(cache)
    return comment


def synthesize_risk(
    project_code: str, project_name: str | None, latest_wsr_row: dict, real_risk_note: str | None = None
) -> dict | None:
    flagged = [label for col, label in RISK_TYPE_LABELS.items() if latest_wsr_row.get(col) in ("RED", "AMBER")]
    if not flagged and not real_risk_note:
        return None

    if real_risk_note:
        # Real WSR content, shown as real WSR content -- no AI involved at
        # all, no mitigation manufactured on top of it.
        return {
            "risk_id": f"synthetic::{project_code}",
            "project_code": project_code,
            "project_name": project_name,
            "risk_description": real_risk_note,
            "risk_type": ", ".join(flagged) if flagged else None,
            "mitigation_steps": None,
            "is_synthetic": False,
            "is_ai_generated": False,
        }

    flagged_str = ", ".join(flagged)
    cache = _load_cache()
    cache_key = f"{project_code}::flags::{_fingerprint(flagged_str)}"
    cached = cache.get(cache_key)
    if cached is None:
        ai_result = _call_ai(_FULL_SUGGESTION_PROMPT.format(flagged=flagged_str))
        cached = {
            "risk_description": (ai_result or {}).get("risk_description"),
            "mitigation_steps": (ai_result or {}).get("mitigation_steps"),
        }
        cache[cache_key] = cached
        _save_cache(cache)

    if not cached.get("risk_description"):
        return {
            "risk_id": f"synthetic::{project_code}",
            "project_code": project_code,
            "project_name": project_name,
            "risk_description": (
                f"AI-generated suggestion unavailable right now -- WSR flags {flagged_str} for this project, "
                "but no AI provider responded."
            ),
            "risk_type": ", ".join(flagged),
            "mitigation_steps": None,
            "is_synthetic": True,
            "is_ai_generated": False,
        }

    return {
        "risk_id": f"synthetic::{project_code}",
        "project_code": project_code,
        "project_name": project_name,
        "risk_description": cached["risk_description"],
        "risk_type": ", ".join(flagged),
        "mitigation_steps": cached.get("mitigation_steps"),
        "is_synthetic": True,
        "is_ai_generated": True,
    }
