"""AI summary of an employee's most recently completed KRA/performance cycle,
focused on Projects and Products (delivery execution + technical/product
contribution -- the two sections most tied to skill/capability, as opposed to
People/Sales which are more behavioral/BD-oriented) plus that cycle's Overall
Feedback.

Lets a Resource Manager get the gist without opening the cycle and reading
every KRA line. Same posture as the rest of this app's AI features: uses the
existing provider failover chain (see app/ai/llm.py), never fabricates beyond
what's in the real (synthetic-but-labeled) cycle data, and is a manual-review
aid only -- never wired into recommendation_service.py or scoring.py.
"""
import pandas as pd

from app.ai import llm
from app.core.adapter import get_adapter

SUMMARY_CATEGORIES = ["Projects", "Products"]

SYSTEM_PROMPT = """You are summarizing a JMAN employee's most recently completed performance
appraisal cycle for a Resource Manager who needs a fast read instead of opening the cycle and
reading every KRA line individually.

You will be given that cycle's Projects and Products KRA line items (each with the
appraiser's rating text) and its Overall Appraiser Feedback / Areas of Improvement text.
These two KRA categories reflect delivery execution and technical/product contribution --
the parts of the appraisal most relevant to a resourcing decision.

Write a concise 2-4 sentence summary covering: (1) how delivery/productivity landed this
cycle, (2) any technical/product contribution worth noting, (3) any area of improvement
worth flagging to the RM. Base this ONLY on the text given -- never invent a rating, project,
or comment that isn't present, and never mention People or Sales KRAs since none are given to
you. Synthesize in your own words rather than repeating input phrases verbatim. Respond with
ONLY the summary text, no preamble or headers."""


def _latest_closed_cycle_with_items(employee_id: str) -> dict | None:
    adapter = get_adapter()
    cycles = adapter.get_performance_cycles()
    rows = cycles[(cycles["employee_id"] == employee_id) & cycles["total_score"].notna()]
    rows = rows.sort_values("published_on", ascending=False)
    if rows.empty:
        return None
    cycle = rows.iloc[0]

    items = adapter.get_performance_kra_items()
    cycle_items = items[(items["cycle_id"] == cycle["cycle_id"]) & (items["category"].isin(SUMMARY_CATEGORIES))]
    return {
        "cycle_label": cycle["cycle_label"],
        "total_score": int(cycle["total_score"]),
        "performance_rating_label": cycle.get("performance_rating_label"),
        "items": [
            {"category": r["category"], "kra_name": r["kra_name"], "appraiser_rating_text": r["appraiser_rating_text"]}
            for _, r in cycle_items.iterrows()
        ],
        "overall_appraiser_feedback": cycle.get("overall_appraiser_feedback") if pd.notna(cycle.get("overall_appraiser_feedback")) else None,
        "overall_areas_of_improvement": cycle.get("overall_areas_of_improvement") if pd.notna(cycle.get("overall_areas_of_improvement")) else None,
    }


def _build_user_message(c: dict) -> str:
    lines = [f"Cycle {c['cycle_label']} (Total Score {c['total_score']}/400, {c['performance_rating_label']}):"]
    for category in SUMMARY_CATEGORIES:
        cat_items = [i for i in c["items"] if i["category"] == category]
        if not cat_items:
            continue
        lines.append(f"  {category}:")
        for i in cat_items:
            lines.append(f"    - {i['kra_name']}: {i['appraiser_rating_text']}")
    if c["overall_appraiser_feedback"]:
        lines.append(f"  Overall Appraiser Feedback: {c['overall_appraiser_feedback']}")
    if c["overall_areas_of_improvement"]:
        lines.append(f"  Areas of Improvement: {c['overall_areas_of_improvement']}")
    return "\n".join(lines)


def get_performance_ai_summary(employee_id: str) -> dict:
    cycle = _latest_closed_cycle_with_items(employee_id)
    if cycle is None:
        return {"available": True, "summary": None, "cycle_label": None, "reason": "No closed performance cycles yet for this employee."}

    providers = llm.get_providers()
    if not providers:
        return {"available": False, "summary": None, "cycle_label": cycle["cycle_label"], "reason": "No AI provider configured."}

    user_message = _build_user_message(cycle)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]

    for provider in providers:
        try:
            turn = provider.generate_with_tools(messages, [], temperature=0.2, max_tokens=300)
        except Exception:
            continue
        content = (turn or {}).get("content")
        if content and content.strip():
            return {"available": True, "summary": content.strip(), "cycle_label": cycle["cycle_label"], "reason": None}

    return {"available": False, "summary": None, "cycle_label": cycle["cycle_label"], "reason": "AI provider call failed."}
