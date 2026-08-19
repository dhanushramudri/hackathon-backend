"""KRA/Performance Reviews -- half-yearly appraisal cycles modeled on JMAN's
real internal Performance Edge / KRA-KPI Forms tool (see
generate_kra_performance_data.py for the synthetic dataset; real one never
provided by the hackathon team, grounded in real employee/manager pairs --
the appraiser is always a real manager_employee_id, never fabricated).

Same manual-review "proof" posture as feedback_engine.py: a second tab
alongside Project Feedback in the Employee Profile modal. Deliberately NOT
wired into recommendation_service.py or scoring.py.
"""
import pandas as pd

from app.core.adapter import get_adapter

CATEGORY_ORDER = ["Projects", "People", "Products", "Sales"]


def list_employee_cycles(employee_id: str) -> list[dict]:
    """Cycle-list view -- one row per cycle, matching the real "KRA-KPI Forms"
    list page (Form Name, Cycle, Published On, Form End Date, Status)."""
    adapter = get_adapter()
    cycles = adapter.get_performance_cycles()
    if cycles.empty:
        return []
    rows = cycles[cycles["employee_id"] == employee_id].sort_values("published_on", ascending=False)

    out = []
    for _, r in rows.iterrows():
        out.append({
            "cycle_id": r["cycle_id"],
            "form_name": r["form_name"],
            "cycle_label": r["cycle_label"],
            "published_on": r["published_on"].strftime("%Y-%m-%d") if pd.notna(r["published_on"]) else None,
            "form_end_date": r["form_end_date"].strftime("%Y-%m-%d") if pd.notna(r["form_end_date"]) else None,
            "status": r["status"],
            "appraiser_employee_id": r["appraiser_employee_id"],
            "total_score": None if pd.isna(r["total_score"]) else int(r["total_score"]),
            "performance_rating_code": r.get("performance_rating_code") if pd.notna(r.get("performance_rating_code")) else None,
            "performance_rating_label": r.get("performance_rating_label") if pd.notna(r.get("performance_rating_label")) else None,
        })
    return out


def get_cycle_detail(employee_id: str, cycle_id: str) -> dict | None:
    """Full detail view for one cycle -- stage tracker, total score/grade, and
    the Projects/People/Products/Sales/Overall-Feedback sections, matching the
    real "KRA" detail page."""
    adapter = get_adapter()
    cycles = adapter.get_performance_cycles()
    cycle_rows = cycles[(cycles["employee_id"] == employee_id) & (cycles["cycle_id"] == cycle_id)]
    if cycle_rows.empty:
        return None
    cycle = cycle_rows.iloc[0]

    items = adapter.get_performance_kra_items()
    cycle_items = items[items["cycle_id"] == cycle_id]

    sections = []
    for category in CATEGORY_ORDER:
        cat_rows = cycle_items[cycle_items["category"] == category]
        if cat_rows.empty:
            continue
        kra_list = []
        for _, r in cat_rows.iterrows():
            kra_list.append({
                "kra_name": r["kra_name"],
                "weight": int(r["weight"]),
                "kra_kpi_description": r["kra_kpi_description"],
                "goal_text": r["goal_text"],
                "appraisee_rating_text": r["appraisee_rating_text"],
                "appraisee_score": None if pd.isna(r["appraisee_score"]) else int(r["appraisee_score"]),
                "appraiser_rating_text": r["appraiser_rating_text"],
                "appraiser_score": None if pd.isna(r["appraiser_score"]) else int(r["appraiser_score"]),
            })
        sections.append({"category": category, "items": kra_list})

    stage_tracker = _stage_tracker(cycle)

    return {
        "cycle_id": cycle["cycle_id"],
        "employee_id": employee_id,
        "appraiser_employee_id": cycle["appraiser_employee_id"],
        "form_name": cycle["form_name"],
        "cycle_label": cycle["cycle_label"],
        "published_on": cycle["published_on"].strftime("%Y-%m-%d") if pd.notna(cycle["published_on"]) else None,
        "form_end_date": cycle["form_end_date"].strftime("%Y-%m-%d") if pd.notna(cycle["form_end_date"]) else None,
        "status": cycle["status"],
        "total_score": None if pd.isna(cycle["total_score"]) else int(cycle["total_score"]),
        "performance_rating_code": cycle.get("performance_rating_code") if pd.notna(cycle.get("performance_rating_code")) else None,
        "performance_rating_label": cycle.get("performance_rating_label") if pd.notna(cycle.get("performance_rating_label")) else None,
        "stage_tracker": stage_tracker,
        "sections": sections,
        "overall_appraiser_feedback": cycle.get("overall_appraiser_feedback") if pd.notna(cycle.get("overall_appraiser_feedback")) else None,
        "overall_areas_of_improvement": cycle.get("overall_areas_of_improvement") if pd.notna(cycle.get("overall_areas_of_improvement")) else None,
    }


# Mirrors the real tool's 8-step workflow (Goal set -> KRA Agreed -> Appraisee
# Submit -> Appraiser Submit -> Management Review -> Reviewer Intervention ->
# Reviewer Submit -> Closed). The synthetic generator only ever lands a cycle
# on "Appraisee Submit", "Management Review", or "Closed" -- treat every
# other step as implicitly complete once the cycle has reached that point,
# and steps after it as not yet reached.
_STAGE_SEQUENCE = [
    "Goal set", "KRA Agreed", "Appraisee Submit", "Appraiser Submit",
    "Management Review", "Reviewer Intervention", "Reviewer Submit", "Closed",
]


def _stage_tracker(cycle: pd.Series) -> list[dict]:
    current_status = cycle["status"]
    current_index = _STAGE_SEQUENCE.index(current_status) if current_status in _STAGE_SEQUENCE else len(_STAGE_SEQUENCE) - 1
    employee_id = cycle["employee_id"]
    appraiser_id = cycle["appraiser_employee_id"]

    stages = []
    for i, stage_name in enumerate(_STAGE_SEQUENCE):
        if i < current_index:
            state = "complete"
        elif i == current_index:
            state = "complete" if stage_name == "Closed" else "current"
        else:
            state = "pending"
        assignee = appraiser_id if stage_name in ("Appraiser Submit", "Management Review", "Reviewer Intervention", "Reviewer Submit") else employee_id
        stages.append({"stage": stage_name, "state": state, "assignee_employee_id": assignee})
    return stages
