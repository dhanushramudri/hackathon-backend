import pandas as pd

from app.core.adapter import get_adapter
from app.services.allocation_report_service import OVER_ALLOCATED_THRESHOLD, UNDER_UTILIZED_THRESHOLD, get_allocation_report
from app.services.timesheet_insights_service import (
    OVERTIME_DAILY_HOURS_THRESHOLD,
    SUSTAINED_OVERTIME_MIN_DAYS,
    SUSTAINED_OVERTIME_WINDOW_DAYS,
    get_employee_overtime_risk,
    get_employee_recent_daily_hours,
)

_SKILL_SOURCE_RANK = {"observed": 0, "imputed_peer": 1, "imputed_default": 2}

class EmployeeNotFound(Exception):

    def __init__(self, employee_id: str):
        self.employee_id = employee_id
        super().__init__(f"employee_id {employee_id!r} not found")

def find_employees(query: str, limit: int = 10) -> list[dict]:
    employees = get_adapter().get_employees()
    q = query.strip().lower()
    if not q:
        return []
    cols = ["employee_id", "job_name", "department_name", "location"]
    mask = pd.Series(False, index=employees.index)
    for col in cols:
        mask |= employees[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
    matches = employees[mask].head(limit)
    return [
        {
            "employee_id": r["employee_id"],
            "job_name": r.get("job_name") if pd.notna(r.get("job_name")) else None,
            "department_name": r.get("department_name") if pd.notna(r.get("department_name")) else None,
            "location": r.get("location") if pd.notna(r.get("location")) else None,
        }
        for _, r in matches.iterrows()
    ]

def get_employee_headcount_summary() -> dict:
    employees = get_adapter().get_employees()
    today = pd.Timestamp.now().normalize()
    resignation = employees["date_of_resignation"]
    already_departed = resignation.notna() & (resignation <= today)
    in_notice_period = resignation.notna() & (resignation > today)
    return {
        "total_ever": int(len(employees)),
        "currently_active": int((~already_departed).sum()),
        "already_departed": int(already_departed.sum()),
        "in_notice_period": int(in_notice_period.sum()),
    }

def skills_for(employee_id: str, skills: pd.DataFrame) -> list[dict]:
    rows = skills[skills["employee_id"] == employee_id].copy()
    rows["_source_rank"] = rows["skill_source"].map(_SKILL_SOURCE_RANK).fillna(9)
    rows = rows.sort_values(["_source_rank", "score"], ascending=[True, False])

    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "coe": r.get("coe") if pd.notna(r.get("coe")) else None,
                "coe_skill": r.get("coe_skill") if pd.notna(r.get("coe_skill")) else None,
                "skill": r.get("skill") if pd.notna(r.get("skill")) else None,
                "subskill": r.get("subskill") if pd.notna(r.get("subskill")) else None,
                "experience": r.get("experience") if pd.notna(r.get("experience")) else None,
                "score": float(r["score"]) if pd.notna(r["score"]) else None,
                "skill_source": r["skill_source"],
            }
        )
    return out

def _competencies_for(employee_id: str, competencies: pd.DataFrame) -> list[dict]:
    rows = competencies[competencies["employee_id"] == employee_id].sort_values("score", ascending=False)
    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "competency_sheet": r.get("competency_sheet") if pd.notna(r.get("competency_sheet")) else None,
                "competency_question": r.get("competency_question") if pd.notna(r.get("competency_question")) else None,
                "response": r.get("response") if pd.notna(r.get("response")) else None,
                "score": float(r["score"]) if pd.notna(r["score"]) else None,
                "competency_source": r["competency_source"],
            }
        )
    return out

def _allocations_for(employee_id: str, allocations: pd.DataFrame, projects: pd.DataFrame) -> list[dict]:
    rows = allocations[allocations["employee_id"] == employee_id].merge(
        projects[["project_code", "client_id", "type_of_project", "project_status"]],
        left_on="project_id", right_on="project_code", how="left",
    )
    rows = rows.sort_values(["is_allocation_active", "allocated_start_date"], ascending=[False, False])

    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "project_id": r["project_id"],
                "client_id": r.get("client_id") if pd.notna(r.get("client_id")) else None,
                "type_of_project": r.get("type_of_project") if pd.notna(r.get("type_of_project")) else None,
                "project_status": r.get("project_status") if pd.notna(r.get("project_status")) else None,
                "resourcing_status": r["resourcing_status"],
                "allocation_by_percentage": float(r["allocation_by_percentage"]) if pd.notna(r["allocation_by_percentage"]) else None,
                "allocated_start_date": r["allocated_start_date"].strftime("%Y-%m-%d") if pd.notna(r["allocated_start_date"]) else None,
                "allocated_end_date": r["allocated_end_date"].strftime("%Y-%m-%d") if pd.notna(r["allocated_end_date"]) else None,
                "is_allocation_active": bool(r["is_allocation_active"]),
            }
        )
    return out

def _leaves_for(employee_id: str, leaves: pd.DataFrame) -> list[dict]:
    rows = leaves[leaves["employee_id"] == employee_id].sort_values("leave_start_date", ascending=False)
    today = pd.Timestamp.now().normalize()
    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "leave_type": r["leave_type"],
                "leave_start_date": r["leave_start_date"].strftime("%Y-%m-%d") if pd.notna(r["leave_start_date"]) else None,
                "leave_end_date": r["leave_end_date"].strftime("%Y-%m-%d") if pd.notna(r["leave_end_date"]) else None,
                "status": r["status"],
                "source": r["source"],
                "is_currently_on_leave": bool(pd.notna(r["leave_start_date"]) and pd.notna(r["leave_end_date"]) and r["leave_start_date"] <= today <= r["leave_end_date"]),
            }
        )
    return out

def get_employee_profile(employee_id: str) -> dict:
    adapter = get_adapter()
    employees = adapter.get_employees()

    match = employees[employees["employee_id"] == employee_id]
    if match.empty:
        raise EmployeeNotFound(employee_id)
    employee_row = match.iloc[0]

    current_allocations = [r for r in get_allocation_report() if r["employee_id"] == employee_id]
    employee_total_allocation_pct = current_allocations[0]["employee_total_allocation_pct"] if current_allocations else None
    overtime_risk = get_employee_overtime_risk().get(
        employee_id, {"overtime_days_recent": 0, "max_daily_hours_recent": 0.0, "is_sustained_overtime": False}
    )

    signals = {
        "over_allocated": bool(employee_total_allocation_pct is not None and employee_total_allocation_pct > OVER_ALLOCATED_THRESHOLD),
        "over_allocated_threshold": OVER_ALLOCATED_THRESHOLD,
        "under_utilized": bool(employee_total_allocation_pct is not None and employee_total_allocation_pct < UNDER_UTILIZED_THRESHOLD),
        "under_utilized_threshold": UNDER_UTILIZED_THRESHOLD,
        "sustained_overtime": bool(overtime_risk["is_sustained_overtime"]),
        "overtime_daily_threshold_hours": OVERTIME_DAILY_HOURS_THRESHOLD,
        "overtime_sustained_min_days": SUSTAINED_OVERTIME_MIN_DAYS,
        "overtime_window_days": SUSTAINED_OVERTIME_WINDOW_DAYS,
        "possible_unplanned_absence": any(r["possible_unplanned_absence"] for r in current_allocations),
    }

    return {
        "employee_id": employee_id,
        "job_name": employee_row.get("job_name") if pd.notna(employee_row.get("job_name")) else None,
        "department_name": employee_row.get("department_name") if pd.notna(employee_row.get("department_name")) else None,
        "location": employee_row.get("location") if pd.notna(employee_row.get("location")) else None,
        "date_of_join": employee_row["date_of_join"].strftime("%Y-%m-%d") if pd.notna(employee_row.get("date_of_join")) else None,
        "account_status": bool(employee_row["account_status"]) if pd.notna(employee_row.get("account_status")) else None,
        "employee_total_allocation_pct": employee_total_allocation_pct,
        "skills": skills_for(employee_id, adapter.get_skills()),
        "competencies": _competencies_for(employee_id, adapter.get_competencies()),
        "allocations": _allocations_for(employee_id, adapter.get_allocations(), adapter.get_projects()),
        "current_allocations": current_allocations,
        "overtime_risk": overtime_risk,
        "daily_hours_recent": get_employee_recent_daily_hours(employee_id),
        "leaves": _leaves_for(employee_id, adapter.get_leaves()),
        "signals": signals,
    }
