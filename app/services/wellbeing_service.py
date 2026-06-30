from app.core.adapter import get_adapter
from app.services.health_monitor_service import get_health_report
from app.services.timesheet_insights_service import (
    get_employee_overtime_risk,
    get_employee_recent_daily_hours,
    get_employee_recent_projects,
)

def get_project_burnout_overview() -> dict:
    """Every active project genuinely flagged for overtime risk and/or understaffing --
    the same get_health_report() data already used on the Health page, just curated
    down to the projects that actually matter for wellbeing, with the two signals
    separated out for honest counts (a project can fire both)."""
    report = get_health_report()
    flagged = [r for r in report if "overtime_risk" in r["root_causes"] or "understaffed" in r["root_causes"]]
    flagged.sort(key=lambda r: -r["risk_score"])
    return {
        "total_flagged": len(flagged),
        "overtime_count": sum(1 for r in flagged if "overtime_risk" in r["root_causes"]),
        "understaffed_count": sum(1 for r in flagged if "understaffed" in r["root_causes"]),
        "projects": flagged,
    }

def get_employee_burnout_overview() -> dict:
    """Sustained overtime: real, precise overwork signal (>9h on 4+ of last 14 days)."""
    adapter = get_adapter()
    employees = adapter.get_employees()
    job_name_by_id = employees.set_index("employee_id")["job_name"].to_dict()
    dept_by_id = employees.set_index("employee_id")["department_name"].to_dict()

    # Same flagged-project set the Project Burnout tab uses, so an overworked
    # employee can be pointed straight at "relief staffing could help your project"
    # rather than just a number -- this is the actionable, supportive half of the
    # employee view.
    health_report = get_health_report()
    flagged_projects = {
        r["project_code"] for r in health_report if "overtime_risk" in r["root_causes"] or "understaffed" in r["root_causes"]
    }

    risk = get_employee_overtime_risk()
    overtime_employees = []
    for emp_id, r in risk.items():
        if not r.get("is_sustained_overtime"):
            continue
        recent_projects = get_employee_recent_projects(emp_id)
        for p in recent_projects:
            p["needs_support"] = p["project_id"] in flagged_projects
        overtime_employees.append(
            {
                "employee_id": emp_id,
                "job_name": job_name_by_id.get(emp_id),
                "department_name": dept_by_id.get(emp_id),
                "overtime_days_recent": r["overtime_days_recent"],
                "max_daily_hours_recent": r["max_daily_hours_recent"],
                "daily_hours": get_employee_recent_daily_hours(emp_id),
                "recent_projects": recent_projects,
            }
        )
    overtime_employees.sort(key=lambda e: -e["max_daily_hours_recent"])

    return {
        "overtime_employee_count": len(overtime_employees),
        "overtime_employees": overtime_employees,
    }
