import pandas as pd

from app.core.adapter import get_adapter

def _date_str(value) -> str | None:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else None

def get_project_roster(project_code: str) -> dict:
    adapter = get_adapter()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()

    rows = allocations[allocations["project_id"] == project_code].merge(
        employees[["employee_id", "job_name", "department_name", "location"]], on="employee_id", how="left"
    )
    if rows.empty:
        return {"project_code": project_code, "roster": [], "distinct_employees": 0}

    rows = rows.sort_values("allocated_start_date")
    roster = []
    for _, r in rows.iterrows():
        roster.append(
            {
                "allocation_id": r["project_rolebased_user_id"],
                "employee_id": r["employee_id"],
                "job_name": r["job_name"],
                "resourcing_status": r["resourcing_status"],
                "allocation_by_percentage": r["allocation_by_percentage"],
                "allocated_start_date": _date_str(r["allocated_start_date"]),
                "allocated_end_date": _date_str(r["allocated_end_date"]),
                "extended_start_date": _date_str(r.get("extended_start_date")),
                "extended_end_date": _date_str(r.get("extended_end_date")),
                "extended_status": r.get("extended_status") if pd.notna(r.get("extended_status")) else None,
                "is_allocation_active": bool(r["is_allocation_active"]),
                "shift_type": r.get("shift_type") if pd.notna(r.get("shift_type")) else None,
                "reviewer_employee_id": r.get("reviewer_employee_id") if pd.notna(r.get("reviewer_employee_id")) else None,
            }
        )

    return {
        "project_code": project_code,
        "roster": roster,
        "distinct_employees": int(rows["employee_id"].nunique()),
    }

def get_project_info(project_code: str) -> dict | None:
    adapter = get_adapter()
    projects = adapter.get_projects()
    match = projects[projects["project_code"] == project_code]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "project_code": project_code,
        "client_id": row.get("client_id") if pd.notna(row.get("client_id")) else None,
        "type_of_project": row.get("type_of_project") if pd.notna(row.get("type_of_project")) else None,
        "tech_coe": row.get("tech_coe") if pd.notna(row.get("tech_coe")) else None,
        "proposition_coe": row.get("proposition_coe") if pd.notna(row.get("proposition_coe")) else None,
        "project_status": row.get("project_status") if pd.notna(row.get("project_status")) else None,
        "project_start_date": row["project_start_date"].strftime("%Y-%m-%d") if pd.notna(row["project_start_date"]) else None,
        "project_end_date": row["project_end_date"].strftime("%Y-%m-%d") if pd.notna(row["project_end_date"]) else None,
        "is_health_tracked": bool(row.get("project_status") == "ACTIVE"),
    }
