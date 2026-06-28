import pandas as pd

from app.core.adapter import get_adapter

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
                "employee_id": r["employee_id"],
                "job_name": r["job_name"],
                "resourcing_status": r["resourcing_status"],
                "allocation_by_percentage": r["allocation_by_percentage"],
                "allocated_start_date": r["allocated_start_date"].strftime("%Y-%m-%d") if pd.notna(r["allocated_start_date"]) else None,
                "allocated_end_date": r["allocated_end_date"].strftime("%Y-%m-%d") if pd.notna(r["allocated_end_date"]) else None,
                "is_allocation_active": bool(r["is_allocation_active"]),
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
        "project_status": row.get("project_status") if pd.notna(row.get("project_status")) else None,
        "project_start_date": row["project_start_date"].strftime("%Y-%m-%d") if pd.notna(row["project_start_date"]) else None,
        "project_end_date": row["project_end_date"].strftime("%Y-%m-%d") if pd.notna(row["project_end_date"]) else None,
        "is_health_tracked": bool(row.get("project_status") == "ACTIVE"),
    }
