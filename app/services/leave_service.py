import pandas as pd

from app.core.adapter import get_adapter
from app.engines.role_mix_engine import canonical_project_coe
from app.services.free_pool_service import get_free_pool_by_designation

MAX_BACKFILL_SHOWN = 3

def get_leave_impact() -> list[dict]:
    adapter = get_adapter()
    leaves = adapter.get_leaves()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()
    projects = adapter.get_projects()
    pool_by_designation = get_free_pool_by_designation()
    tech_coe_by_project = projects.set_index("project_code")["tech_coe"]

    today = pd.Timestamp.now().normalize()
    relevant_leaves = leaves[leaves["leave_end_date"] >= today]
    active_alloc = allocations[allocations["is_allocation_active"] == 1]

    impacts = []
    for _, leave in relevant_leaves.iterrows():
        emp_id = leave["employee_id"]
        emp_row = employees[employees["employee_id"] == emp_id]
        job_name = emp_row["job_name"].iloc[0] if not emp_row.empty else None

        backfill_pool = [c for c in pool_by_designation.get(job_name, []) if c["employee_id"] != emp_id]

        affected = active_alloc[active_alloc["employee_id"] == emp_id]
        for _, alloc in affected.iterrows():
            impacts.append(
                {
                    "employee_id": emp_id,
                    "job_name": job_name,
                    "leave_type": leave["leave_type"],
                    "leave_start_date": leave["leave_start_date"].strftime("%Y-%m-%d"),
                    "leave_end_date": leave["leave_end_date"].strftime("%Y-%m-%d"),
                    "is_currently_on_leave": bool(leave["leave_start_date"] <= today <= leave["leave_end_date"]),
                    "project_id": alloc["project_id"],
                    "coe": canonical_project_coe(tech_coe_by_project.get(alloc["project_id"])),
                    "allocation_by_percentage": alloc["allocation_by_percentage"],
                    "backfill_candidates": backfill_pool[:MAX_BACKFILL_SHOWN],
                    "backfill_available": len(backfill_pool) > 0,
                }
            )

    return sorted(impacts, key=lambda i: (not i["is_currently_on_leave"], i["leave_start_date"]))
