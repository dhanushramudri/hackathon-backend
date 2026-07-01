import pandas as pd

from app.core.adapter import get_adapter
from app.engines.employee_coe import get_employee_primary_coe_map
from app.services.allocation_report_service import INTERNAL_PROJECT_TYPE, UNDER_UTILIZED_THRESHOLD, get_allocation_report
from app.services.rate_card_service import get_hourly_rate

STANDARD_MONTHLY_HOURS = 160

def _idle_value_usd_per_month(job_name, idle_pct: float) -> float | None:
    rate = get_hourly_rate(job_name)
    if rate is None:
        return None
    return round((idle_pct / 100) * rate * STANDARD_MONTHLY_HOURS, 0)

def get_free_pool(include_redeploy_summary: bool = True) -> list[dict]:
    from app.services.recommendation_service import NON_DELIVERY_ROLES  # local to avoid circular import
    adapter = get_adapter()
    employees = adapter.get_employees()
    allocations = adapter.get_allocations()
    report = get_allocation_report()
    coe_map = get_employee_primary_coe_map()
    today = pd.Timestamp.now().normalize()

    delivery_ids = set(
        employees[
            (employees["account_status"] == 1)
            & (~employees["job_name"].isin(NON_DELIVERY_ROLES))
        ]["employee_id"]
    )

    ended = allocations[allocations["allocated_end_date"] < today]
    # .last() after an ascending sort carries the project_id of that specific max-date
    # row along with it -- a separate .max() on the date alone would lose which
    # project it actually was, leaving "free for Xd" with no real proof behind it.
    last_ended_row_by_employee = ended.sort_values("allocated_end_date").groupby("employee_id").last()

    ending_rows_by_emp: dict[str, list[dict]] = {}
    for r in report:
        if r["employee_id"] not in delivery_ids:
            continue
        # An ending internal-project allocation doesn't free up "capacity" in any real
        # sense -- it was never blocking client work to begin with.
        if r["ending_soon"] and r["type_of_project"] != INTERNAL_PROJECT_TYPE:
            ending_rows_by_emp.setdefault(r["employee_id"], []).append(r)

    pool: dict[str, dict] = {}
    for emp_id, rows in ending_rows_by_emp.items():
        rows = sorted(rows, key=lambda r: r["days_to_end"])
        nearest = rows[0]
        pool[emp_id] = {
            "employee_id": emp_id, "job_name": nearest["job_name"], "department_name": nearest["department_name"],
            "location": nearest["location"], "reason": "ending_soon", "project_id": nearest["project_id"],
            "days_to_end": nearest["days_to_end"], "current_allocation_pct": nearest["employee_total_allocation_pct"],
            "ending_allocation_pct": sum(r["allocation_by_percentage"] for r in rows),
            "ending_allocations": [
                {"project_id": r["project_id"], "allocation_pct": r["allocation_by_percentage"], "days_to_end": r["days_to_end"]}
                for r in rows
            ],
        }

    for r in report:
        emp_id = r["employee_id"]
        if emp_id not in delivery_ids:
            continue
        if emp_id in pool:
            continue
        # Judged on client-only allocation, not utilization_band's total-based check --
        # someone fully loaded with internal work but with no/low client allocation is
        # genuinely available for new client work, even though their total looks busy.
        if r["employee_client_allocation_pct"] < UNDER_UTILIZED_THRESHOLD:
            pool[emp_id] = {
                "employee_id": emp_id, "job_name": r["job_name"], "department_name": r["department_name"],
                "location": r["location"], "reason": "under_utilized", "project_id": r["project_id"],
                "current_allocation_pct": r["employee_client_allocation_pct"],
            }

    allocated_ids = {r["employee_id"] for r in report}
    fully_free = employees[
        (employees["account_status"] == 1)
        & (~employees["job_name"].isin(NON_DELIVERY_ROLES))
        & (~employees["employee_id"].isin(allocated_ids))
    ]
    for _, row in fully_free.iterrows():
        pool.setdefault(row["employee_id"], {
            "employee_id": row["employee_id"], "job_name": row["job_name"], "department_name": row["department_name"],
            "location": row["location"], "reason": "fully_free", "project_id": None, "current_allocation_pct": 0.0,
        })

    for emp_id, c in pool.items():
        if c["reason"] == "ending_soon":
            idle_pct = max(0.0, min(100.0, c.get("ending_allocation_pct") or 0.0))
        else:
            idle_pct = max(0.0, 100.0 - (c.get("current_allocation_pct") or 0.0))
        c["primary_coe"] = coe_map.get(emp_id)
        c["idle_capacity_pct"] = round(idle_pct, 1)
        c["hourly_rate_usd"] = get_hourly_rate(c["job_name"])
        c["idle_value_usd_per_month"] = _idle_value_usd_per_month(c["job_name"], idle_pct)
        if c["reason"] == "fully_free":
            last_row = last_ended_row_by_employee.loc[emp_id] if emp_id in last_ended_row_by_employee.index else None
            last_ended = last_row["allocated_end_date"] if last_row is not None else None
            c["days_free"] = int((today - last_ended).days) if pd.notna(last_ended) else None
            c["last_ended_project_id"] = last_row["project_id"] if last_row is not None else None
            c["last_ended_date"] = last_ended.strftime("%Y-%m-%d") if pd.notna(last_ended) else None
        else:
            c["days_free"] = None
            c["last_ended_project_id"] = None
            c["last_ended_date"] = None

    if include_redeploy_summary:
        # Imported here, not at module level -- recommendation_service doesn't import
        # free_pool_service, but keeping the import local avoids any future risk of a
        # circular import as both modules grow, and makes the dependency explicit at
        # the one call site that actually needs it.
        from app.services.recommendation_service import get_redeploy_summary_for_employees

        summary = get_redeploy_summary_for_employees(list(pool.keys()))
        for emp_id, c in pool.items():
            s = summary.get(emp_id, {"recommended_project_count": 0, "top_match": None})
            c["recommended_project_count"] = s["recommended_project_count"]
            c["top_recommended_project"] = s["top_match"]

    return sorted(pool.values(), key=lambda c: (c["reason"] != "fully_free", c.get("current_allocation_pct", 0)))

def get_free_pool_by_designation() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    # Leave's backfill matching doesn't use "recommended projects" -- skip the extra
    # composite-score pass over every open pipeline row for every candidate here.
    for c in get_free_pool(include_redeploy_summary=False):
        grouped.setdefault(c["job_name"], []).append(c)
    return grouped
