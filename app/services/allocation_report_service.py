import uuid
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from app.core.adapter import get_adapter
from app.core.config import TRANSFORMED_DIR
from app.core import db as db_module
from app.engines.role_mix_engine import canonical_project_coe

ENDING_SOON_DAYS = 30
OVER_ALLOCATED_THRESHOLD = 100
UNDER_UTILIZED_THRESHOLD = 70
STANDARD_HOURS_PER_DAY = 8
UNPLANNED_ABSENCE_WINDOW_DAYS = 14
INTERNAL_PROJECT_TYPE = "Internal Project"

ALLOCATIONS_CSV = TRANSFORMED_DIR / "03_Project_Allocation_clean.csv"

def create_allocation(
    employee_id: str, project_id: str, allocation_pct: float,
    start_date: str, end_date: str, resourcing_status: str = "BILLABLE",
) -> dict:
    """Assign an employee to a project -- appends a new allocation row to the
    source CSV (source of truth for every other read in this app) and reloads
    the in-memory DB so it's immediately queryable.
    ponytail: whole-file read/rewrite per assign, fine at this data size (~17k
    rows); move to a real DB/append-only store if assign volume ever matters."""
    adapter = get_adapter()
    if employee_id not in set(adapter.get_employees()["employee_id"]):
        raise ValueError(f"Employee {employee_id!r} not found")
    if project_id not in set(adapter.get_projects()["project_code"]):
        raise ValueError(f"Project {project_id!r} not found")
    if not (0 < allocation_pct <= 100):
        raise ValueError("allocation_pct must be between 0 and 100")
    if pd.to_datetime(end_date) < pd.to_datetime(start_date):
        raise ValueError("end_date cannot be before start_date")

    df = pd.read_csv(ALLOCATIONS_CSV)
    df.columns = [c.strip() for c in df.columns]
    new_row = {
        "project_rolebased_user_id": str(uuid.uuid4()).upper(),
        "project_id": project_id,
        "employee_id": employee_id,
        "resourcing_status": resourcing_status,
        "allocated_start_date": start_date,
        "allocated_end_date": end_date,
        "is_allocation_active": 1,
        "allocation_by_percentage": allocation_pct,
        "is_active_version": 1,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ALLOCATIONS_CSV, index=False)
    db_module.reload()
    return new_row

def _utilization_band(total_pct: float, client_pct: float) -> str:
    # Over-allocation is judged on client_pct (Client Project/Managed Services/BAU/Sales),
    # never on internal-project work -- internal projects are discretionary ("contribute
    # when you have time"), not a hard commitment, so they shouldn't make someone look
    # over capacity. Under-utilization still looks at the real total, since spare time
    # spent on internal work is still spare time from a staffing perspective.
    if client_pct > OVER_ALLOCATED_THRESHOLD:
        return "over_allocated"
    if total_pct < UNDER_UTILIZED_THRESHOLD:
        return "under_utilized"
    return "normal"

def _hours_metrics(active: pd.DataFrame, timesheets: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    active = active.copy()
    active["_row_id"] = active.index
    active["_window_end"] = active["allocated_end_date"].clip(upper=today)

    ts = timesheets[["employee_id", "project_id", "date", "time"]].dropna(subset=["date"])
    merged = active[["_row_id", "employee_id", "project_id", "allocated_start_date", "_window_end"]].merge(
        ts, on=["employee_id", "project_id"], how="left"
    )
    in_window = (
        merged["date"].notna()
        & (merged["date"] >= merged["allocated_start_date"])
        & (merged["date"] <= merged["_window_end"])
    )
    merged["_time_in_window"] = merged["time"].where(in_window, 0.0)
    actual_hours = merged.groupby("_row_id")["_time_in_window"].sum().rename("actual_hours_logged")

    begin = active["allocated_start_date"].values.astype("datetime64[D]")
    end = (active["_window_end"] + pd.Timedelta(days=1)).values.astype("datetime64[D]")
    working_days = np.maximum(np.busday_count(begin, end), 0)
    active["expected_hours"] = working_days * STANDARD_HOURS_PER_DAY * (active["allocation_by_percentage"] / 100)

    active = active.merge(actual_hours, left_on="_row_id", right_index=True, how="left")
    active["actual_hours_logged"] = active["actual_hours_logged"].fillna(0.0)
    active["hours_data_available"] = active["expected_hours"] > 0
    active["hours_utilization_pct"] = (active["actual_hours_logged"] / active["expected_hours"] * 100).where(
        active["hours_data_available"]
    )

    active["actual_hours_logged"] = active["actual_hours_logged"].round(1)
    active["expected_hours"] = active["expected_hours"].round(1)
    active["hours_utilization_pct"] = active["hours_utilization_pct"].round(1)

    window_start = today - pd.Timedelta(days=UNPLANNED_ABSENCE_WINDOW_DAYS)
    in_window = merged["date"].notna() & (merged["date"] >= merged["allocated_start_date"]) & (merged["date"] <= today)
    in_recent_window = in_window & (merged["date"] >= window_start)
    in_prior_window = in_window & (merged["date"] < window_start)
    merged["_time_recent"] = merged["time"].where(in_recent_window, 0.0)
    merged["_time_prior"] = merged["time"].where(in_prior_window, 0.0)
    recent_hours = merged.groupby("_row_id")["_time_recent"].sum().rename("_recent_hours_logged")
    prior_hours = merged.groupby("_row_id")["_time_prior"].sum().rename("_prior_hours_logged")
    active = active.merge(recent_hours, left_on="_row_id", right_index=True, how="left")
    active = active.merge(prior_hours, left_on="_row_id", right_index=True, how="left")
    active["_recent_hours_logged"] = active["_recent_hours_logged"].fillna(0.0)
    active["_prior_hours_logged"] = active["_prior_hours_logged"].fillna(0.0)
    is_ongoing = (active["allocated_start_date"] <= today) & (active["allocated_end_date"] >= today)
    active["possible_unplanned_absence"] = (
        is_ongoing & (active["_recent_hours_logged"] <= 0) & (active["_prior_hours_logged"] > 0)
    )

    return active.drop(columns=["_row_id", "_window_end", "_recent_hours_logged", "_prior_hours_logged"])

class AllocationNotFound(Exception):

    def __init__(self, employee_id: str, project_id: str):
        self.employee_id = employee_id
        self.project_id = project_id
        super().__init__(f"no active allocation for employee_id {employee_id!r} on project_id {project_id!r}")

def get_allocation_timesheet(employee_id: str, project_id: str) -> dict:
    report_row = next(
        (r for r in get_allocation_report() if r["employee_id"] == employee_id and r["project_id"] == project_id),
        None,
    )
    if report_row is None:
        raise AllocationNotFound(employee_id, project_id)

    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    today = pd.Timestamp.now().normalize()
    start = pd.Timestamp(report_row["allocated_start_date"])
    window_end = min(pd.Timestamp(report_row["allocated_end_date"]), today)

    ts = timesheets[
        (timesheets["employee_id"] == employee_id)
        & (timesheets["project_id"] == project_id)
        & timesheets["date"].notna()
        & (timesheets["date"] >= start)
        & (timesheets["date"] <= window_end)
    ]
    daily = ts.groupby("date")["time"].sum().sort_index()

    pct = report_row["allocation_by_percentage"] / 100
    daily_hours = []
    for d in pd.date_range(start, window_end, freq="D"):
        is_workday = d.weekday() < 5
        expected_that_day = round(STANDARD_HOURS_PER_DAY * pct, 2) if is_workday else 0.0
        if d in daily.index:
            h = daily.loc[d]
            daily_hours.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "hours": float(round(h, 2)),
                    "expected_hours": expected_that_day,
                    "utilization_pct": round(float(h) / expected_that_day * 100, 1) if expected_that_day > 0 else None,
                    "is_missing": False,
                }
            )
        elif is_workday:
            daily_hours.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "hours": None,
                    "expected_hours": expected_that_day,
                    "utilization_pct": None,
                    "is_missing": True,
                }
            )

    return {
        **report_row,
        "hours_window_end": window_end.strftime("%Y-%m-%d"),
        "daily_hours": daily_hours,
    }

def get_allocation_report() -> list[dict]:
    adapter = get_adapter()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()
    timesheets = adapter.get_timesheets()
    projects = adapter.get_projects()

    # An allocation row can still carry is_allocation_active=1 even after the employee
    # has genuinely departed (their resignation date has passed) -- the allocation was
    # never formally closed out. Without this filter, departed people keep showing up
    # as "available" in the free pool and elsewhere downstream of this report.
    currently_active_ids = set(employees[employees["account_status"] == 1]["employee_id"])

    active = allocations[
        (allocations["is_allocation_active"] == 1) & (allocations["employee_id"].isin(currently_active_ids))
    ].copy()

    active = active.merge(
        projects[["project_code", "type_of_project", "tech_coe"]].rename(columns={"project_code": "project_id"}),
        on="project_id", how="left",
    )

    employee_total_pct = (
        active.groupby("employee_id")["allocation_by_percentage"].sum().rename("employee_total_allocation_pct")
    )
    client_rows = active[active["type_of_project"] != INTERNAL_PROJECT_TYPE]
    employee_client_pct = (
        client_rows.groupby("employee_id")["allocation_by_percentage"].sum().rename("employee_client_allocation_pct")
    )
    active = active.merge(employee_total_pct, on="employee_id", how="left")
    active = active.merge(employee_client_pct, on="employee_id", how="left")
    active["employee_client_allocation_pct"] = active["employee_client_allocation_pct"].fillna(0.0)
    active["employee_internal_allocation_pct"] = (
        active["employee_total_allocation_pct"] - active["employee_client_allocation_pct"]
    ).round(2)
    active["over_allocated_due_to_internal"] = (active["employee_total_allocation_pct"] > OVER_ALLOCATED_THRESHOLD) & (
        active["employee_client_allocation_pct"] <= OVER_ALLOCATED_THRESHOLD
    )
    active["utilization_band"] = [
        _utilization_band(t, c)
        for t, c in zip(active["employee_total_allocation_pct"], active["employee_client_allocation_pct"])
    ]

    today = pd.Timestamp.now().normalize()
    active = _hours_metrics(active, timesheets, today)

    active = active.merge(
        employees[["employee_id", "job_name", "department_name", "location"]],
        on="employee_id", how="left",
    )

    active["days_to_end"] = (active["allocated_end_date"] - today).dt.days
    active["ending_soon"] = active["days_to_end"].between(0, ENDING_SOON_DAYS)

    cols = [
        "employee_id", "job_name", "department_name", "location", "project_id", "type_of_project",
        "resourcing_status", "allocation_by_percentage", "allocated_start_date",
        "allocated_end_date", "employee_total_allocation_pct", "employee_client_allocation_pct",
        "employee_internal_allocation_pct", "over_allocated_due_to_internal", "utilization_band",
        "actual_hours_logged", "expected_hours", "hours_utilization_pct", "hours_data_available",
        "possible_unplanned_absence", "days_to_end", "ending_soon",
    ]
    coe_values = [canonical_project_coe(v) for v in active["tech_coe"].tolist()]

    out = active[cols].copy()
    for date_col in ["allocated_start_date", "allocated_end_date"]:
        out[date_col] = out[date_col].dt.strftime("%Y-%m-%d")
    out["hours_utilization_pct"] = out["hours_utilization_pct"].where(out["hours_utilization_pct"].notna(), None)
    out["type_of_project"] = out["type_of_project"].where(out["type_of_project"].notna(), None)
    records = out.to_dict(orient="records")
    for record, coe in zip(records, coe_values):
        record["coe"] = coe
    return records


def get_availability_as_of(as_of_date: pd.Timestamp | None = None) -> list[dict]:
    """Who is available (and how available) on a given date -- not just today.
    'Available' is judged on client_pct only, same convention as utilization_band:
    internal-project work is discretionary and shouldn't count against availability."""
    adapter = get_adapter()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()
    projects = adapter.get_projects()

    as_of = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.now().normalize()
    logger.warning(f"[availability] as_of={as_of}, allocations dtype={allocations['allocated_start_date'].dtype}, {allocations['allocated_end_date'].dtype}")

    if "date_of_resignation" in employees.columns:
        resignation = pd.to_datetime(employees["date_of_resignation"], errors="coerce")
        will_still_be_active = (employees["account_status"] == 1) & (resignation.isna() | (resignation > as_of))
        currently_active_ids = set(employees[will_still_be_active]["employee_id"])
    else:
        currently_active_ids = set(employees[employees["account_status"] == 1]["employee_id"])

    logger.warning(f"[availability] as_of={as_of}, currently_active_ids count={len(currently_active_ids)}")

    covering = allocations[
        (allocations["allocated_start_date"] <= as_of)
        & (allocations["allocated_end_date"] >= as_of)
        & (allocations["is_allocation_active"] == 1)
        & (allocations["employee_id"].isin(currently_active_ids))
    ].copy()

    logger.warning(f"[availability] as_of={as_of}, covering rows={len(covering)}")

    covering = covering.merge(
        projects[["project_code", "type_of_project"]].rename(columns={"project_code": "project_id"}),
        on="project_id", how="left",
    )

    total_pct = covering.groupby("employee_id")["allocation_by_percentage"].sum().rename("total_allocated_pct")
    client_rows = covering[covering["type_of_project"] != INTERNAL_PROJECT_TYPE]
    client_pct = client_rows.groupby("employee_id")["allocation_by_percentage"].sum().rename("client_allocated_pct")

    projects_by_employee: dict[str, list[dict]] = {}
    for emp_id, group in covering.groupby("employee_id"):
        projects_by_employee[emp_id] = [
            {
                "project_id": row["project_id"],
                "type_of_project": row.get("type_of_project"),
                "allocation_by_percentage": row["allocation_by_percentage"],
                "resourcing_status": row["resourcing_status"],
            }
            for _, row in group.iterrows()
        ]

    emp_df = employees[employees["employee_id"].isin(currently_active_ids)][
        ["employee_id", "job_name", "department_name", "location"]
    ].copy()
    emp_df = emp_df.merge(total_pct, on="employee_id", how="left")
    emp_df = emp_df.merge(client_pct, on="employee_id", how="left")
    emp_df["total_allocated_pct"] = emp_df["total_allocated_pct"].fillna(0.0)
    emp_df["client_allocated_pct"] = emp_df["client_allocated_pct"].fillna(0.0)
    emp_df["available_pct"] = (100 - emp_df["client_allocated_pct"]).clip(lower=0)
    emp_df["is_fully_free"] = emp_df["client_allocated_pct"] <= 0
    logger.warning(f"[availability] as_of={as_of}, fully_free count={(emp_df['is_fully_free']).sum()}, avg available_pct={emp_df['available_pct'].mean():.1f}")

    records = emp_df.to_dict(orient="records")
    for r in records:
        r["as_of_date"] = as_of.strftime("%Y-%m-%d")
        r["current_projects"] = projects_by_employee.get(r["employee_id"], [])
    records.sort(key=lambda r: r["available_pct"], reverse=True)
    return records