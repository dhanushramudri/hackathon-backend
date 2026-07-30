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
PROJECTS_CSV = TRANSFORMED_DIR / "02_Project_Details_clean.csv"

class AllocationRowNotFound(Exception):
    def __init__(self, allocation_id: str):
        self.allocation_id = allocation_id
        super().__init__(f"allocation {allocation_id!r} not found")

class ProjectNotFoundForExtension(Exception):
    def __init__(self, project_code: str):
        self.project_code = project_code
        super().__init__(f"project_code {project_code!r} not found")

EXTENSION_STATUSES = {"BILLABLE", "UNBILLABLE"}
ALLOCATION_EXTENSION_STATUSES = {"BILLABLE", "UNBILLABLE", "SHADOW"}

def extend_allocation_end_date(allocation_id: str, extended_end_date: str, status: str | None = None) -> dict:
    """Sets (or clears, when extended_end_date is falsy) the extended_end_date on a
    single allocation row -- the resource manager's own record of a real-world
    extension for that person on that project, kept separate from the original
    allocated_end_date so both remain visible. Setting a date requires a status
    for the extension period (billable/unbillable/shadow -- unlike a project,
    an individual allocation can be shadow work), stored separately from the
    row's own resourcing_status so the original allocation's status is untouched.

    Gated on the PROJECT already having its own extended_end_date set: a person
    can't be extended past a project that hasn't itself been formally extended,
    and can't be extended further out than the project's own extension covers."""
    alloc_df = pd.read_csv(ALLOCATIONS_CSV, dtype=str)
    alloc_df.columns = [c.strip() for c in alloc_df.columns]
    ids = alloc_df["project_rolebased_user_id"].str.strip()
    matches = alloc_df[ids == allocation_id.strip()]
    if matches.empty:
        raise AllocationRowNotFound(allocation_id)
    row_idx = matches.index[0]

    if extended_end_date:
        if status not in ALLOCATION_EXTENSION_STATUSES:
            raise ValueError(f"status must be one of {sorted(ALLOCATION_EXTENSION_STATUSES)}")

        project_id = str(alloc_df.at[row_idx, "project_id"]).strip()
        proj_df = pd.read_csv(PROJECTS_CSV, dtype=str)
        proj_df.columns = [c.strip() for c in proj_df.columns]
        proj_matches = proj_df[proj_df["project_code"].str.strip() == project_id]
        project_extended_end = (
            pd.to_datetime(str(proj_matches.iloc[0]["extended_end_date"]).strip(), errors="coerce")
            if not proj_matches.empty else pd.NaT
        )
        if pd.isna(project_extended_end):
            raise ValueError(
                f"Project {project_id!r} has not been extended yet -- extend the project's end date first."
            )

        current_end = pd.to_datetime(str(alloc_df.at[row_idx, "allocated_end_date"]).strip(), errors="coerce")
        new_end = pd.to_datetime(extended_end_date, errors="coerce")
        if pd.isna(new_end):
            raise ValueError(f"invalid extended_end_date {extended_end_date!r}")
        if pd.notna(current_end) and new_end < current_end:
            raise ValueError("extended_end_date cannot be before the current allocated_end_date")
        if new_end > project_extended_end:
            raise ValueError(
                f"extended_end_date cannot be later than the project's extended end date "
                f"({project_extended_end.strftime('%Y-%m-%d')})"
            )
        # extended_start_date is never entered -- it's always the day right after
        # this allocation's own current end date, i.e. where the extension period
        # begins. Computed here, not by the caller, so it can't drift out of sync.
        extended_start = current_end + pd.Timedelta(days=1) if pd.notna(current_end) else new_end
        alloc_df.at[row_idx, "extended_end_date"] = new_end.strftime("%Y-%m-%d")
        alloc_df.at[row_idx, "extended_start_date"] = extended_start.strftime("%Y-%m-%d")
        alloc_df.at[row_idx, "extended_status"] = status
    else:
        alloc_df.at[row_idx, "extended_end_date"] = ""
        alloc_df.at[row_idx, "extended_start_date"] = ""
        alloc_df.at[row_idx, "extended_status"] = ""

    alloc_df.to_csv(ALLOCATIONS_CSV, index=False)
    db_module.reload()
    return {
        "allocation_id": allocation_id,
        "extended_start_date": alloc_df.at[row_idx, "extended_start_date"] or None,
        "extended_end_date": alloc_df.at[row_idx, "extended_end_date"] or None,
        "extended_status": alloc_df.at[row_idx, "extended_status"] or None,
    }

def extend_project_end_date(project_code: str, extended_end_date: str, status: str | None = None) -> dict:
    """Sets (or clears) the extended_end_date on a project -- an explicit,
    resource-manager-entered override of the project's currently-expected end
    date, kept separate from the original project_end_date. Setting a date
    requires a billable/unbillable status for that extension period (a project
    has no SHADOW concept -- that's an individual allocation's resourcing_status,
    not a whole project's)."""
    df = pd.read_csv(PROJECTS_CSV, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    codes = df["project_code"].str.strip()
    matches = df[codes == project_code.strip()]
    if matches.empty:
        raise ProjectNotFoundForExtension(project_code)
    row_idx = matches.index[0]

    if extended_end_date:
        if status not in EXTENSION_STATUSES:
            raise ValueError(f"status must be one of {sorted(EXTENSION_STATUSES)}")
        current_end = pd.to_datetime(str(df.at[row_idx, "project_end_date"]).strip(), errors="coerce")
        new_end = pd.to_datetime(extended_end_date, errors="coerce")
        if pd.isna(new_end):
            raise ValueError(f"invalid extended_end_date {extended_end_date!r}")
        if pd.notna(current_end) and new_end < current_end:
            raise ValueError("extended_end_date cannot be before the current project_end_date")
        df.at[row_idx, "extended_end_date"] = new_end.strftime("%Y-%m-%d")
        df.at[row_idx, "extended_end_status"] = status
    else:
        df.at[row_idx, "extended_end_date"] = ""
        df.at[row_idx, "extended_end_status"] = ""

    df.to_csv(PROJECTS_CSV, index=False)
    db_module.reload()
    return {
        "project_code": project_code,
        "extended_end_date": df.at[row_idx, "extended_end_date"] or None,
        "extended_end_status": df.at[row_idx, "extended_end_status"] or None,
    }

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
        projects[["project_code", "type_of_project", "tech_coe", "project_end_date", "extended_end_date", "extended_end_status"]]
        .rename(columns={
            "project_code": "project_id",
            "extended_end_date": "project_extended_end_date",
            "extended_end_status": "project_extended_end_status",
        }),
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
        "project_rolebased_user_id", "employee_id", "job_name", "department_name", "location", "project_id", "type_of_project",
        "resourcing_status", "allocation_by_percentage", "allocated_start_date",
        "allocated_end_date", "extended_start_date", "extended_end_date", "extended_status", "project_end_date", "project_extended_end_date",
        "project_extended_end_status",
        "employee_total_allocation_pct", "employee_client_allocation_pct",
        "employee_internal_allocation_pct", "over_allocated_due_to_internal", "utilization_band",
        "actual_hours_logged", "expected_hours", "hours_utilization_pct", "hours_data_available",
        "possible_unplanned_absence", "days_to_end", "ending_soon",
    ]
    coe_values = [canonical_project_coe(v) for v in active["tech_coe"].tolist()]

    out = active[cols].copy()
    for date_col in ["allocated_start_date", "allocated_end_date"]:
        out[date_col] = out[date_col].dt.strftime("%Y-%m-%d")
    for nullable_date_col in ["extended_start_date", "extended_end_date", "project_end_date", "project_extended_end_date"]:
        out[nullable_date_col] = out[nullable_date_col].dt.strftime("%Y-%m-%d").where(out[nullable_date_col].notna(), None)
    out["hours_utilization_pct"] = out["hours_utilization_pct"].where(out["hours_utilization_pct"].notna(), None)
    out["type_of_project"] = out["type_of_project"].where(out["type_of_project"].notna(), None)
    out["project_extended_end_status"] = out["project_extended_end_status"].where(out["project_extended_end_status"].notna(), None)
    out["extended_status"] = out["extended_status"].where(out["extended_status"].notna(), None)
    out = out.rename(columns={"project_rolebased_user_id": "allocation_id"})
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