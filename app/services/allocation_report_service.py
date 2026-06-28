import numpy as np
import pandas as pd

from app.core.adapter import get_adapter
from app.engines.role_mix_engine import canonical_project_coe

ENDING_SOON_DAYS = 30
OVER_ALLOCATED_THRESHOLD = 100
UNDER_UTILIZED_THRESHOLD = 70
STANDARD_HOURS_PER_DAY = 8
UNPLANNED_ABSENCE_WINDOW_DAYS = 14

def _utilization_band(pct: float) -> str:
    if pct > OVER_ALLOCATED_THRESHOLD:
        return "over_allocated"
    if pct < UNDER_UTILIZED_THRESHOLD:
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

    active = allocations[allocations["is_allocation_active"] == 1].copy()

    employee_total_pct = (
        active.groupby("employee_id")["allocation_by_percentage"].sum().rename("employee_total_allocation_pct")
    )
    active = active.merge(employee_total_pct, on="employee_id", how="left")
    active["utilization_band"] = active["employee_total_allocation_pct"].apply(_utilization_band)

    today = pd.Timestamp.now().normalize()
    active = _hours_metrics(active, timesheets, today)

    active = active.merge(
        employees[["employee_id", "job_name", "department_name", "location"]],
        on="employee_id", how="left",
    )
    active = active.merge(
        projects[["project_code", "type_of_project", "tech_coe"]].rename(columns={"project_code": "project_id"}),
        on="project_id", how="left",
    )

    active["days_to_end"] = (active["allocated_end_date"] - today).dt.days
    active["ending_soon"] = active["days_to_end"].between(0, ENDING_SOON_DAYS)

    cols = [
        "employee_id", "job_name", "department_name", "location", "project_id", "type_of_project",
        "resourcing_status", "allocation_by_percentage", "allocated_start_date",
        "allocated_end_date", "employee_total_allocation_pct", "utilization_band",
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
