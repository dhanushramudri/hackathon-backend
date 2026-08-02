import pandas as pd

from app.core.adapter import get_adapter

MAX_PLAUSIBLE_DAILY_HOURS = 24

OVERTIME_DAILY_HOURS_THRESHOLD = 11
SUSTAINED_OVERTIME_WINDOW_DAYS = 14
SUSTAINED_OVERTIME_MIN_DAYS = 4

EFFORT_SPIKE_RATIO_THRESHOLD = 1.5
EFFORT_SPIKE_MIN_BASELINE_WEEKS = 3

def _clean_daily_hours(timesheets: pd.DataFrame) -> pd.Series:
    daily = timesheets.groupby(["employee_id", "date"])["time"].sum()
    return daily[daily <= MAX_PLAUSIBLE_DAILY_HOURS]

def _latest_timesheet_date(timesheets: pd.DataFrame) -> pd.Timestamp:
    """Anchored on the data's own latest entry, not wall-clock "now" -- these
    are fixed snapshots that stop well before "today" ever ticks past them, so
    a calendar-relative window would silently go empty once real time moves on."""
    return timesheets["date"].max().normalize()

def get_employee_overtime_risk() -> dict[str, dict]:
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    daily = _clean_daily_hours(timesheets).reset_index(name="hours")

    today = _latest_timesheet_date(timesheets)
    window_start = today - pd.Timedelta(days=SUSTAINED_OVERTIME_WINDOW_DAYS)
    recent = daily[(daily["date"] >= window_start) & (daily["date"] <= today)]

    is_overtime = recent["hours"] >= OVERTIME_DAILY_HOURS_THRESHOLD
    overtime_days = recent[is_overtime].groupby("employee_id").size().rename("overtime_days_recent")
    max_hours = recent.groupby("employee_id")["hours"].max().rename("max_daily_hours_recent")

    summary = pd.concat([overtime_days, max_hours], axis=1).fillna(0)
    summary["overtime_days_recent"] = summary["overtime_days_recent"].astype(int)
    summary["is_sustained_overtime"] = summary["overtime_days_recent"] >= SUSTAINED_OVERTIME_MIN_DAYS

    return {
        emp_id: {
            "overtime_days_recent": int(row["overtime_days_recent"]),
            "max_daily_hours_recent": float(round(row["max_daily_hours_recent"], 1)),
            "is_sustained_overtime": bool(row["is_sustained_overtime"]),
        }
        for emp_id, row in summary.iterrows()
    }

def get_project_effort_spikes() -> dict[str, dict]:
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    ts = timesheets.dropna(subset=["date", "project_id"]).copy()
    ts["week"] = ts["date"].dt.to_period("W")

    weekly = ts.groupby(["project_id", "week"])["time"].sum().reset_index()
    weekly = weekly.sort_values(["project_id", "week"])

    result: dict[str, dict] = {}
    for project_id, group in weekly.groupby("project_id"):
        if len(group) < EFFORT_SPIKE_MIN_BASELINE_WEEKS + 1:
            continue
        latest = group.iloc[-1]
        baseline = group.iloc[-(EFFORT_SPIKE_MIN_BASELINE_WEEKS + 1):-1]["time"].mean()
        if baseline <= 0:
            continue
        ratio = latest["time"] / baseline
        result[project_id] = {
            "latest_week_hours": float(round(latest["time"], 1)),
            "baseline_avg_weekly_hours": float(round(baseline, 1)),
            "is_effort_spike": bool(ratio > EFFORT_SPIKE_RATIO_THRESHOLD),
        }
    return result

def get_employee_recent_daily_hours(employee_id: str) -> list[dict]:
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    daily = _clean_daily_hours(timesheets).reset_index(name="hours")

    today = _latest_timesheet_date(timesheets)
    window_start = today - pd.Timedelta(days=SUSTAINED_OVERTIME_WINDOW_DAYS)
    rows = daily[
        (daily["employee_id"] == employee_id) & (daily["date"] >= window_start) & (daily["date"] <= today)
    ].sort_values("date")

    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "hours": float(round(h, 1)),
            "is_overtime": bool(h >= OVERTIME_DAILY_HOURS_THRESHOLD),
        }
        for d, h in zip(rows["date"], rows["hours"])
    ]

def get_employee_recent_projects(employee_id: str) -> list[dict]:
    """Which project(s) this employee actually logged hours against in the recent
    overtime window, ranked by hours -- lets the wellbeing page point an overworked
    employee at the specific project where relief staffing would help them."""
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    ts = timesheets.dropna(subset=["date", "project_id"])
    ts = ts[ts["employee_id"] == employee_id]

    today = _latest_timesheet_date(timesheets)
    window_start = today - pd.Timedelta(days=SUSTAINED_OVERTIME_WINDOW_DAYS)
    recent = ts[(ts["date"] >= window_start) & (ts["date"] <= today)]

    by_project = recent.groupby("project_id")["time"].sum().sort_values(ascending=False)
    return [{"project_id": pid, "hours_recent": float(round(h, 1))} for pid, h in by_project.items()]

def get_employee_timesheet_entries(
    employee_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    project_id: str | None = None,
    billing_status: str | None = None,
) -> dict:
    """Real per-day timesheet rows for one employee, filterable by date range,
    project, and billing status -- the raw proof surface behind the profile
    modal's Timesheet tab, not a derived/aggregate signal."""
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    ts = timesheets[timesheets["employee_id"] == employee_id].copy()
    # A real, sizeable share of raw rows have no billing_status recorded --
    # an honest "Not set" bucket beats silently dropping them from groupby/filter.
    ts["billing_status"] = ts["billing_status"].fillna("NOT_SET")

    available_projects = sorted(ts["project_id"].dropna().unique().tolist())
    data_min_date = ts["date"].min()
    data_max_date = ts["date"].max()

    if project_id:
        ts = ts[ts["project_id"] == project_id]
    if billing_status:
        ts = ts[ts["billing_status"] == billing_status]
    if start_date:
        ts = ts[ts["date"] >= pd.Timestamp(start_date)]
    if end_date:
        ts = ts[ts["date"] <= pd.Timestamp(end_date)]

    ts = ts.sort_values("date")

    rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "project_id": pid,
            "job_name": job,
            "hours": float(round(h, 2)),
            "status": status,
            "billing_status": billing,
        }
        for d, pid, job, h, status, billing in zip(
            ts["date"], ts["project_id"], ts["job_name"], ts["time"], ts["status"], ts["billing_status"]
        )
    ]

    days_logged = int(ts["date"].nunique())
    by_project = ts.groupby("project_id")["time"].sum().sort_values(ascending=False) if not ts.empty else pd.Series(dtype=float)
    by_billing_status = ts.groupby("billing_status")["time"].sum() if not ts.empty else pd.Series(dtype=float)

    return {
        "employee_id": employee_id,
        "total_hours": float(round(ts["time"].sum(), 2)) if not ts.empty else 0.0,
        "days_logged": days_logged,
        "entry_count": int(len(ts)),
        "avg_hours_per_logged_day": float(round(ts["time"].sum() / days_logged, 2)) if days_logged > 0 else 0.0,
        "data_start_date": data_min_date.strftime("%Y-%m-%d") if pd.notna(data_min_date) else None,
        "data_end_date": data_max_date.strftime("%Y-%m-%d") if pd.notna(data_max_date) else None,
        "available_projects": available_projects,
        "by_project": [{"project_id": pid, "hours": float(round(h, 2))} for pid, h in by_project.items()],
        "by_billing_status": {k: float(round(v, 2)) for k, v in by_billing_status.items()},
        "rows": rows,
    }

def get_project_weekly_hours(project_id: str, n_weeks: int = 8) -> list[dict]:
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    ts = timesheets.dropna(subset=["date", "project_id"])
    ts = ts[ts["project_id"] == project_id].copy()
    ts["week"] = ts["date"].dt.to_period("W")

    weekly = ts.groupby("week")["time"].sum().reset_index().sort_values("week").tail(n_weeks)
    return [{"week": str(w), "hours": float(round(h, 1))} for w, h in zip(weekly["week"], weekly["time"])]
