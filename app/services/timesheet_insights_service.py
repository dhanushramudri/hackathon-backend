import pandas as pd

from app.core.adapter import get_adapter

MAX_PLAUSIBLE_DAILY_HOURS = 24

OVERTIME_DAILY_HOURS_THRESHOLD = 9
SUSTAINED_OVERTIME_WINDOW_DAYS = 14
SUSTAINED_OVERTIME_MIN_DAYS = 4

EFFORT_SPIKE_RATIO_THRESHOLD = 1.5
EFFORT_SPIKE_MIN_BASELINE_WEEKS = 3

def _clean_daily_hours(timesheets: pd.DataFrame) -> pd.Series:
    daily = timesheets.groupby(["employee_id", "date"])["time"].sum()
    return daily[daily <= MAX_PLAUSIBLE_DAILY_HOURS]

def get_employee_overtime_risk() -> dict[str, dict]:
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    daily = _clean_daily_hours(timesheets).reset_index(name="hours")

    today = pd.Timestamp.now().normalize()
    window_start = today - pd.Timedelta(days=SUSTAINED_OVERTIME_WINDOW_DAYS)
    recent = daily[(daily["date"] >= window_start) & (daily["date"] <= today)]

    is_overtime = recent["hours"] > OVERTIME_DAILY_HOURS_THRESHOLD
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

    today = pd.Timestamp.now().normalize()
    window_start = today - pd.Timedelta(days=SUSTAINED_OVERTIME_WINDOW_DAYS)
    rows = daily[
        (daily["employee_id"] == employee_id) & (daily["date"] >= window_start) & (daily["date"] <= today)
    ].sort_values("date")

    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "hours": float(round(h, 1)),
            "is_overtime": bool(h > OVERTIME_DAILY_HOURS_THRESHOLD),
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

    today = pd.Timestamp.now().normalize()
    window_start = today - pd.Timedelta(days=SUSTAINED_OVERTIME_WINDOW_DAYS)
    recent = ts[(ts["date"] >= window_start) & (ts["date"] <= today)]

    by_project = recent.groupby("project_id")["time"].sum().sort_values(ascending=False)
    return [{"project_id": pid, "hours_recent": float(round(h, 1))} for pid, h in by_project.items()]

def get_project_weekly_hours(project_id: str, n_weeks: int = 8) -> list[dict]:
    adapter = get_adapter()
    timesheets = adapter.get_timesheets()
    ts = timesheets.dropna(subset=["date", "project_id"])
    ts = ts[ts["project_id"] == project_id].copy()
    ts["week"] = ts["date"].dt.to_period("W")

    weekly = ts.groupby("week")["time"].sum().reset_index().sort_values("week").tail(n_weeks)
    return [{"week": str(w), "hours": float(round(h, 1))} for w, h in zip(weekly["week"], weekly["time"])]
