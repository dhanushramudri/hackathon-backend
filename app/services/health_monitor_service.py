"""
health_service.py  —  modified to include DevOps board extension-risk signal.

CHANGES vs original (search for  ← NEW  to find every addition):
  1. Import devops_insights_service                         (top of file)
  2. fetch_open_devops_tickets() called ONCE before loop   (in get_health_report)
  3. compute_devops_extension_risk() called per project    (inside loop)
  4. "devops_extension_risk" added to root_causes          (inside loop)
  5. DevOps fields added to returned record dict           (inside loop)

All other logic is UNCHANGED from the original.
"""

import os 
import numpy as np                                                                     # ← needed for AZURE_DEVOPS_PAT check
import math
import pandas as pd

from app.core.adapter import get_adapter
from app.engines.pulse_engine import get_project_pulse_table
from app.engines.role_mix_engine import build_role_mix_templates, canonical_project_coe, get_role_mix
from app.engines.sentiment_engine import analyze_comment, summarize_project_sentiment
from app.services.rate_card_service import get_hourly_rate
from app.services.timesheet_insights_service import get_employee_overtime_risk, get_project_effort_spikes

# ← NEW: DevOps board extension-risk service
from app.services.devops_insights_service import (
    compute_devops_extension_risk,
    compute_devops_ticket_stats,
    fetch_open_devops_tickets,
    fetch_open_devops_tickets_cached,
    group_tickets_by_project_code,
    no_devops_config_risk,
)

OVERRUN_DAYS_THRESHOLD = 14
SHADOW_SHARE_THRESHOLD = 0.3
RAMP_DOWN_WINDOW_DAYS = 30
UNDERSTAFFED_RATIO_THRESHOLD = 0.75
STANDARD_MONTHLY_HOURS = 160
EXTENSION_DAILY_HOURS = 8.0 

# ── DEMO ONLY ────────────────────────────────────────────────────────────
# Statically maps every project's DevOps signal to this one real, busy
# DevOps project code so every card in the health monitor shows meaningful
# ticket data during demos. Remove this override (and revert the lookup
# below to `_devops_tickets_by_project.get(project_code, [])`) once each
# project has its own real, correctly-tagged AreaPath on the board.
DEMO_STATIC_DEVOPS_PROJECT_CODE = "JMG_242"
# ── END DEMO ─────────────────────────────────────────────────────────────

WSR_TREND_RECENT_REPORTS = 3
WSR_TREND_BASELINE_REPORTS = 3
WSR_TREND_LOOKBACK_REPORTS = WSR_TREND_RECENT_REPORTS + WSR_TREND_BASELINE_REPORTS
WSR_TREND_SHIFT_THRESHOLD = 0.3

WSR_CRITICAL_SEVERITY_THRESHOLD = 1.0
WSR_CRITICAL_MIN_REPORTS = WSR_TREND_RECENT_REPORTS

WSR_BASELINE_REPORTS = 3
WSR_LONG_TERM_MIN_REPORTS = WSR_BASELINE_REPORTS + WSR_TREND_RECENT_REPORTS


# Explicit category per root cause -- lets callers ask "is this project at risk
# of running LONG" (extension) vs "does this need someone's attention right
# now" (escalation) vs other operational risk, instead of one undifferentiated
# risk_score. A project can fire causes in multiple categories at once.
ROOT_CAUSE_CATEGORY: dict[str, str] = {
    "overrunning": "extension",
    "devops_extension_risk": "extension",
    "wsr_deteriorating": "escalation",
    "wsr_critical": "escalation",
    "wsr_long_term_decline": "escalation",
    "effort_spike": "escalation",
    "shadow_heavy": "financial",
    "understaffed": "staffing",
    "high_churn": "staffing",
    "overtime_risk": "people",
    "pulse_risk": "people",
}

def categorize_root_causes(root_causes: list[str]) -> dict[str, list[str]]:
    """Groups a project's fired root causes by category. A project can appear
    in multiple categories at once -- e.g. overrunning (extension) AND
    wsr_critical (escalation) simultaneously; that's real and shouldn't be
    collapsed into one label."""
    grouped: dict[str, list[str]] = {}
    for cause in root_causes:
        category = ROOT_CAUSE_CATEGORY.get(cause, "other")
        grouped.setdefault(category, []).append(cause)
    return grouped

_RAG_SEVERITY = {"RED": 2, "AMBER": 1, "GREEN": 0, "NO_COLOR": -1}
_SEVERITY_TO_STATUS = {2: "RED", 1: "AMBER", 0: "GREEN", -1: "NO_COLOR"}
_RAG_COLUMNS = ["scope_status", "schedule_status", "quality_status", "csat_status", "team_status"]


def _date_str(value) -> str | None:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else None

def worst_wsr_signal_vectorized(wsr: pd.DataFrame) -> pd.Series:
    severities = wsr[_RAG_COLUMNS].apply(lambda col: col.map(_RAG_SEVERITY))
    return severities.max(axis=1).map(_SEVERITY_TO_STATUS)


def wsr_severity_rows(wsr: pd.DataFrame) -> pd.DataFrame:
    df = wsr.copy()
    df["severity"] = df[_RAG_COLUMNS].apply(lambda col: col.map(_RAG_SEVERITY)).max(axis=1)
    return df[df["severity"] >= 0].sort_values("week_start_date")


def trend_from_severity_series(severities: pd.Series) -> dict:
    n = len(severities)
    recent_avg_severity = float(severities.iloc[-WSR_TREND_RECENT_REPORTS:].mean()) if n >= WSR_CRITICAL_MIN_REPORTS else None
    is_critical = recent_avg_severity is not None and recent_avg_severity >= WSR_CRITICAL_SEVERITY_THRESHOLD

    baseline_avg_severity = None
    is_long_term_decline = False
    if n >= WSR_LONG_TERM_MIN_REPORTS:
        baseline_avg_severity = float(severities.iloc[:WSR_BASELINE_REPORTS].mean())
        is_long_term_decline = recent_avg_severity > baseline_avg_severity + WSR_TREND_SHIFT_THRESHOLD

    if n < WSR_TREND_LOOKBACK_REPORTS:
        return {
            "trend": None,
            "recent_avg_severity": round(recent_avg_severity, 2) if recent_avg_severity is not None else None,
            "prior_avg_severity": None,
            "is_critical": is_critical,
            "baseline_avg_severity": round(baseline_avg_severity, 2) if baseline_avg_severity is not None else None,
            "is_long_term_decline": is_long_term_decline,
        }
    window = severities.iloc[-WSR_TREND_LOOKBACK_REPORTS:]
    prior_avg_severity = float(window.iloc[:-WSR_TREND_RECENT_REPORTS].mean())
    if recent_avg_severity > prior_avg_severity + WSR_TREND_SHIFT_THRESHOLD:
        trend = "deteriorating"
    elif recent_avg_severity < prior_avg_severity - WSR_TREND_SHIFT_THRESHOLD:
        trend = "improving"
    else:
        trend = "stable"
    return {
        "trend": trend,
        "recent_avg_severity": round(recent_avg_severity, 2),
        "prior_avg_severity": round(prior_avg_severity, 2),
        "is_critical": is_critical,
        "baseline_avg_severity": round(baseline_avg_severity, 2) if baseline_avg_severity is not None else None,
        "is_long_term_decline": is_long_term_decline,
    }


def wsr_trend(wsr: pd.DataFrame) -> dict[str, dict]:
    df = wsr_severity_rows(wsr)
    results: dict[str, dict] = {}
    for project_id, group in df.groupby("project_id_masked"):
        result = trend_from_severity_series(group["severity"])
        if result["recent_avg_severity"] is not None:
            results[project_id] = result
    return results


def churn_p75_threshold() -> float:
    adapter = get_adapter()
    projects = adapter.get_projects()
    allocations = adapter.get_allocations()
    active = projects[projects["project_status"] == "ACTIVE"].copy()
    n_employees = allocations.groupby("project_id")["employee_id"].nunique().rename("n_employees")
    active = active.merge(n_employees, left_on="project_code", right_index=True, how="left")
    duration_days = (active["project_end_date"] - active["project_start_date"]).dt.days.clip(lower=1)
    churn_per_month = active["n_employees"] / (duration_days / 30)
    return round(float(churn_per_month.quantile(0.75)), 2)

def count_working_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Weekday (Mon-Fri) count between start and end, inclusive of both ends.
    Weekends are never counted -- this keeps the daily-charge revenue-loss
    math aligned with a real 40h/week at 100% allocation, not calendar days."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    return int(np.busday_count(start.date(), (end + pd.Timedelta(days=1)).date()))


def add_working_days(start: pd.Timestamp, working_days: float | None) -> pd.Timestamp | None:
    """Advances `start` forward by `working_days` (Mon-Fri only). Rounds UP to
    the next whole day since np.busday_offset needs an integer offset --
    ceiling (not rounding) so a fractional shortfall (e.g. 2.3 days) is never
    displayed as finishing a day earlier than it actually would."""
    if pd.isna(start) or not working_days or working_days <= 0:
        return None
    n = math.ceil(working_days)
    result = np.busday_offset(start.date(), n, roll="forward")
    return pd.Timestamp(result)


def extension_duration_label(working_days: float | None) -> str | None:
    """Human label for a working-day duration, as whole weeks + whole days
    (e.g. '3d', '1w', '1w 4d'). Days are always rounded UP -- any fraction
    of a day booked counts as a full day (2.4 -> 3). A day-remainder of 4
    or more out of a 5-day week rounds up into the next full week (4d -> 1w,
    1w+4d -> 2w); a remainder under 4 stays within the current week,
    rounded up (1w+3.5d -> '1w 4d', not bumped to 2w)."""
    if not working_days or working_days <= 0:
        return None

    weeks, remainder = divmod(working_days, 5)
    weeks = int(weeks)
    if remainder >= 4:
        weeks += 1
        remainder = 0
    else:
        remainder = math.ceil(remainder)

    if weeks == 0:
        return f"{remainder}d"
    if remainder == 0:
        return f"{weeks}w"
    return f"{weeks}w {remainder}d"

def _projected_additional_extension_days(devops_risk: dict) -> tuple[float | None, str]:
    """Working-day forecast of how much LONGER a project will run, from DevOps
    ticket effort vs. the team's current daily capacity. This is the same
    calculation the project detail page uses for its 'Extension outlook'
    panel (see _estimate_extension in project_detail_service.py) -- kept as a
    pure, read-only duplicate here so the health-report revenue prediction
    below can use it without importing project_detail_service.py (which
    already imports FROM this file, so importing back would be circular)."""
    daily_rate = devops_risk.get("team_daily_capacity_hours") or 0.0
    remaining_hours = devops_risk.get("remaining_effort_hours") or 0.0
    is_overdue = devops_risk.get("is_overdue", False)
    within_window = devops_risk.get("within_risk_window", False)
    capacity_after_leave = devops_risk.get("team_capacity_hours_after_leave") or 0.0

    if daily_rate <= 0 or not (is_overdue or within_window):
        return None, "none"

    shortfall_hours = remaining_hours if is_overdue else max(0.0, remaining_hours - capacity_after_leave)
    if shortfall_hours <= 0:
        return None, "none"

    projected_days = round(shortfall_hours / daily_rate, 1)
    open_count = devops_risk.get("open_ticket_count") or 0
    gap = (devops_risk.get("tickets_missing_remaining_estimate", 0) + devops_risk.get("tickets_with_no_effort_data", 0))
    gap_ratio = gap / open_count if open_count else 1.0
    confidence = "low" if gap_ratio > 0.3 else "medium"
    return projected_days, confidence

def get_health_report() -> list[dict]:
    adapter = get_adapter()
    projects = adapter.get_projects()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()
    wsr = adapter.get_wsr_reports()

    active = projects[projects["project_status"] == "ACTIVE"].copy()

    role_mix_templates = build_role_mix_templates()

    alloc_with_rate = allocations.merge(employees[["employee_id", "job_name"]], on="employee_id", how="left")
    alloc_with_rate["hourly_rate"] = alloc_with_rate["job_name"].apply(get_hourly_rate)
    is_unbilled = alloc_with_rate["resourcing_status"].isin(["SHADOW", "UNBILLED"])
    alloc_with_rate["unbilled_monthly_value"] = (
        is_unbilled * (alloc_with_rate["allocation_by_percentage"] / 100) * alloc_with_rate["hourly_rate"].fillna(0) * STANDARD_MONTHLY_HOURS
    )

    n_employees = allocations.groupby("project_id")["employee_id"].nunique().rename("n_employees")
    max_alloc_end = allocations.groupby("project_id")["allocated_end_date"].max().rename("max_alloc_end")
    shadow_share = (
        allocations.assign(is_shadow_unbilled=allocations["resourcing_status"].isin(["SHADOW", "UNBILLED"]))
        .groupby("project_id")["is_shadow_unbilled"]
        .mean()
        .rename("shadow_unbilled_share")
    )
    unbilled_value = (
        alloc_with_rate[alloc_with_rate["is_allocation_active"] == 1]
        .groupby("project_id")["unbilled_monthly_value"]
        .sum()
        .rename("monthly_unbilled_value_usd")
    )

    # ── Extension-related unbilled value: billable-status employees still
    # allocated PAST their project's official end date, counted in working
    # days (weekdays only) at the daily-charge basis (8h/day @ 100%
    # allocation). SHADOW/UNBILLED rows are excluded so this never overlaps
    # with unbilled_monthly_value_usd above.
    alloc_with_project_end = alloc_with_rate.merge(
        active[["project_code", "project_end_date"]], left_on="project_id", right_on="project_code", how="inner"
    )
    is_extension_row = (
        (alloc_with_project_end["is_allocation_active"] == 1)
        & (alloc_with_project_end["allocated_end_date"] > alloc_with_project_end["project_end_date"])
        & (~alloc_with_project_end["resourcing_status"].isin(["SHADOW", "UNBILLED"]))
    )
    _today = pd.Timestamp.now().normalize()
    def _accrued_working_days(r):
        if pd.isna(r["project_end_date"]) or pd.isna(r["allocated_end_date"]) or r["allocated_end_date"] <= r["project_end_date"]:
            return 0
        # Nothing has accrued until the project has actually passed its end date
        accrual_end = min(_today, r["allocated_end_date"])
        if accrual_end <= r["project_end_date"]:
            return 0
        return count_working_days(r["project_end_date"] + pd.Timedelta(days=1), accrual_end)
    alloc_with_project_end["overrun_working_days"] = alloc_with_project_end.apply(_accrued_working_days, axis=1)
    alloc_with_project_end["extension_unbilled_value"] = (
        is_extension_row
        * alloc_with_project_end["overrun_working_days"]
        * (alloc_with_project_end["allocation_by_percentage"] / 100)
        * alloc_with_project_end["hourly_rate"].fillna(0)
        * EXTENSION_DAILY_HOURS
    )
    extension_unbilled_value = (
        alloc_with_project_end.groupby("project_id")["extension_unbilled_value"]
        .sum()
        .rename("extension_unbilled_value_usd")
    )

    # Team's blended $/working-day if the CURRENT active billable team keeps
    # working past the end date. Used to translate a predicted extension
    # (in working days, from DevOps capacity) into a predicted $ loss.
    is_billable_active_row = (
        (alloc_with_project_end["is_allocation_active"] == 1)
        & (~alloc_with_project_end["resourcing_status"].isin(["SHADOW", "UNBILLED"]))
    )
    alloc_with_project_end["daily_extension_cost"] = (
        is_billable_active_row
        * (alloc_with_project_end["allocation_by_percentage"] / 100)
        * alloc_with_project_end["hourly_rate"].fillna(0)
        * EXTENSION_DAILY_HOURS
    )
    team_daily_extension_cost = (
        alloc_with_project_end.groupby("project_id")["daily_extension_cost"]
        .sum()
        .rename("team_daily_extension_cost_usd")
    )



    active = active.merge(n_employees, left_on="project_code", right_index=True, how="left")
    active = active.merge(max_alloc_end, left_on="project_code", right_index=True, how="left")
    active = active.merge(shadow_share, left_on="project_code", right_index=True, how="left")
    active = active.merge(unbilled_value, left_on="project_code", right_index=True, how="left")
    active = active.merge(extension_unbilled_value, left_on="project_code", right_index=True, how="left")
    active = active.merge(team_daily_extension_cost, left_on="project_code", right_index=True, how="left")
    active = active.merge(get_project_pulse_table(), left_on="project_code", right_index=True, how="left")
    active["is_pulse_risk"] = active["is_pulse_risk"].fillna(False)

    duration_days = (active["project_end_date"] - active["project_start_date"]).dt.days.clip(lower=1)
    active["churn_per_month"] = (active["n_employees"] / (duration_days / 30)).round(2)
    churn_p75 = active["churn_per_month"].quantile(0.75)

    active["overrun_days"] = (active["max_alloc_end"] - active["project_end_date"]).dt.days
    active["is_overrunning"] = active["overrun_days"] > OVERRUN_DAYS_THRESHOLD
    active["is_shadow_heavy"] = active["shadow_unbilled_share"] > SHADOW_SHARE_THRESHOLD
    active["is_high_churn"] = active["churn_per_month"] > churn_p75

    today = pd.Timestamp.now().normalize()
    active["days_to_ramp_down"] = (active["project_end_date"] - today).dt.days
    active["is_ramp_down_candidate"] = active["days_to_ramp_down"].between(0, RAMP_DOWN_WINDOW_DAYS)

    wsr_worst = wsr.copy()
    wsr_worst["worst_signal"] = worst_wsr_signal_vectorized(wsr_worst)
    wsr_real = wsr_worst[wsr_worst["worst_signal"] != "NO_COLOR"]
    wsr_summary = (
        wsr_real.groupby("project_id_masked")["worst_signal"]
        .agg(lambda s: max(s, key=lambda v: _RAG_SEVERITY[v]))
        .rename("wsr_worst_signal")
    )
    wsr_latest_summary = (
        wsr_real.sort_values("week_start_date").groupby("project_id_masked")["worst_signal"].last().rename("wsr_latest_signal")
    )
    active = active.merge(wsr_summary, left_on="project_code", right_index=True, how="left")
    active = active.merge(wsr_latest_summary, left_on="project_code", right_index=True, how="left")
    active["wsr_data_available"] = active["wsr_worst_signal"].notna()

    wsr_trend_by_project = wsr_trend(wsr)
    effort_spikes_by_project = get_project_effort_spikes()
    overtime_risk_by_employee = get_employee_overtime_risk()

    currently_allocated = allocations[allocations["is_allocation_active"] == 1]
    is_employee_overtime = currently_allocated["employee_id"].map(
        lambda emp_id: overtime_risk_by_employee.get(emp_id, {}).get("is_sustained_overtime", False)
    )
    overtime_employee_count = (
        currently_allocated[is_employee_overtime]
        .groupby("project_id")["employee_id"]
        .nunique()
        .rename("overtime_employee_count")
    )
    active = active.merge(overtime_employee_count, left_on="project_code", right_index=True, how="left")
    active["overtime_employee_count"] = active["overtime_employee_count"].fillna(0).astype(int)

    # ── ← NEW: DevOps board data — fetch ONCE before the loop ──────────────
    # A single WIQL + batch-detail call retrieves all open (non-Done) tickets
    # for every project in Azure DevOps, grouped by project code from AreaPath.
    # This avoids one network call per project and keeps the health report fast.
    _devops_enabled = bool(os.getenv("AZURE_DEVOPS_PAT"))
    if _devops_enabled:
        # _all_open_devops_tickets = fetch_open_devops_tickets()
        _all_open_devops_tickets = fetch_open_devops_tickets_cached()
        _devops_tickets_by_project = group_tickets_by_project_code(_all_open_devops_tickets)
    else:
        _devops_tickets_by_project = {}
    # ── END NEW ─────────────────────────────────────────────────────────────

    _demo_devops_tickets = _devops_tickets_by_project.get(DEMO_STATIC_DEVOPS_PROJECT_CODE, [])
    _demo_devops_ticket_stats = (
        compute_devops_ticket_stats(_demo_devops_tickets) if _devops_enabled else None
    )

    records = []
    for _, row in active.iterrows():
        expected = get_role_mix(row["type_of_project"], row["tech_coe"], templates=role_mix_templates)
        expected_headcount = expected.get("expected_headcount_common")
        actual_headcount = row["n_employees"] if pd.notna(row["n_employees"]) else 0
        is_understaffed = bool(
            expected_headcount and expected_headcount > 0 and actual_headcount < expected_headcount * UNDERSTAFFED_RATIO_THRESHOLD
        )

        project_code = row["project_code"]
        spike = effort_spikes_by_project.get(project_code, {})
        is_effort_spike = bool(spike.get("is_effort_spike", False))
        project_wsr = wsr_trend_by_project.get(project_code, {})
        project_wsr_trend = project_wsr.get("trend")
        is_wsr_critical = bool(project_wsr.get("is_critical"))
        is_wsr_long_term_decline = bool(project_wsr.get("is_long_term_decline"))
        overtime_employee_count = int(row["overtime_employee_count"])

        # ── ← NEW: per-project DevOps extension-risk metrics ───────────────
        # devops_tickets = _devops_tickets_by_project.get(project_code, [])
        # devops_risk = (
        #     compute_devops_extension_risk(devops_tickets, row["project_end_date"])
        #     if _devops_enabled
        #     else no_devops_config_risk()
        # )
        # is_devops_extension_risk = bool(devops_risk["has_devops_extension_risk"])

        # DEMO: force every project to read the same busy DevOps project's
        # tickets instead of its own (see DEMO_STATIC_DEVOPS_PROJECT_CODE above).
        devops_tickets = _demo_devops_tickets

        devops_risk = (
             compute_devops_extension_risk(devops_tickets, row["project_end_date"], project_code, ticket_stats=_demo_devops_ticket_stats)
             if _devops_enabled
             else no_devops_config_risk()
         )
        is_devops_extension_risk = bool(devops_risk["has_devops_extension_risk"])
        # ── END NEW ─────────────────────────────────────────────────────────

        # ── ← NEW: forward-looking extension revenue prediction ────────────
        projected_extension_days, projected_extension_confidence = _projected_additional_extension_days(devops_risk)
        team_daily_extension_cost_usd = (
            row["team_daily_extension_cost_usd"] if pd.notna(row.get("team_daily_extension_cost_usd")) else 0.0
        )
        predicted_extension_revenue_loss_usd = (
            round(projected_extension_days * team_daily_extension_cost_usd, 0) if projected_extension_days else 0
        )

        # Anchor date the projected extra days count FROM: if already overdue,
        # the extra work continues from today; if still within the risk
        # window (not yet overdue), the shortfall spills over starting right
        # after the project's own end date.
        _extension_anchor = today if devops_risk.get("is_overdue") else row["project_end_date"]
        predicted_extension_start_date = _date_str(_extension_anchor) if projected_extension_days else None
        predicted_extension_end_date = (
            _date_str(add_working_days(_extension_anchor, projected_extension_days))
            if projected_extension_days else None
        )
        projected_extension_duration_label = extension_duration_label(projected_extension_days)

        root_causes = []
        if row["is_overrunning"]:
            root_causes.append("overrunning")
        if row["is_shadow_heavy"]:
            root_causes.append("shadow_heavy")
        if row["is_high_churn"]:
            root_causes.append("high_churn")
        if is_understaffed:
            root_causes.append("understaffed")
        if overtime_employee_count > 0:
            root_causes.append("overtime_risk")
        if is_effort_spike:
            root_causes.append("effort_spike")
        if project_wsr_trend == "deteriorating":
            root_causes.append("wsr_deteriorating")
        if is_wsr_critical:
            root_causes.append("wsr_critical")
        if is_wsr_long_term_decline:
            root_causes.append("wsr_long_term_decline")
        if is_devops_extension_risk:                                          # ← NEW
            root_causes.append("devops_extension_risk")                       # ← NEW
        if bool(row.get("is_pulse_risk")):
            root_causes.append("pulse_risk")

        risk_score = len(root_causes)
        risk_band = "high" if risk_score >= 3 else ("medium" if risk_score == 2 else "low")

        records.append(
            {
                "project_code": project_code,
                "client_id": row.get("client_id"),
                "type_of_project": row["type_of_project"],
                "tech_coe": row["tech_coe"],
                "coe": canonical_project_coe(row["tech_coe"]),
                "n_employees": int(actual_headcount),
                "expected_headcount": round(expected_headcount, 1) if expected_headcount else None,
                "is_understaffed": is_understaffed,
                "overrun_days": int(row["overrun_days"]) if pd.notna(row["overrun_days"]) else None,
                "shadow_unbilled_share": round(row["shadow_unbilled_share"], 2) if pd.notna(row["shadow_unbilled_share"]) else None,
                "monthly_unbilled_value_usd": round(row["monthly_unbilled_value_usd"], 0) if pd.notna(row.get("monthly_unbilled_value_usd")) else 0,
                "extension_unbilled_value_usd": round(row["extension_unbilled_value_usd"], 0) if pd.notna(row.get("extension_unbilled_value_usd")) else 0,
                "team_daily_extension_cost_usd": round(team_daily_extension_cost_usd, 2),
                "projected_extension_days": projected_extension_days,
                "projected_extension_weeks": round(projected_extension_days / 5, 1) if projected_extension_days else None,
                "projected_extension_confidence": projected_extension_confidence,
                "predicted_extension_start_date": predicted_extension_start_date,
                "predicted_extension_end_date": predicted_extension_end_date,
                "projected_extension_duration_label": projected_extension_duration_label,
                "predicted_extension_revenue_loss_usd": predicted_extension_revenue_loss_usd,
                "churn_per_month": row["churn_per_month"] if pd.notna(row["churn_per_month"]) else None,
                "overtime_employee_count": overtime_employee_count,
                "is_effort_spike": is_effort_spike,
                "wsr_trend": project_wsr_trend,
                "is_wsr_critical": is_wsr_critical,
                "is_wsr_long_term_decline": is_wsr_long_term_decline,
                "wsr_recent_avg_severity": project_wsr.get("recent_avg_severity"),
                "wsr_baseline_avg_severity": project_wsr.get("baseline_avg_severity"),
                "risk_score": risk_score,
                "risk_band": risk_band,
                "root_causes": root_causes,
                "root_cause_categories": categorize_root_causes(root_causes),
                "is_extension_risk": any(ROOT_CAUSE_CATEGORY.get(c) == "extension" for c in root_causes),
                "is_escalation_risk": any(ROOT_CAUSE_CATEGORY.get(c) == "escalation" for c in root_causes),
                "is_pulse_risk": bool(row.get("is_pulse_risk")),
                "pulse_avg_score": round(row["pulse_avg_score"], 2) if pd.notna(row.get("pulse_avg_score")) else None,
                "pulse_response_count": int(row["pulse_response_count"]) if pd.notna(row.get("pulse_response_count")) else 0,
                "is_ramp_down_candidate": bool(row["is_ramp_down_candidate"]),
                "days_to_ramp_down": int(row["days_to_ramp_down"]) if pd.notna(row["days_to_ramp_down"]) else None,
                "wsr_data_available": bool(row["wsr_data_available"]),
                "wsr_worst_signal": row.get("wsr_worst_signal") if pd.notna(row.get("wsr_worst_signal")) else None,
                "wsr_latest_signal": row.get("wsr_latest_signal") if pd.notna(row.get("wsr_latest_signal")) else None,
                # ── ← NEW: DevOps board extension-risk fields ──────────────
                "devops_data_available":           devops_risk["devops_data_available"],
                "devops_extension_risk":           is_devops_extension_risk,
                "devops_open_tickets":             devops_risk["open_ticket_count"],
                "devops_blocked_tickets":          devops_risk["blocked_ticket_count"],
                "devops_in_progress_tickets":      devops_risk["in_progress_ticket_count"],
                "devops_tickets_past_project_end": devops_risk["tickets_due_past_project_end"],
                "devops_remaining_effort_hours":   devops_risk["remaining_effort_hours"],
                "devops_completed_work_hours":     devops_risk["completed_work_hours"],
                "devops_original_estimate_hours":  devops_risk["original_estimate_hours"],
                "devops_effort_completion_pct":    devops_risk["effort_completion_pct"],
                 "devops_to_do_tickets":             devops_risk["to_do_ticket_count"],
                "devops_within_risk_window":        devops_risk["within_risk_window"],
                "devops_working_days_in_window":    devops_risk["working_days_in_window"],
                "devops_team_capacity_hours":       devops_risk["team_capacity_hours"],
                "devops_team_capacity_hours_after_leave": devops_risk["team_capacity_hours_after_leave"],
                "devops_team_daily_capacity_hours": devops_risk.get("team_daily_capacity_hours", 0.0),
                "devops_capacity_surplus_hours":    devops_risk["capacity_surplus_hours"],
                "devops_days_to_clear_backlog":     devops_risk.get("days_to_clear_backlog"),
                "devops_is_overdue":                devops_risk["is_overdue"],
                "devops_tickets_missing_remaining_estimate": devops_risk["tickets_missing_remaining_estimate"],
                "devops_tickets_with_no_effort_data": devops_risk["tickets_with_no_effort_data"],
                # ── END NEW ────────────────────────────────────────────────
            }
        )

    return sorted(records, key=lambda r: r["risk_score"], reverse=True)


# ============================================================
# Everything below this line is UNCHANGED from the original.
# ============================================================

def get_all_project_sentiments(records: list[dict]) -> dict[str, dict]:
    """Return BERT sentiment for every project that has WSR data with comments."""
    wsr = get_adapter().get_wsr_reports()
    result: dict[str, dict] = {}
    for rec in records:
        code = rec["project_code"]
        if rec.get("wsr_data_available"):
            result[code] = _latest_sentiment(wsr, code)
        else:
            result[code] = {"has_data": False, "label": None, "compound": None, "risk_signal": "none", "latest_comment": None}
    return result


def _latest_sentiment(wsr: pd.DataFrame, project_code: str) -> dict:
    proj_wsr = wsr[wsr["project_id_masked"] == project_code].sort_values("week_start_date")
    if "comment" not in proj_wsr.columns:
        return {"has_data": False, "label": None, "compound": None, "risk_signal": "none", "latest_comment": None}
    comments = proj_wsr[proj_wsr["comment"].notna() & (proj_wsr["comment"].str.strip() != "")]
    if comments.empty:
        return {"has_data": False, "label": None, "compound": None, "risk_signal": "none", "latest_comment": None}
    latest = comments.iloc[-1]
    result = analyze_comment(str(latest["comment"]))
    return {
        "has_data": True,
        "label": result["label"],
        "compound": result["compound"],
        "risk_signal": result["risk_signal"],
        "latest_comment": str(latest["comment"])[:300],
    }


def get_project_wsr_sentiment(project_code: str, last_n: int = 8) -> dict:
    wsr = get_adapter().get_wsr_reports()
    proj_wsr = wsr[wsr["project_id_masked"] == project_code].sort_values("week_start_date")
    if "comment" not in proj_wsr.columns:
        return summarize_project_sentiment([])
    has_comments = proj_wsr[proj_wsr["comment"].notna() & (proj_wsr["comment"].str.strip() != "")]
    recent = has_comments.tail(last_n)
    entries = [
        {"date": str(row["week_start_date"])[:10], "comment": str(row["comment"])}
        for _, row in recent.iterrows()
    ]
    return summarize_project_sentiment(entries)


def get_validation_summary(records: list[dict]) -> dict:
    with_wsr = [r for r in records if r["wsr_data_available"]]
    agree = sum(1 for r in with_wsr if (r["risk_band"] != "low") == (r["wsr_worst_signal"] in ("RED", "AMBER")))
    return {
        "projects_with_real_wsr": len(with_wsr),
        "projects_total": len(records),
        "derived_risk_agrees_with_wsr_pct": round(100 * agree / len(with_wsr), 1) if with_wsr else None,
    }