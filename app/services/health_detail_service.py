import pandas as pd
import os
from app.core.adapter import get_adapter
from app.engines import embedding_engine, experience_engine, scoring
from app.engines.coe_skill_engine import GENERIC_SKILL_COES, derive_skills_for_coes
from app.engines.role_mix_engine import canonical_project_coe, get_role_mix
from app.services.free_pool_service import get_free_pool
from app.services.recommendation_service import (
    COST_TIE_BAND_PCT,
    _COE_AFFINITY_NEUTRAL,
    _coe_affinity_rank,
)
from app.services.health_monitor_service import (
     OVERRUN_DAYS_THRESHOLD,
     DEMO_STATIC_DEVOPS_PROJECT_CODE,
     EXTENSION_DAILY_HOURS,         
     count_working_days,
     add_working_days,
     extension_duration_label,
     SHADOW_SHARE_THRESHOLD,
     STANDARD_MONTHLY_HOURS,
     UNDERSTAFFED_RATIO_THRESHOLD,
     WSR_CRITICAL_MIN_REPORTS,
     WSR_CRITICAL_SEVERITY_THRESHOLD,
     WSR_LONG_TERM_MIN_REPORTS,
     WSR_TREND_LOOKBACK_REPORTS,
     WSR_TREND_RECENT_REPORTS,
     churn_p75_threshold,
     get_health_report,
     trend_from_severity_series,
     worst_wsr_signal_vectorized,
     wsr_severity_rows,
)
from app.services.devops_insights_service import (
    EXTENSION_RISK_WINDOW_DAYS,
    compute_sprint_breakdown,
    fetch_open_devops_tickets,
    fetch_open_devops_tickets_cached,
    group_tickets_by_project_code,
    list_devops_tickets_for_display,
)
from app.services.project_roster_service import get_project_roster
from app.services.rate_card_service import get_hourly_rate
from app.services.timesheet_insights_service import (
    EFFORT_SPIKE_MIN_BASELINE_WEEKS,
    EFFORT_SPIKE_RATIO_THRESHOLD,
    OVERTIME_DAILY_HOURS_THRESHOLD,
    SUSTAINED_OVERTIME_MIN_DAYS,
    SUSTAINED_OVERTIME_WINDOW_DAYS,
    get_employee_overtime_risk,
    get_employee_recent_daily_hours,
    get_project_weekly_hours,
)

class ProjectNotFound(Exception):

    def __init__(self, project_code: str):
        self.project_code = project_code
        super().__init__(f"project_code {project_code!r} not found or not an active project")

def _date_str(value) -> str | None:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else None

def get_project_health_detail(project_code: str) -> dict:
    summary = next((r for r in get_health_report() if r["project_code"] == project_code), None)
    if summary is None:
        raise ProjectNotFound(project_code)
    root_causes = summary["root_causes"]

    adapter = get_adapter()
    projects = adapter.get_projects()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()
    wsr = adapter.get_wsr_reports()

    project_row = projects[projects["project_code"] == project_code].iloc[0]
    proj_allocs = allocations[allocations["project_id"] == project_code].merge(
        employees[["employee_id", "job_name"]], on="employee_id", how="left"
    )
    active_allocs = proj_allocs[proj_allocs["is_allocation_active"] == 1]
    roster = get_project_roster(project_code)["roster"]

    project_end = project_row["project_end_date"]
    overrun_rows = (
        proj_allocs[proj_allocs["allocated_end_date"] > project_end]
        if pd.notna(project_end)
        else proj_allocs.iloc[0:0]
    )
    overrun_proof = {
        "fired": "overrunning" in root_causes,
        "threshold_days": OVERRUN_DAYS_THRESHOLD,
        "overrun_days": summary["overrun_days"],
        "project_end_date": _date_str(project_end),
        "qualifying_allocations": [
            {
                "employee_id": r["employee_id"],
                "job_name": r.get("job_name") if pd.notna(r.get("job_name")) else None,
                "resourcing_status": r["resourcing_status"],
                "allocated_end_date": _date_str(r["allocated_end_date"]),
                "days_past_project_end": int((r["allocated_end_date"] - project_end).days),
                "is_allocation_active": bool(r["is_allocation_active"]),
            }
            for _, r in overrun_rows.sort_values("allocated_end_date", ascending=False).iterrows()
        ],
    }

    # Active allocations only -- monthly_unbilled_value_usd is a CURRENT, ongoing cost
    # figure, so its proof rows must match (a historical/ended shadow allocation isn't
    # costing anything right now, even though the row still exists in the data).
    shadow_rows = active_allocs[active_allocs["resourcing_status"].isin(["SHADOW", "UNBILLED"])].copy()
    if shadow_rows.empty:
        shadow_rows["hourly_rate_usd"] = pd.Series(dtype=float)
        shadow_rows["monthly_unbilled_value_usd"] = pd.Series(dtype=float)
    else:
        shadow_rows["hourly_rate_usd"] = shadow_rows["job_name"].apply(get_hourly_rate)
        shadow_rows["monthly_unbilled_value_usd"] = (
            (shadow_rows["allocation_by_percentage"] / 100) * shadow_rows["hourly_rate_usd"].fillna(0) * STANDARD_MONTHLY_HOURS
        ).round(0)
    shadow_proof = {
        "fired": "shadow_heavy" in root_causes,
        "threshold_share": SHADOW_SHARE_THRESHOLD,
        "shadow_unbilled_share": summary["shadow_unbilled_share"],
        "monthly_unbilled_value_usd": summary["monthly_unbilled_value_usd"],
        "total_allocation_rows": int(len(proj_allocs)),
        "shadow_allocation_rows": int(len(shadow_rows)),
        "qualifying_allocations": [
            {
                "employee_id": r["employee_id"],
                "job_name": r.get("job_name") if pd.notna(r.get("job_name")) else None,
                "resourcing_status": r["resourcing_status"],
                "allocation_by_percentage": float(r["allocation_by_percentage"]),
                "hourly_rate_usd": float(r["hourly_rate_usd"]) if pd.notna(r["hourly_rate_usd"]) else None,
                "monthly_unbilled_value_usd": float(r["monthly_unbilled_value_usd"]),
                "allocated_start_date": _date_str(r["allocated_start_date"]),
                "allocated_end_date": _date_str(r["allocated_end_date"]),
            }
            for _, r in shadow_rows.sort_values(["employee_id", "allocated_start_date"]).iterrows()
        ],
    }

    extension_rows = active_allocs[
        (active_allocs["allocated_end_date"] > project_end)
        & (~active_allocs["resourcing_status"].isin(["SHADOW", "UNBILLED"]))
    ].copy() if pd.notna(project_end) else active_allocs.iloc[0:0].copy()

    if extension_rows.empty:
        extension_rows["hourly_rate_usd"] = pd.Series(dtype=float)
        extension_rows["overrun_working_days"] = pd.Series(dtype=int)
        extension_rows["extension_unbilled_value_usd"] = pd.Series(dtype=float)
    else:
        extension_rows["hourly_rate_usd"] = extension_rows["job_name"].apply(get_hourly_rate)
        _today = pd.Timestamp.now().normalize()
        def _accrued_days(end):
            if pd.isna(project_end) or pd.isna(end) or end <= project_end:
                return 0
            accrual_end = min(_today, end)
            if accrual_end <= project_end:
                return 0
            return count_working_days(project_end + pd.Timedelta(days=1), accrual_end)
        extension_rows["overrun_working_days"] = extension_rows["allocated_end_date"].apply(_accrued_days)
        extension_rows["extension_unbilled_value_usd"] = (
            extension_rows["overrun_working_days"]
            * (extension_rows["allocation_by_percentage"] / 100)
            * extension_rows["hourly_rate_usd"].fillna(0)
            * EXTENSION_DAILY_HOURS
        ).round(0)

    proj_end_for_calc = project_end
    billable_active = active_allocs[~active_allocs["resourcing_status"].isin(["SHADOW", "UNBILLED"])].copy()
    if not billable_active.empty:
        billable_active["hourly_rate_usd"] = billable_active["job_name"].apply(get_hourly_rate)
        _proj_days = summary["projected_extension_days"] or 0
        billable_active["predicted_additional_usd"] = (
            (billable_active["allocation_by_percentage"] / 100) * billable_active["hourly_rate_usd"].fillna(0)
            * EXTENSION_DAILY_HOURS * _proj_days
        ).round(0)
    else:
        billable_active["hourly_rate_usd"] = pd.Series(dtype=float)
        billable_active["predicted_additional_usd"] = pd.Series(dtype=float)

    predicted_breakdown = [
        {
            "employee_id": r["employee_id"],
            "job_name": r.get("job_name") if pd.notna(r.get("job_name")) else None,
            "resourcing_status": r["resourcing_status"],
            "allocation_by_percentage": float(r["allocation_by_percentage"]),
            "hourly_rate_usd": float(r["hourly_rate_usd"]) if pd.notna(r["hourly_rate_usd"]) else None,
            "predicted_additional_usd": float(r["predicted_additional_usd"]),
        }
        for _, r in billable_active.sort_values("predicted_additional_usd", ascending=False).iterrows()
        if r["predicted_additional_usd"] > 0
    ]

    extension_revenue_proof = {
        "fired": "overrunning" in root_causes or "devops_extension_risk" in root_causes,
        "daily_hours_basis": EXTENSION_DAILY_HOURS,
        "extension_unbilled_value_usd": summary["extension_unbilled_value_usd"],
        "team_daily_extension_cost_usd": summary["team_daily_extension_cost_usd"],
        "projected_extension_days": summary["projected_extension_days"],
        "projected_extension_weeks": summary["projected_extension_weeks"],
        "projected_extension_confidence": summary["projected_extension_confidence"],
        "predicted_extension_revenue_loss_usd": summary["predicted_extension_revenue_loss_usd"],
        "predicted_extension_start_date": summary.get("predicted_extension_start_date"),
        "predicted_extension_end_date": summary.get("predicted_extension_end_date"),
        "projected_extension_duration_label": summary.get("projected_extension_duration_label"),
        "note": (
            "Accrued value is a fact: working days (Mon-Fri) already elapsed past this project's end date, "
            "at 8h/day x current allocation % x rate card, for anyone not already marked SHADOW/UNBILLED "
            "(to avoid double-counting with the shadow-work total). Predicted value is a forecast: the "
            "DevOps-capacity-based day estimate (same as the Overview tab's Extension outlook), applied to "
            "every currently-active billable team member's own allocation % and rate, projected forward from today."
        ),
        "predicted_breakdown": predicted_breakdown,
        "qualifying_allocations": [
            {
                "employee_id": r["employee_id"],
                "job_name": r.get("job_name") if pd.notna(r.get("job_name")) else None,
                "resourcing_status": r["resourcing_status"],
                "allocation_by_percentage": float(r["allocation_by_percentage"]),
                "hourly_rate_usd": float(r["hourly_rate_usd"]) if pd.notna(r["hourly_rate_usd"]) else None,
                "overrun_working_days": int(r["overrun_working_days"]),
                "extension_unbilled_value_usd": float(r["extension_unbilled_value_usd"]),
                "allocated_end_date": _date_str(r["allocated_end_date"]),
            }
            for _, r in extension_rows.sort_values("extension_unbilled_value_usd", ascending=False).iterrows()
        ],
    }

    churn_proof = {
        "fired": "high_churn" in root_causes,
        "churn_per_month": summary["churn_per_month"],
        "cohort_p75_threshold": churn_p75_threshold(),
        "distinct_employees": summary["n_employees"],
        "roster_timeline": roster,
    }

    role_mix = get_role_mix(project_row["type_of_project"], project_row.get("tech_coe"))
    headcount_all_time_by_role = {
        k: int(v)
        for k, v in proj_allocs.dropna(subset=["job_name"]).groupby("job_name")["employee_id"].nunique().to_dict().items()
    }
    headcount_active_now_by_role = {
        k: int(v)
        for k, v in active_allocs.dropna(subset=["job_name"]).groupby("job_name")["employee_id"].nunique().to_dict().items()
    }
    fte_active_now_by_role = {
        k: round(float(v) / 100, 2)
        for k, v in active_allocs.dropna(subset=["job_name"]).groupby("job_name")["allocation_by_percentage"].sum().to_dict().items()
    }
    understaffed_proof = {
        "fired": "understaffed" in root_causes,
        "ratio_threshold": UNDERSTAFFED_RATIO_THRESHOLD,
        "actual_headcount_all_time": summary["n_employees"],
        "expected_headcount": summary["expected_headcount"],
        "role_mix_source": role_mix["source"],
        "role_mix_sample_size": role_mix["sample_size"],
        "expected_roles": role_mix.get("roles", []),
        "expected_role_mix": role_mix["role_mix"],
        "actual_headcount_active_now_by_role": headcount_active_now_by_role,
        "actual_fte_active_now_by_role": fte_active_now_by_role,
        "headcount_all_time_by_role": headcount_all_time_by_role,
    }

    overtime_risk = get_employee_overtime_risk()
    overtime_employees = []
    for _, r in active_allocs.iterrows():
        risk = overtime_risk.get(r["employee_id"])
        if risk and risk["is_sustained_overtime"]:
            overtime_employees.append(
                {
                    "employee_id": r["employee_id"],
                    "job_name": r.get("job_name") if pd.notna(r.get("job_name")) else None,
                    "overtime_days_recent": risk["overtime_days_recent"],
                    "max_daily_hours_recent": risk["max_daily_hours_recent"],
                    "is_sustained_overtime": risk["is_sustained_overtime"],
                    "daily_hours": get_employee_recent_daily_hours(r["employee_id"]),
                }
            )
    overtime_proof = {
        "fired": "overtime_risk" in root_causes,
        "daily_threshold_hours": OVERTIME_DAILY_HOURS_THRESHOLD,
        "sustained_min_days": SUSTAINED_OVERTIME_MIN_DAYS,
        "window_days": SUSTAINED_OVERTIME_WINDOW_DAYS,
        "overtime_employee_count": summary["overtime_employee_count"],
        "employees": overtime_employees,
    }

    effort_spike_proof = {
        "fired": "effort_spike" in root_causes,
        "ratio_threshold": EFFORT_SPIKE_RATIO_THRESHOLD,
        "min_baseline_weeks": EFFORT_SPIKE_MIN_BASELINE_WEEKS,
        "weekly_hours": get_project_weekly_hours(project_code),
    }

    proj_wsr_all = wsr[wsr["project_id_masked"] == project_code].copy()
    proj_wsr_all["worst_signal"] = worst_wsr_signal_vectorized(proj_wsr_all)
    proj_wsr_all = proj_wsr_all.sort_values("week_start_date")
    proj_wsr_severity = wsr_severity_rows(wsr[wsr["project_id_masked"] == project_code])
    trend_detail = (
        trend_from_severity_series(proj_wsr_severity["severity"])
        if not proj_wsr_severity.empty
        else {
            "trend": None,
            "recent_avg_severity": None,
            "prior_avg_severity": None,
            "is_critical": False,
            "baseline_avg_severity": None,
            "is_long_term_decline": False,
        }
    )
    wsr_proof = {
        "fired_deteriorating": "wsr_deteriorating" in root_causes,
        "fired_critical": "wsr_critical" in root_causes,
        "fired_long_term_decline": "wsr_long_term_decline" in root_causes,
        "data_available": summary["wsr_data_available"],
        "worst_signal": summary["wsr_worst_signal"],
        "latest_signal": summary["wsr_latest_signal"],
        "trend": trend_detail["trend"],
        "is_critical": trend_detail["is_critical"],
        "is_long_term_decline": trend_detail["is_long_term_decline"],
        "recent_avg_severity": trend_detail["recent_avg_severity"],
        "prior_avg_severity": trend_detail["prior_avg_severity"],
        "baseline_avg_severity": trend_detail["baseline_avg_severity"],
        "critical_severity_threshold": WSR_CRITICAL_SEVERITY_THRESHOLD,
        "recent_n": WSR_TREND_RECENT_REPORTS,
        "min_reports_required": WSR_TREND_LOOKBACK_REPORTS,
        "critical_min_reports_required": WSR_CRITICAL_MIN_REPORTS,
        "long_term_min_reports_required": WSR_LONG_TERM_MIN_REPORTS,
        "reports": [
            {
                "week_start_date": _date_str(r["week_start_date"]),
                "week_end_date": _date_str(r["week_end_date"]),
                "scope_status": r["scope_status"],
                "schedule_status": r["schedule_status"],
                "quality_status": r["quality_status"],
                "csat_status": r["csat_status"],
                "team_status": r["team_status"],
                "worst_signal": r["worst_signal"],
                "comment": str(r["comment"]) if "comment" in r and pd.notna(r.get("comment")) else None,
            }
            for _, r in proj_wsr_all.iterrows()
        ],
    }
    _devops_enabled = bool(os.getenv("AZURE_DEVOPS_PAT"))
    if _devops_enabled:
        # _devops_tickets_by_project = group_tickets_by_project_code(fetch_open_devops_tickets())
        _devops_tickets_by_project = group_tickets_by_project_code(fetch_open_devops_tickets_cached())
        _devops_raw_tickets = _devops_tickets_by_project.get(DEMO_STATIC_DEVOPS_PROJECT_CODE, [])
    else:
        _devops_raw_tickets = []

    devops_proof = {
        "fired": "devops_extension_risk" in root_causes,
        "data_available": summary["devops_data_available"],
        "window_days": EXTENSION_RISK_WINDOW_DAYS,
        "open_ticket_count": summary["devops_open_tickets"],
        "blocked_ticket_count": summary["devops_blocked_tickets"],
        "in_progress_ticket_count": summary["devops_in_progress_tickets"],
        "to_do_ticket_count": summary["devops_to_do_tickets"],
        "tickets_due_past_project_end": summary["devops_tickets_past_project_end"],
        "remaining_effort_hours": summary["devops_remaining_effort_hours"],
        "completed_work_hours": summary["devops_completed_work_hours"],
        "original_estimate_hours": summary["devops_original_estimate_hours"],
        "effort_completion_pct": summary["devops_effort_completion_pct"],
        "within_risk_window": summary["devops_within_risk_window"],
        "working_days_in_window": summary["devops_working_days_in_window"],
        "team_capacity_hours": summary["devops_team_capacity_hours"],
        "team_capacity_hours_after_leave": summary["devops_team_capacity_hours_after_leave"],
        "team_daily_capacity_hours": summary.get("devops_team_daily_capacity_hours", 0.0),
        "capacity_surplus_hours": summary["devops_capacity_surplus_hours"],
        "is_overdue": summary["devops_is_overdue"],
        "tickets_missing_remaining_estimate": summary["devops_tickets_missing_remaining_estimate"],
        "tickets_with_no_effort_data": summary["devops_tickets_with_no_effort_data"],
        "sprint_breakdown": compute_sprint_breakdown(_devops_raw_tickets, project_end),
        "tickets": list_devops_tickets_for_display(_devops_raw_tickets, project_end),
        
    }

    return {
        "project_code": project_code,
        "client_id": summary["client_id"],
        "type_of_project": summary["type_of_project"],
        "tech_coe": summary["tech_coe"],
        "project_start_date": _date_str(project_row["project_start_date"]),
        "project_end_date": _date_str(project_end),
        "risk_score": summary["risk_score"],
        "risk_band": summary["risk_band"],
        "root_causes": root_causes,
        "root_cause_categories": summary.get("root_cause_categories", {}),
        "is_extension_risk": summary.get("is_extension_risk", False),
        "is_escalation_risk": summary.get("is_escalation_risk", False),
        "overrun": overrun_proof,
        "shadow_heavy": shadow_proof,
         "extension_revenue": extension_revenue_proof,
        "high_churn": churn_proof,
        "understaffed": understaffed_proof,
        "overtime_risk": overtime_proof,
        "effort_spike": effort_spike_proof,
        "wsr": wsr_proof,
        "devops": devops_proof,
        "extension_estimate": _estimate_extension(summary["overrun_days"], devops_proof, project_end),
        "allocations_roster": roster,
    }

TOP_N_RELIEF_REQUIRED_SKILLS = 8
MIN_ROSTER_FOR_RELIEF_SKILLS = 2
MAX_RELIEF_CANDIDATES_SHOWN = 30

def _estimate_extension(overrun_days: int | None, devops_risk: dict, project_end_date: pd.Timestamp) -> dict:
    """Best-effort estimate of how much longer the project may run, from two
    independent signals:
      1. committed_overrun_days -- allocations already booked past the
         official end date TODAY. A fact from current resourcing data, not a
         projection.
      2. projected_additional_days -- derived from DevOps ticket data using
         team_daily_capacity_hours (steady-state hours/weekday for this
         project's active team, independent of the risk window -- see
         _team_daily_capacity_hours in devops_insights_service.py). Two cases:
           - Overdue (is_overdue=True): the entire remaining_effort_hours is
             now past-due work; additional_days = remaining_hours / daily_rate.
           - Within the risk window but not yet overdue: only the SHORTFALL
             matters -- work that won't fit in the days left --
             additional_days = shortfall_hours / daily_rate, where
             shortfall_hours = max(0, remaining_hours - capacity_after_leave).
      This is a projection: it assumes ticket effort estimates are accurate,
      no new scope appears, and the team's current daily rate holds steady --
      none of which are guaranteed, hence the explicit confidence field."""
    committed_days = overrun_days if (overrun_days is not None and overrun_days > 0) else 0

    daily_rate = devops_risk.get("team_daily_capacity_hours") or 0.0
    remaining_hours = devops_risk.get("remaining_effort_hours") or 0.0
    is_overdue = devops_risk.get("is_overdue", False)
    within_window = devops_risk.get("within_risk_window", False)
    capacity_after_leave = devops_risk.get("team_capacity_hours_after_leave") or 0.0
    predicted_extension_start_date = None
    predicted_extension_end_date = None
    duration_label = None

    projected_days = None
    projected_basis = None
    projection_confidence = "none"

    if daily_rate > 0 and (is_overdue or within_window):
        if is_overdue:
            shortfall_hours = remaining_hours
            projected_basis = "all remaining ticket effort, since the project end date has already passed"
        else:
            shortfall_hours = max(0.0, remaining_hours - capacity_after_leave)
            projected_basis = (
                "remaining ticket effort that exceeds team capacity within the risk window"
                if shortfall_hours > 0 else None
            )

        if shortfall_hours > 0:
            projected_days = round(shortfall_hours / daily_rate, 1)
            open_count = devops_risk.get("open_ticket_count") or 0
            gap = (devops_risk.get("tickets_missing_remaining_estimate", 0)
                   + devops_risk.get("tickets_with_no_effort_data", 0))
            gap_ratio = gap / open_count if open_count else 1.0
            projection_confidence = "low" if gap_ratio > 0.3 else "medium"

            anchor = pd.Timestamp.now().normalize() if is_overdue else project_end_date
            predicted_extension_start_date = _date_str(anchor)
            predicted_extension_end_date = _date_str(add_working_days(anchor, projected_days))
            duration_label = extension_duration_label(projected_days)

    return {
        "committed_overrun_days": committed_days,
        "committed_overrun_source": "allocations already scheduled past project_end_date",
        "projected_additional_days": projected_days,
        "projected_additional_weeks": round(projected_days / 5, 1) if projected_days is not None else None,
        "projected_additional_days_confidence": projection_confidence,
        "projected_basis": projected_basis,
        "predicted_extension_start_date": predicted_extension_start_date,
        "predicted_extension_end_date": predicted_extension_end_date,
        "projected_extension_duration_label": duration_label,
        "note": (
            "committed_overrun_days is a fact from current resourcing data; "
            "projected_additional_days/weeks is an estimate based on DevOps "
            "ticket effort and the team's current daily capacity, and assumes "
            "no new scope and a stable daily rate -- treat it as a planning "
            "signal, not a guarantee."
        ),
    }

def get_relief_staffing_candidates(
    project_code: str,
    top_n: int = MAX_RELIEF_CANDIDATES_SHOWN,
    *,
    # All 5 ranking parameters are independently selectable, same defaults as
    # get_recommendations() -- skill/competency/availability on, category_match/
    # project_count off. See scoring.composite_score_v2 for renormalization.
    include_skill: bool = True,
    include_competency: bool = True,
    include_availability: bool = True,
    include_category_match: bool = False,
    include_project_count: bool = False,
    include_coe_affinity: bool = True,
    # Off by default -- only breaks ties among already-close composite scores,
    # same COST_TIE_BAND_PCT behavior as get_recommendations().
    include_cost_efficiency: bool = False,
    # NOTE: get_recommendations()'s include_below_capacity/near_capacity_tolerance_pct
    # are intentionally NOT ported here. Both exist there because a requested
    # allocation % is a real gate: "does this person meet the % the role needs".
    # Relief staffing has no such target -- the whole premise is "who has real
    # idle capacity" (fully_free/under_utilized) or a real dated free-up
    # (ending_soon), not "who meets X% of a requested role." Forcing a capacity
    # gate/tolerance concept onto that would be a fake knob with nothing real
    # to gate against.
) -> dict:
    """Who from the Free Pool could realistically be added to a project that's
    overtime-risk and/or understaffed -- the same composite (skill + competency +
    availability) scoring used everywhere else, with the required skillset derived
    the same way the Leave page derives one for backfill: from this project's own
    team's real observed skills, falling back to typical skills for the project's
    CoE when the roster is too thin to trust as a signature.

    Two tiers, both real and both scored the same way (skill + competency, same
    weights as everywhere else): people with REAL idle capacity right now
    (fully_free/under_utilized), and people who are still busy but have a real,
    dated end to that -- "ending_soon" -- shown separately with their actual free
    date so relief isn't limited to only who happens to be idle today."""
    summary = next((r for r in get_health_report() if r["project_code"] == project_code), None)
    if summary is None:
        raise ProjectNotFound(project_code)
    root_causes = summary["root_causes"]

    adapter = get_adapter()
    skills = adapter.get_skills()
    allocations = adapter.get_allocations()
    competencies = adapter.get_competencies()

    proj_allocs = allocations[allocations["project_id"] == project_code]
    roster_ids = proj_allocs[proj_allocs["is_allocation_active"] == 1]["employee_id"].unique()

    required_phrases: list[str] = []
    required_skill_source = "none"
    if len(roster_ids) >= MIN_ROSTER_FOR_RELIEF_SKILLS:
        required_phrases = scoring.top_skill_phrases_for_employees(
            skills[skills["employee_id"].isin(roster_ids)], GENERIC_SKILL_COES, TOP_N_RELIEF_REQUIRED_SKILLS
        )
        if required_phrases:
            required_skill_source = "project_roster"

    project_coe = canonical_project_coe(summary.get("tech_coe"))
    if not required_phrases and project_coe:
        coe_skills = derive_skills_for_coes([project_coe], TOP_N_RELIEF_REQUIRED_SKILLS)["combined"]
        required_phrases = [
            f"{s['skill']} - {s['subskill']}" if s.get("subskill") else s["skill"] for s in coe_skills
        ]
        if required_phrases:
            required_skill_source = "coe_typical"

    skill_index = scoring.build_employee_skill_index(skills)
    competency_index = scoring.build_employee_competency_index(competencies)
    today = pd.Timestamp.now().normalize()

    # Semantic embedding layer — same 65/35 blend used in get_recommendations()
    emp_embedding_index = embedding_engine.build_employee_embedding_index(skills)
    skillset_text = " | ".join(required_phrases) if required_phrases else ""
    job_vec = embedding_engine.embed_jobspec(skillset_text) if skillset_text else None
    semantic_scores: dict[str, float] = {}
    if emp_embedding_index is not None and job_vec is not None:
        semantic_scores = embedding_engine.batch_cosine_similarity(job_vec, emp_embedding_index)

    # Track-record / experience layer -- same engine used by get_recommendations().
    # No deal "Solution" field exists on this surface (unlike the pipeline-row
    # recommendations flow), so requested_solution is always None here, same
    # pattern as Leave backfill; match_experience() falls back to tech_coe-only
    # matching against the project's own CoE when a solution isn't supplied.
    experience_profiles = experience_engine.build_employee_experience_profiles()
    requested_tech_coes = [project_coe] if project_coe else []

    def score_one(c: dict, available_now: bool) -> dict:
        emp_id = c["employee_id"]
        word_result = scoring.score_skill_match(required_phrases, skill_index.get(emp_id, {}))
        sem_score = semantic_scores.get(emp_id)
        if sem_score is not None and required_phrases:
            blended = 0.65 * sem_score + 0.35 * word_result["score"]
            confidence = word_result["confidence"]
            if confidence == "no_match" and sem_score >= 0.35:
                confidence = "semantic_match"
            skill_result = {"score": round(min(blended, 1.0), 3), "matched": word_result["matched"], "missing": word_result["missing"], "confidence": confidence}
        else:
            skill_result = word_result
        competency_entry = competency_index.get(emp_id, {"score": scoring.DEFAULT_COMPETENCY_SCORE, "confidence": "imputed"})
        # An "ending_soon" person isn't free yet, so their current idle_capacity_pct
        # (which is 0 or near it today) isn't a real availability signal -- scoring
        # them at 0 here keeps the composite honest; available_from_date carries the
        # actual real-world timing separately instead of faking it into the score.
        availability_score = min((c.get("idle_capacity_pct") or 0.0) / 100.0, 1.0) if available_now else 0.0
        experience = experience_engine.match_experience(
            experience_profiles.get(emp_id), requested_solution=None, requested_tech_coes=requested_tech_coes
        )
        composite = scoring.composite_score_v2(
            skill_result["score"], competency_entry["score"], availability_score,
            experience["relevant_project_ratio"], experience["project_count_score"],
            include={
                "skill": include_skill,
                "competency": include_competency,
                "availability": include_availability,
                "category_match": include_category_match,
                "project_count": include_project_count,
            },
        )
        coe_affinity_rank = (
            _coe_affinity_rank(c.get("primary_coe"), [project_coe] if project_coe else None)
            if include_coe_affinity else _COE_AFFINITY_NEUTRAL
        )
        return {
            **c,
            "composite_score": composite,
            "skill_score": skill_result["score"],
            "matched_skills": skill_result["matched"],
            "missing_skills": skill_result["missing"],
            "skill_confidence": skill_result["confidence"],
            "competency_score": competency_entry["score"],
            "competency_confidence": competency_entry["confidence"],
            "skill_bucket": scoring.bucket(skill_result["score"], skill_result["confidence"]),
            "coe_matches_project": bool(project_coe) and c.get("primary_coe") == project_coe,
            "coe_affinity_rank": coe_affinity_rank,
            "total_projects": experience["total_projects"],
            "distinct_clients": experience["distinct_clients"],
            "relevant_project_count": experience["relevant_project_count"],
            "relevant_project_ratio": experience["relevant_project_ratio"],
            "experience_confidence": experience["experience_confidence"],
            "top_categories": experience["top_categories"],
            "project_count_score": experience["project_count_score"],
        }

    # Cost-efficiency tie-break -- same shape as get_recommendations()'s
    # _cost_key/_composite_for_sort: only breaks ties among candidates whose
    # composite scores already sit within COST_TIE_BAND_PCT of each other, never
    # lets a cheap poor match outrank a genuinely stronger one. hourly_rate_usd
    # is already present on every free-pool candidate (see free_pool_service.py).
    def _cost_key(c: dict) -> float:
        rate = c.get("hourly_rate_usd")
        return -(rate if rate is not None else float("inf"))

    def _composite_for_sort(c: dict) -> float:
        if not include_cost_efficiency:
            return c["composite_score"]
        return round(c["composite_score"] * 100 / COST_TIE_BAND_PCT)

    def _sort_key(c: dict) -> tuple:
        return (
            c["coe_affinity_rank"],
            _composite_for_sort(c),
            _cost_key(c) if include_cost_efficiency else 0,
            c["composite_score"],
        )

    free_pool = get_free_pool()
    now_pool = [c for c in free_pool if c["reason"] in ("fully_free", "under_utilized") and c["employee_id"] not in roster_ids]
    soon_pool = [c for c in free_pool if c["reason"] == "ending_soon" and c["employee_id"] not in roster_ids]

    candidates = sorted((score_one(c, True) for c in now_pool), key=_sort_key, reverse=True)

    available_soon = []
    for c in soon_pool:
        scored = score_one(c, False)
        days = c.get("days_to_end")
        scored["days_to_available"] = days
        scored["available_from_date"] = (today + pd.Timedelta(days=days)).strftime("%Y-%m-%d") if days is not None else None
        available_soon.append(scored)
    # Final tiebreaker (after coe_affinity/cost/composite, same as `candidates`):
    # soonest real free date wins among otherwise-equal candidates. Missing days
    # sort last, matching the previous fallback of 999.
    available_soon.sort(
        key=lambda c: (*_sort_key(c), -(c.get("days_to_available") if c.get("days_to_available") is not None else 999)),
        reverse=True,
    )

    return {
        "project_code": project_code,
        "overtime_fired": "overtime_risk" in root_causes,
        "understaffed_fired": "understaffed" in root_causes,
        "overtime_employee_count": summary.get("overtime_employee_count", 0),
        "project_coe": project_coe,
        "required_skills": required_phrases,
        "required_skill_source": required_skill_source,
        "candidate_pool_size": len(now_pool),
        "candidates": candidates[:top_n],
        "available_soon_candidates": available_soon[:top_n],
    }
