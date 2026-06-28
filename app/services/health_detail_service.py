import pandas as pd

from app.core.adapter import get_adapter
from app.engines.role_mix_engine import get_role_mix
from app.services.health_monitor_service import (
    OVERRUN_DAYS_THRESHOLD,
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

    shadow_rows = proj_allocs[proj_allocs["resourcing_status"].isin(["SHADOW", "UNBILLED"])].copy()
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
            }
            for _, r in proj_wsr_all.iterrows()
        ],
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
        "overrun": overrun_proof,
        "shadow_heavy": shadow_proof,
        "high_churn": churn_proof,
        "understaffed": understaffed_proof,
        "overtime_risk": overtime_proof,
        "effort_spike": effort_spike_proof,
        "wsr": wsr_proof,
        "allocations_roster": roster,
    }
