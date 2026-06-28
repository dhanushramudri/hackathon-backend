import pandas as pd

from app.core.adapter import get_adapter
from app.engines.role_mix_engine import build_role_mix_templates, canonical_project_coe, get_role_mix
from app.services.rate_card_service import get_hourly_rate
from app.services.timesheet_insights_service import get_employee_overtime_risk, get_project_effort_spikes

OVERRUN_DAYS_THRESHOLD = 14
SHADOW_SHARE_THRESHOLD = 0.3
RAMP_DOWN_WINDOW_DAYS = 30
UNDERSTAFFED_RATIO_THRESHOLD = 0.75
STANDARD_MONTHLY_HOURS = 160

WSR_TREND_RECENT_REPORTS = 3
WSR_TREND_BASELINE_REPORTS = 3
WSR_TREND_LOOKBACK_REPORTS = WSR_TREND_RECENT_REPORTS + WSR_TREND_BASELINE_REPORTS
WSR_TREND_SHIFT_THRESHOLD = 0.3

WSR_CRITICAL_SEVERITY_THRESHOLD = 1.0
WSR_CRITICAL_MIN_REPORTS = WSR_TREND_RECENT_REPORTS

WSR_BASELINE_REPORTS = 3
WSR_LONG_TERM_MIN_REPORTS = WSR_BASELINE_REPORTS + WSR_TREND_RECENT_REPORTS

_RAG_SEVERITY = {"RED": 2, "AMBER": 1, "GREEN": 0, "NO_COLOR": -1}
_SEVERITY_TO_STATUS = {2: "RED", 1: "AMBER", 0: "GREEN", -1: "NO_COLOR"}
_RAG_COLUMNS = ["scope_status", "schedule_status", "quality_status", "csat_status", "team_status"]

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
    # monthly_unbilled_value_usd is a CURRENT, ongoing monthly cost figure -- it must
    # only count allocations that are actually active today. Without this filter it
    # sums every SHADOW/UNBILLED allocation ever recorded for the project, including
    # ones that ended months ago, inflating the real figure by ~5-6x.
    unbilled_value = (
        alloc_with_rate[alloc_with_rate["is_allocation_active"] == 1]
        .groupby("project_id")["unbilled_monthly_value"]
        .sum()
        .rename("monthly_unbilled_value_usd")
    )

    active = active.merge(n_employees, left_on="project_code", right_index=True, how="left")
    active = active.merge(max_alloc_end, left_on="project_code", right_index=True, how="left")
    active = active.merge(shadow_share, left_on="project_code", right_index=True, how="left")
    active = active.merge(unbilled_value, left_on="project_code", right_index=True, how="left")

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

        risk_score = len(root_causes)
        risk_band = "high" if risk_score >= 2 else ("medium" if risk_score == 1 else "low")

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
                "is_ramp_down_candidate": bool(row["is_ramp_down_candidate"]),
                "days_to_ramp_down": int(row["days_to_ramp_down"]) if pd.notna(row["days_to_ramp_down"]) else None,
                "wsr_data_available": bool(row["wsr_data_available"]),
                "wsr_worst_signal": row.get("wsr_worst_signal") if pd.notna(row.get("wsr_worst_signal")) else None,
                "wsr_latest_signal": row.get("wsr_latest_signal") if pd.notna(row.get("wsr_latest_signal")) else None,
            }
        )

    return sorted(records, key=lambda r: r["risk_score"], reverse=True)

def get_validation_summary(records: list[dict]) -> dict:
    with_wsr = [r for r in records if r["wsr_data_available"]]
    agree = sum(1 for r in with_wsr if (r["risk_band"] != "low") == (r["wsr_worst_signal"] in ("RED", "AMBER")))
    return {
        "projects_with_real_wsr": len(with_wsr),
        "projects_total": len(records),
        "derived_risk_agrees_with_wsr_pct": round(100 * agree / len(with_wsr), 1) if with_wsr else None,
    }
