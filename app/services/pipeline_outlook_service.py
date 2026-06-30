import pandas as pd

from app.core.adapter import get_adapter
from app.engines.resource_code_decoder import decode_resource_code, group_label
from app.engines.skillset_classifier import classify_skillset
from app.services.demand_forecast_service import MIN_AVAILABLE_PCT_TO_SURFACE, STANDARD_MONTHLY_HOURS
from app.services.rate_card_service import get_hourly_rate
from app.services.recommendation_service import INTERNAL_PROJECT_TYPE, availability_as_of

OUTLOOK_MONTHS = 6
MAX_HORIZON_MONTHS = 36
SUPPLY_ANOMALY_SHARE_THRESHOLD = 0.90
LATE_NOTICE_THRESHOLD_DAYS = 14

def _avg_rate(designations: list[str]) -> float | None:
    rates = [r for d in designations if (r := get_hourly_rate(d)) is not None]
    return sum(rates) / len(rates) if rates else None

def _row_value_usd(designations: list[str], requested_pct) -> float | None:
    rate = _avg_rate(designations)
    if rate is None:
        return None
    pct = pd.to_numeric(requested_pct, errors="coerce")
    pct = 100.0 if pd.isna(pct) else float(pct)
    return round(rate * STANDARD_MONTHLY_HOURS * (pct / 100.0), 2)

def _fmt_date(value) -> str | None:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else None

_GRANULARITY_FREQ = {"month": "M", "week": "W"}

def _period_label(period: pd.Period, granularity: str) -> str:
    return period.start_time.strftime("%Y-%m-%d") if granularity == "week" else str(period)

def _period_label_series(dates: pd.Series, granularity: str) -> pd.Series:
    periods = dates.dt.to_period(_GRANULARITY_FREQ[granularity])
    if granularity == "week":
        return periods.apply(lambda p: p.start_time.strftime("%Y-%m-%d") if pd.notna(p) else "NaT")
    return periods.astype(str)

def _compute_periods(start: pd.Timestamp, horizon_months: int, granularity: str) -> list[pd.Period]:
    end = (pd.Period(start, freq="M") + horizon_months - 1).end_time
    return list(pd.period_range(start=start, end=end, freq=_GRANULARITY_FREQ[granularity]))

def _enrich_pipeline(pipeline: pd.DataFrame, granularity: str = "month") -> pd.DataFrame:
    pipeline = pipeline.copy()
    pipeline["likely_start_date"] = pd.to_datetime(pipeline["likely_start_date"], errors="coerce")
    pipeline["request_received"] = pd.to_datetime(pipeline["request_received"], errors="coerce")
    pipeline["original_requested_start_date"] = pd.to_datetime(pipeline["original_requested_start_date"], errors="coerce")
    pipeline["month"] = _period_label_series(pipeline["likely_start_date"], granularity)
    pipeline["role_code"] = pipeline["resources_requested"].astype(str).str.strip()
    pipeline["designations"] = pipeline["role_code"].apply(decode_resource_code)
    pipeline["role_label"] = pipeline["role_code"].apply(group_label)
    pipeline["is_confirmed"] = pipeline["sow_signed"].fillna("No").astype(str).str.strip().str.lower() == "yes"
    pipeline["requested_pct_numeric"] = pd.to_numeric(pipeline["requested_pct"], errors="coerce").fillna(100.0)
    pipeline["value_usd"] = pipeline.apply(lambda r: _row_value_usd(r["designations"], r.get("requested_pct")), axis=1)
    pipeline["skill_areas"] = pipeline["skillset"].apply(classify_skillset)

    notice_days = (pipeline["likely_start_date"] - pipeline["request_received"]).dt.days
    has_notice = notice_days.notna()
    pipeline["notice_days"] = notice_days.where(has_notice, None)
    pipeline["is_late_notice"] = (notice_days < LATE_NOTICE_THRESHOLD_DAYS).where(has_notice, None)
    return pipeline

def _enrich_supply(allocations: pd.DataFrame, employees: pd.DataFrame, granularity: str = "month") -> pd.DataFrame:
    freed = allocations.merge(employees[["employee_id", "job_name", "department_name", "location"]], on="employee_id", how="left")
    freed = freed[freed["is_allocation_active"] == 1].copy()
    freed["end_month"] = _period_label_series(freed["allocated_end_date"], granularity)
    return freed

def _anomaly_date_for_month(freed_in_window: pd.DataFrame, month: str) -> tuple[object, str | None]:
    month_rows = freed_in_window[freed_in_window["end_month"] == month]
    if month_rows.empty:
        return None, None
    total = month_rows["employee_id"].nunique()
    date_counts = month_rows.groupby("allocated_end_date")["employee_id"].nunique()
    top_date, top_count = date_counts.idxmax(), int(date_counts.max())
    if total > 0 and (top_count / total) >= SUPPLY_ANOMALY_SHARE_THRESHOLD:
        note = (
            f"{round(100 * top_count / total)}% of this month's projected supply ({top_count} of {total}) shares the "
            f"single end-date {pd.Timestamp(top_date).strftime('%Y-%m-%d')} -- likely a default/placeholder value, not "
            f"genuinely staggered turnover. Treat this month's supply signal with caution."
        )
        return top_date, note
    return None, None

def _deal_dict(r: pd.Series) -> dict:
    return {
        "deal_id": int(r["deal_id"]) if pd.notna(r.get("deal_id")) else None,
        "client": r.get("client"),
        "cluster": int(r["cluster"]) if pd.notna(r.get("cluster")) else None,
        "client_priority": r.get("client_priority"),
        "em": r.get("em"),
        "solution": r.get("solution"),
        "status": r.get("status"),
        "priority": r.get("priority"),
        "role_code": r.get("role_code"),
        "role_label": r.get("role_label"),
        "resolved_designations": r.get("designations") or [],
        "requested_pct": r.get("requested_pct"),
        "skillset": r.get("skillset"),
        "skill_areas": r.get("skill_areas") or [],
        "request_received": _fmt_date(r.get("request_received")),
        "original_requested_start_date": _fmt_date(r.get("original_requested_start_date")),
        "likely_start_date": _fmt_date(r.get("likely_start_date")),
        "request_type": r.get("request_type"),
        "start_date_confirmed": r.get("start_date_confirmed"),
        "number_of_weeks": r.get("number_of_weeks") if pd.notna(r.get("number_of_weeks")) else None,
        "deal_stage_hubspot": r.get("deal_stage_hubspot"),
        "comments": r.get("comments"),
        "sow_signed": r.get("sow_signed"),
        "is_confirmed": bool(r.get("is_confirmed")),
        "notice_days": int(r["notice_days"]) if pd.notna(r.get("notice_days")) else None,
        "is_late_notice": bool(r["is_late_notice"]) if pd.notna(r.get("is_late_notice")) else None,
        "hourly_rate_usd": round(_avg_rate(r.get("designations") or []), 2) if _avg_rate(r.get("designations") or []) is not None else None,
        "value_usd": round(float(r["value_usd"]), 2) if pd.notna(r.get("value_usd")) else None,
    }

def _supply_dict(r: pd.Series, anomaly_date) -> dict:
    end_date = r.get("allocated_end_date")
    return {
        "employee_id": r["employee_id"],
        "job_name": r.get("job_name"),
        "department_name": r.get("department_name"),
        "location": r.get("location"),
        "project_id": r.get("project_id"),
        "resourcing_status": r.get("resourcing_status"),
        "allocation_by_percentage": r.get("allocation_by_percentage"),
        "allocated_start_date": _fmt_date(r.get("allocated_start_date")),
        "allocated_end_date": _fmt_date(end_date),
        "is_anomaly_cluster": bool(anomaly_date is not None and pd.notna(end_date) and end_date == anomaly_date),
    }

def _period_start_date(label: str, granularity: str) -> pd.Timestamp:
    return pd.to_datetime(label) if granularity == "week" else pd.Period(label, freq="M").start_time

def _designation_roster_as_of(
    target_designations: list[str],
    as_of_date: pd.Timestamp,
    employees: pd.DataFrame,
    allocations: pd.DataFrame,
    projects: pd.DataFrame,
    busy_pct_cache: dict[pd.Timestamp, pd.Series],
) -> list[dict]:
    active = employees[(employees["account_status"] == 1) & (employees["job_name"].isin(target_designations))]
    if active.empty:
        return []

    if as_of_date not in busy_pct_cache:
        busy_pct_cache[as_of_date] = availability_as_of(allocations, as_of_date)
    busy_pct = busy_pct_cache[as_of_date]

    covering = allocations[
        (allocations["is_allocation_active"] == 1)
        & (allocations["allocated_start_date"] <= as_of_date)
        & (allocations["allocated_end_date"] >= as_of_date)
    ].merge(projects[["project_code", "type_of_project"]], left_on="project_id", right_on="project_code", how="left")

    roster = []
    for _, emp in active.iterrows():
        emp_id = emp["employee_id"]
        busy = float(busy_pct.get(emp_id, 0.0))
        available_pct = max(0.0, round(100.0 - busy, 1))
        is_available = busy == 0 or available_pct >= MIN_AVAILABLE_PCT_TO_SURFACE
        current = covering[covering["employee_id"] == emp_id]
        roster.append(
            {
                "employee_id": emp_id,
                "job_name": emp.get("job_name"),
                "location": emp.get("location"),
                "department_name": emp.get("department_name"),
                "available_pct": available_pct,
                "is_available": bool(is_available),
                "current_allocations": [
                    {
                        "project_id": r["project_id"],
                        "type_of_project": r.get("type_of_project"),
                        "allocation_by_percentage": r["allocation_by_percentage"],
                        "allocated_start_date": _fmt_date(r.get("allocated_start_date")),
                        "allocated_end_date": _fmt_date(r.get("allocated_end_date")),
                        "is_internal": r.get("type_of_project") == INTERNAL_PROJECT_TYPE,
                    }
                    for _, r in current.iterrows()
                ],
            }
        )
    roster.sort(key=lambda r: (not r["is_available"], -r["available_pct"]))
    return roster

def _role_demand_rows(
    rows: pd.DataFrame, employees: pd.DataFrame, allocations: pd.DataFrame, projects: pd.DataFrame, granularity: str, is_confirmed: bool
) -> tuple[list[dict], str | None, list[dict]]:
    busy_pct_cache: dict[pd.Timestamp, pd.Series] = {}
    role_need = rows.groupby(["month", "role_label"]).agg(
        needed_headcount=("role_code", "size"),
        value_usd=("value_usd", "sum"),
        role_code=("role_code", "first"),
        avg_requested_pct=("requested_pct_numeric", "mean"),
    ).reset_index()

    out = []
    first_shortfall_month = None
    for _, row in role_need.sort_values(["month", "needed_headcount"], ascending=[True, False]).iterrows():
        designations = decode_resource_code(row["role_code"])
        available = None
        shortfall = None
        shortfall_value = 0.0
        if designations:
            as_of_date = _period_start_date(row["month"], granularity)
            roster = _designation_roster_as_of(designations, as_of_date, employees, allocations, projects, busy_pct_cache)
            available = sum(1 for r in roster if r["is_available"])
        if available is not None:
            shortfall = max(0, int(row["needed_headcount"]) - available)
            rate = _avg_rate(designations)
            avg_pct = float(row["avg_requested_pct"]) if pd.notna(row["avg_requested_pct"]) else 100.0
            shortfall_value = round(shortfall * rate * STANDARD_MONTHLY_HOURS * (avg_pct / 100), 2) if (shortfall and rate) else 0.0
        # Only CONFIRMED shortfalls drive the page's headline "first shortfall" alert --
        # unconfirmed/speculative requests may never materialize, so they still get a real
        # available/shortfall number for reference (computed identically either way), but
        # don't get to trigger the actionable warning banner.
        if is_confirmed and shortfall and (first_shortfall_month is None or row["month"] < first_shortfall_month):
            first_shortfall_month = row["month"]
        out.append(
            {
                "month": row["month"],
                "role": row["role_label"],
                "role_code": row["role_code"],
                "resolved_designations": designations,
                "needed_headcount": int(row["needed_headcount"]),
                "available_headcount": available,
                "shortfall": shortfall,
                "shortfall_value_usd": shortfall_value,
                "value_usd": round(float(row["value_usd"]), 2) if pd.notna(row["value_usd"]) else None,
                "is_confirmed": is_confirmed,
            }
        )
    first_shortfall_roles = [r for r in out if r["month"] == first_shortfall_month and r["shortfall"]] if first_shortfall_month else []
    return out, first_shortfall_month, first_shortfall_roles

def get_pipeline_outlook(
    start_date: str | None = None, horizon_months: int = OUTLOOK_MONTHS, granularity: str = "month"
) -> dict:
    horizon_months = max(1, min(horizon_months, MAX_HORIZON_MONTHS))
    granularity = granularity if granularity in _GRANULARITY_FREQ else "month"
    adapter = get_adapter()
    employees = adapter.get_employees()
    allocations = adapter.get_allocations()
    projects = adapter.get_projects()
    pipeline = _enrich_pipeline(adapter.get_pipeline_forecast(), granularity)
    freed = _enrich_supply(allocations, employees, granularity)

    real_max_demand_month = pipeline["month"][pipeline["likely_start_date"].notna()].max() if pipeline["likely_start_date"].notna().any() else None
    real_max_supply_month = freed["end_month"][freed["allocated_end_date"].notna()].max() if freed["allocated_end_date"].notna().any() else None

    start = pd.to_datetime(start_date) if start_date else (pd.Timestamp.now().normalize() + pd.Timedelta(days=1))
    months = [_period_label(p, granularity) for p in _compute_periods(start, horizon_months, granularity)]

    in_window = pipeline[pipeline["month"].isin(months)]
    freed_in_window = freed[freed["end_month"].isin(months)]

    supply_by_month = freed_in_window.groupby("end_month")["employee_id"].nunique()

    anomaly_by_month: dict[str, str] = {}
    for m in months:
        _, note = _anomaly_date_for_month(freed_in_window, m)
        if note:
            anomaly_by_month[m] = note

    demand_counts = in_window.groupby(["month", "is_confirmed"]).size()
    confirmed_value_by_month = in_window[in_window["is_confirmed"]].groupby("month")["value_usd"].sum()
    unconfirmed_value_by_month = in_window[~in_window["is_confirmed"]].groupby("month")["value_usd"].sum()

    confirmed_role_rows, first_shortfall_month, first_shortfall_roles = _role_demand_rows(
        in_window[in_window["is_confirmed"]], employees, allocations, projects, granularity, is_confirmed=True
    )
    unconfirmed_role_rows, _, _ = _role_demand_rows(
        in_window[~in_window["is_confirmed"]], employees, allocations, projects, granularity, is_confirmed=False
    )
    role_demand_by_month = confirmed_role_rows + unconfirmed_role_rows

    skill_rows = in_window[in_window["is_confirmed"]][["month", "skill_areas"]].explode("skill_areas")
    skill_area_demand_by_month = (
        skill_rows.dropna(subset=["skill_areas"])
        .groupby(["month", "skill_areas"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"skill_areas": "skill_area"})
        .sort_values(["month", "count"], ascending=[True, False])
        .to_dict(orient="records")
    )
    no_skill_area_count = int(skill_rows["skill_areas"].isna().sum())

    month_rows = []
    for m in months:
        confirmed_n = int(demand_counts.get((m, True), 0))
        unconfirmed_n = int(demand_counts.get((m, False), 0))
        supply_n = int(supply_by_month.get(m, 0))
        month_rows.append(
            {
                "month": m,
                "confirmed_demand_count": confirmed_n,
                "unconfirmed_demand_count": unconfirmed_n,
                "projected_supply_count": supply_n,
                "net_confirmed_surplus_shortfall": supply_n - confirmed_n,
                "early_warning": (supply_n - confirmed_n) < 0,
                "has_real_demand_data": bool(real_max_demand_month) and m <= real_max_demand_month,
                "has_real_supply_data": bool(real_max_supply_month) and m <= real_max_supply_month,
                "supply_anomaly_note": anomaly_by_month.get(m),
                "confirmed_value_usd": round(float(confirmed_value_by_month.get(m, 0.0) or 0.0), 2),
                "unconfirmed_value_usd": round(float(unconfirmed_value_by_month.get(m, 0.0) or 0.0), 2),
            }
        )

    cluster_mix = (
        in_window.groupby(["month", "cluster"]).size().rename("count").reset_index()
        .sort_values(["month", "cluster"]).to_dict(orient="records")
    )
    with_solution = in_window[in_window["solution"].notna()]
    solution_mix = (
        with_solution.groupby(["month", "solution"]).size().rename("count").reset_index()
        .sort_values(["month", "count"], ascending=[True, False]).to_dict(orient="records")
        if not with_solution.empty else []
    )

    cluster_scorecards = []
    for cluster_id, grp in in_window.groupby("cluster"):
        resolved_value = grp["value_usd"].sum()
        top_roles = grp["role_label"].value_counts().head(3)
        top_skills = pd.Series([s for row in grp["skill_areas"] for s in row]).value_counts().head(3) if grp["skill_areas"].apply(len).sum() else pd.Series(dtype=int)
        cluster_scorecards.append(
            {
                "cluster": int(cluster_id),
                "deal_count": int(len(grp)),
                "confirmed_count": int(grp["is_confirmed"].sum()),
                "unconfirmed_count": int((~grp["is_confirmed"]).sum()),
                "sow_signed_rate_pct": round(100 * grp["is_confirmed"].mean(), 1) if len(grp) else 0.0,
                "value_usd": round(float(resolved_value), 2) if pd.notna(resolved_value) else 0.0,
                "top_roles": [{"role": k, "count": int(v)} for k, v in top_roles.items()],
                "top_skill_areas": [{"skill_area": k, "count": int(v)} for k, v in top_skills.items()],
                "clients": sorted(grp["client"].dropna().unique().tolist()),
            }
        )
    cluster_scorecards.sort(key=lambda c: -c["deal_count"])

    return {
        "start_date": start.strftime("%Y-%m-%d"),
        "horizon_months": horizon_months,
        "granularity": granularity,
        "months": month_rows,
        "first_shortfall_month": first_shortfall_month,
        "first_shortfall_roles": first_shortfall_roles,
        "real_demand_data_through": real_max_demand_month,
        "real_supply_data_through": real_max_supply_month,
        "role_demand_by_month": role_demand_by_month,
        "skill_area_demand_by_month": skill_area_demand_by_month,
        "no_skill_area_specified_count": no_skill_area_count,
        "project_mix_by_cluster_by_month": cluster_mix,
        "project_mix_by_solution_by_month": solution_mix,
        "cluster_scorecards": cluster_scorecards,
        "assumption": (
            "resources_requested codes are decoded against real JMAN org knowledge: most resolve to "
            "one real designation; several are genuinely "
            "flexible in the source itself (e.g. 'SAC/AC' literally means either Senior Associate "
            "Consultant or Associate Consultant works) and count as covered if real spare capacity "
            "exists in any of the listed designations, with the dollar figure averaged across them. "
            "A few codes ('EM', 'GTM Architect', 'Sr DS SME') have no real designation at all and are "
            "excluded from dollar figures, though still counted in headcount. Dollar values use the "
            "illustrative Rate Card (no real cost data exists in any source file) weighted by each "
            "deal's own real requested %. Unconfirmed demand's $ value is shown as-is, not weighted by "
            "deal stage -- no historical win-rate data exists to calibrate a real probability against, "
            "so we deliberately don't manufacture one. Pipeline demand has zero real rows past "
            f"{real_max_demand_month or 'the available data'} -- months beyond that show real zeros, "
            "not an estimate. Every number on this page is click-through to the exact real rows behind "
            "it -- nothing here is a black box."
        ),
    }

def get_six_month_outlook() -> dict:
    return get_pipeline_outlook(start_date=None, horizon_months=OUTLOOK_MONTHS)

def get_pipeline_outlook_drilldown(
    dimension: str,
    value: str | None = None,
    month: str | None = None,
    start_date: str | None = None,
    horizon_months: int = OUTLOOK_MONTHS,
    granularity: str = "month",
    is_confirmed: bool = True,
) -> dict:
    adapter = get_adapter()
    granularity = granularity if granularity in _GRANULARITY_FREQ else "month"
    pipeline = _enrich_pipeline(adapter.get_pipeline_forecast(), granularity)

    deals: list[dict] = []
    if dimension == "confirmed_demand" and month:
        rows = pipeline[(pipeline["month"] == month) & pipeline["is_confirmed"]]
        deals = [_deal_dict(r) for _, r in rows.iterrows()]
    elif dimension == "unconfirmed_demand" and month:
        rows = pipeline[(pipeline["month"] == month) & ~pipeline["is_confirmed"]]
        deals = [_deal_dict(r) for _, r in rows.iterrows()]
    elif dimension == "role" and month:
        rows = pipeline[(pipeline["month"] == month) & (pipeline["is_confirmed"] == is_confirmed) & (pipeline["role_label"] == value)]
        deals = [_deal_dict(r) for _, r in rows.iterrows()]
    elif dimension == "skill_area" and month:
        rows = pipeline[(pipeline["month"] == month) & pipeline["is_confirmed"] & pipeline["skill_areas"].apply(lambda lst: value in lst)]
        deals = [_deal_dict(r) for _, r in rows.iterrows()]
    elif dimension == "solution" and month:
        rows = pipeline[(pipeline["month"] == month) & (pipeline["solution"] == value)]
        deals = [_deal_dict(r) for _, r in rows.iterrows()]
    elif dimension == "cluster":
        cluster_rows = pipeline[pipeline["cluster"].astype(str) == str(value)]
        if month:
            cluster_rows = cluster_rows[cluster_rows["month"] == month]
        else:
            start = pd.to_datetime(start_date) if start_date else (pd.Timestamp.now().normalize() + pd.Timedelta(days=1))
            window_months = [
                _period_label(p, granularity)
                for p in _compute_periods(start, max(1, min(horizon_months, MAX_HORIZON_MONTHS)), granularity)
            ]
            cluster_rows = cluster_rows[cluster_rows["month"].isin(window_months)]
        rows = cluster_rows
        deals = [_deal_dict(r) for _, r in rows.iterrows()]
    else:
        rows = pipeline.iloc[0:0]

    supply_employees: list[dict] = []
    supply_note: str | None = None
    designation_roster: list[dict] = []
    if dimension == "supply" and month:
        freed = _enrich_supply(adapter.get_allocations(), adapter.get_employees(), granularity)
        anomaly_date, supply_note = _anomaly_date_for_month(freed, month)
        month_freed = freed[freed["end_month"] == month].drop_duplicates("employee_id")
        supply_employees = [_supply_dict(r, anomaly_date) for _, r in month_freed.iterrows()]
    elif dimension == "role" and is_confirmed and len(rows) and month:
        target_designations = decode_resource_code(rows.iloc[0]["role_code"])
        if target_designations:
            as_of_date = _period_start_date(month, granularity)
            designation_roster = _designation_roster_as_of(
                target_designations, as_of_date, adapter.get_employees(), adapter.get_allocations(), adapter.get_projects(), {}
            )

    return {
        "month": month,
        "dimension": dimension,
        "value": value,
        "deals": deals,
        "supply_employees": supply_employees,
        "supply_anomaly_note": supply_note,
        "designation_roster": designation_roster,
    }
