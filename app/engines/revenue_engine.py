"""Synthetic project revenue -- no project in the source data carries a real
revenue figure, so this derives one from data that *does* exist: each
project's real allocation cost (rate card hourly rate x hours x allocation %,
same cost basis rate_card_service already uses elsewhere), scaled by a
per-CoE bill-rate multiplier.

The multiplier encodes relative commercial strength across CoEs -- JMAN is
strongest/most proven in Data Engineering, so it gets the highest multiplier
and is the ceiling every other CoE is capped below, not an exact market
figure. This is illustrative, same as rate_card_service's "illustrative"
hourly rates -- the point is a consistent, defensible ranking across
projects/CoEs to drive the revenue-target forecast's reverse math, not a
claim of real historical billing.
"""
import pandas as pd

from app.core.adapter import get_adapter
from app.engines.role_mix_engine import CANONICAL_COE_MAP, canonical_project_coe
from app.services.rate_card_service import get_hourly_rate

STANDARD_MONTHLY_HOURS = 160

# Bill-rate multiplier applied on top of internal cost (rate x hours x alloc%)
# to derive a synthetic revenue figure. Data Engineering is the ceiling --
# every other CoE is intentionally capped at or below it, reflecting that
# JMAN's commercial strength/track record is deepest in DE. None/unknown CoE
# gets a middling default rather than the lowest, since it usually means a
# mixed-CoE or not-yet-classified project, not a weak one.
COE_BILL_MULTIPLIER: dict[str, float] = {
    "Data Engineering": 3.0,
    "AI & ML": 2.7,
    "Full Stack Engineering": 2.2,
    "BI & Reporting": 1.9,
    "TechOps & Automation": 1.6,
}
DEFAULT_BILL_MULTIPLIER = 2.0
MAX_BILL_MULTIPLIER = COE_BILL_MULTIPLIER["Data Engineering"]

assert all(m <= MAX_BILL_MULTIPLIER for m in COE_BILL_MULTIPLIER.values()), (
    "No CoE's bill multiplier may exceed Data Engineering's -- see module docstring."
)


def _allocation_hours(row: pd.Series) -> float:
    start, end = row.get("allocated_start_date"), row.get("allocated_end_date")
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0.0
    days = (end - start).days + 1
    return (days / 30.0) * STANDARD_MONTHLY_HOURS


def _project_cost_and_coe() -> pd.DataFrame:
    adapter = get_adapter()
    projects = adapter.get_projects()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()

    real = projects[
        (projects["date_source"].isin(["given", "derived_allocation"]))
        & (projects["project_status"].isin(["COMPLETE", "ACTIVE"]))
    ].copy()
    real["coe"] = real["tech_coe"].apply(canonical_project_coe)

    merged = (
        allocations.merge(real[["project_code", "coe", "type_of_project", "project_status"]], left_on="project_id", right_on="project_code")
        .merge(employees[["employee_id", "job_name"]], on="employee_id", how="left")
    )
    merged = merged.dropna(subset=["job_name"])
    merged["hourly_rate"] = merged["job_name"].apply(get_hourly_rate).fillna(0.0)
    merged["hours"] = merged.apply(_allocation_hours, axis=1)
    merged["cost_usd"] = merged["hours"] * (merged["allocation_by_percentage"] / 100.0) * merged["hourly_rate"]
    return merged


def get_project_revenue_map() -> dict[str, dict]:
    """project_code -> {revenue_usd, cost_usd, coe, type_of_project, project_status}.
    One row per real (COMPLETE/ACTIVE) project with at least one billable
    allocation. Revenue = cost x the project's CoE bill multiplier."""
    merged = _project_cost_and_coe()
    if merged.empty:
        return {}

    grouped = merged.groupby("project_code").agg(
        cost_usd=("cost_usd", "sum"),
        coe=("coe", "first"),
        type_of_project=("type_of_project", "first"),
        project_status=("project_status", "first"),
    )
    out: dict[str, dict] = {}
    for project_code, row in grouped.iterrows():
        # groupby/agg turns a Python None "coe" into a float NaN -- `is None`
        # doesn't catch that, so an unclassified project would otherwise slip
        # through as its own bogus "nan" CoE bucket downstream.
        coe = row["coe"] if pd.notna(row["coe"]) else None
        multiplier = COE_BILL_MULTIPLIER.get(coe, DEFAULT_BILL_MULTIPLIER)
        revenue = round(row["cost_usd"] * multiplier, 0)
        out[project_code] = {
            "revenue_usd": float(revenue),
            "cost_usd": round(float(row["cost_usd"]), 0),
            "coe": coe,
            "type_of_project": row["type_of_project"] if pd.notna(row["type_of_project"]) else None,
            "project_status": row["project_status"],
        }
    return out


def get_revenue_benchmarks_by_coe() -> dict[str, dict]:
    """canonical CoE -> {avg_revenue_per_project, avg_revenue_per_fte_month,
    sample_size}. This is the inversion table the revenue-target forecast
    reads backwards: target_revenue / avg_revenue_per_project[coe] ~= how
    many projects of that CoE are needed. avg_revenue_per_fte_month lets the
    same benchmark be cross-checked against role-mix FTE totals."""
    merged = _project_cost_and_coe()
    revenue_map = get_project_revenue_map()
    if merged.empty or not revenue_map:
        return {}

    fte_by_project = merged.groupby("project_code").apply(
        lambda g: (g["allocation_by_percentage"] / 100.0 * (g["hours"] / STANDARD_MONTHLY_HOURS)).sum(),
        include_groups=False,
    )

    by_coe: dict[str, list[str]] = {}
    for project_code, info in revenue_map.items():
        by_coe.setdefault(info["coe"], []).append(project_code)

    benchmarks: dict[str, dict] = {}
    for coe, project_codes in by_coe.items():
        if coe is None:
            continue
        revenues = [revenue_map[p]["revenue_usd"] for p in project_codes]
        fte_months = [float(fte_by_project.get(p, 0.0)) for p in project_codes]
        total_fte_months = sum(fte_months)
        benchmarks[coe] = {
            "avg_revenue_per_project": round(sum(revenues) / len(revenues), 0) if revenues else 0.0,
            "avg_revenue_per_fte_month": round(sum(revenues) / total_fte_months, 0) if total_fte_months > 0 else 0.0,
            "sample_size": len(project_codes),
        }
    return benchmarks
