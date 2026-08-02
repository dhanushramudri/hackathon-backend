from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.engines.headcount_prediction_engine import (
    get_headcount_prediction, get_raw_table, list_raw_tables, simulate_headcount_prediction,
)
from app.engines.revenue_engine import compute_duration_buckets, get_revenue_benchmarks_by_coe
from app.engines.role_mix_engine import get_role_mix_by_coes
from app.engines.simple_forecast_engine import get_prediction_forecast
from app.services.demand_forecast_service import (
    get_new_project_forecast, get_revenue_target_forecast, get_top_candidates_for_role,
)
from app.services.pipeline_outlook_service import OUTLOOK_MONTHS, get_pipeline_outlook, get_pipeline_outlook_drilldown

router = APIRouter(prefix="/forecast", tags=["forecast"])

class NewProjectSpec(BaseModel):
    coes: list[str] | None = None
    type_of_project: str | None = None
    category: str | None = None
    count: int = 1
    role_mix_overrides: dict[str, float] | None = None
    required_skills: list[str] | None = None
    start_date: str | None = None
    duration_weeks: int | None = None

class RoleMixPreviewRequest(BaseModel):
    coes: list[str]
    type_of_project: str | None = None

class RevenueTargetRequest(BaseModel):
    target_revenue_usd: float
    priority_coes: list[str] | None = None
    start_date: str | None = None
    duration_weeks: int | None = None
    type_of_project: str | None = None
    duration_mix: dict[str, float] | None = None
    dnd_win_rate_pct: float | None = None
    target_date: str | None = None
    include_skill: bool = True
    include_competency: bool = True
    include_availability: bool = True
    include_category_match: bool = False
    include_project_count: bool = False

@router.post("/new-projects")
def new_projects(
    specs: list[NewProjectSpec],
    # Same 5-parameter ranking flexibility as get_recommendations -- one
    # selection for the whole forecast run (see DEFAULT_FORECAST_INCLUDE in
    # demand_forecast_service.py for why this is call-level, not per-spec).
    include_skill: bool = Query(default=True),
    include_competency: bool = Query(default=True),
    include_availability: bool = Query(default=True),
    include_category_match: bool = Query(default=False),
    include_project_count: bool = Query(default=False),
) -> dict:
    return get_new_project_forecast(
        [s.model_dump() for s in specs],
        include={
            "skill": include_skill,
            "competency": include_competency,
            "availability": include_availability,
            "category_match": include_category_match,
            "project_count": include_project_count,
        },
    )

@router.post("/role-mix-preview")
def role_mix_preview(body: RoleMixPreviewRequest) -> dict:
    return get_role_mix_by_coes(body.coes, body.type_of_project)

@router.get("/revenue-benchmarks")
def revenue_benchmarks() -> dict:
    return get_revenue_benchmarks_by_coe()

@router.get("/top-candidates-for-role")
def top_candidates_for_role(designation: str, as_of_date: str, limit: int = Query(default=10, ge=1, le=50)) -> list[dict]:
    return get_top_candidates_for_role(designation, as_of_date, limit)

@router.post("/revenue-target")
def revenue_target(body: RevenueTargetRequest) -> dict:
    return get_revenue_target_forecast(
        body.target_revenue_usd,
        priority_coes=body.priority_coes,
        start_date=body.start_date,
        duration_weeks=body.duration_weeks,
        type_of_project=body.type_of_project,
        duration_mix=body.duration_mix,
        dnd_win_rate_pct=body.dnd_win_rate_pct,
        target_date=body.target_date,
        include={
            "skill": body.include_skill,
            "competency": body.include_competency,
            "availability": body.include_availability,
            "category_match": body.include_category_match,
            "project_count": body.include_project_count,
        },
    )

@router.get("/duration-mix-benchmarks")
def duration_mix_benchmarks() -> dict:
    return compute_duration_buckets()

@router.get("/prediction")
def prediction(horizon_months: int = 24) -> dict:
    return get_prediction_forecast(horizon_months)

@router.get("/headcount-prediction")
def headcount_prediction(horizon_months: int = 12) -> dict:
    return get_headcount_prediction(horizon_months)

@router.get("/headcount-prediction/tables")
def headcount_prediction_tables() -> list[dict]:
    return list_raw_tables()

@router.get("/headcount-prediction/raw-data")
def headcount_prediction_raw_data(table: str) -> dict:
    try:
        return get_raw_table(table)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

class HeadcountSimulateRequest(BaseModel):
    horizon_months: int = 12
    history: list[dict]

@router.post("/headcount-prediction/simulate")
def headcount_prediction_simulate(body: HeadcountSimulateRequest) -> dict:
    try:
        return simulate_headcount_prediction(body.history, body.horizon_months)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/six-month-outlook")
def six_month_outlook(start_date: str | None = None, horizon_months: int = OUTLOOK_MONTHS, granularity: str = "month") -> dict:
    return get_pipeline_outlook(start_date=start_date, horizon_months=horizon_months, granularity=granularity)

@router.get("/six-month-outlook/drilldown")
def six_month_outlook_drilldown(
    dimension: str,
    value: str | None = None,
    month: str | None = None,
    start_date: str | None = None,
    horizon_months: int = OUTLOOK_MONTHS,
    granularity: str = "month",
    is_confirmed: bool = True,
) -> dict:
    return get_pipeline_outlook_drilldown(
        dimension=dimension,
        value=value,
        month=month,
        start_date=start_date,
        horizon_months=horizon_months,
        granularity=granularity,
        is_confirmed=is_confirmed,
    )
