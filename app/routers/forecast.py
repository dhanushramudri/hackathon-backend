from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.engines.role_mix_engine import get_role_mix_by_coes
from app.engines.simple_forecast_engine import get_prediction_forecast
from app.services.demand_forecast_service import get_new_project_forecast
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

@router.get("/prediction")
def prediction(horizon_months: int = 24) -> dict:
    return get_prediction_forecast(horizon_months)

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
