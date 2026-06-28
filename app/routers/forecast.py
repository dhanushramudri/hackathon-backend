from fastapi import APIRouter
from pydantic import BaseModel

from app.engines.role_mix_engine import get_role_mix_by_coes
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
def new_projects(specs: list[NewProjectSpec]) -> dict:
    return get_new_project_forecast([s.model_dump() for s in specs])

@router.post("/role-mix-preview")
def role_mix_preview(body: RoleMixPreviewRequest) -> dict:
    return get_role_mix_by_coes(body.coes, body.type_of_project)

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
