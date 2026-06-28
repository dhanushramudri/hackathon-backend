from fastapi import APIRouter, HTTPException

from app.services.health_detail_service import ProjectNotFound, get_project_health_detail
from app.services.health_monitor_service import get_health_report, get_validation_summary
from app.services.project_roster_service import get_project_info, get_project_roster

router = APIRouter(prefix="/health-monitor", tags=["health-monitor"])

@router.get("/projects")
def projects() -> list[dict]:
    return get_health_report()

@router.get("/validation")
def validation() -> dict:
    return get_validation_summary(get_health_report())

@router.get("/projects/{project_code}/roster")
def roster(project_code: str) -> dict:
    return get_project_roster(project_code)

@router.get("/projects/{project_code}/info")
def project_info(project_code: str) -> dict:
    info = get_project_info(project_code)
    if info is None:
        raise HTTPException(status_code=404, detail=f"project_code {project_code!r} not found")
    return info

@router.get("/projects/{project_code}/detail")
def project_detail(project_code: str) -> dict:
    try:
        return get_project_health_detail(project_code)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
