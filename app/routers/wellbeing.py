from fastapi import APIRouter

from app.services.wellbeing_service import get_employee_burnout_overview, get_project_burnout_overview

router = APIRouter(prefix="/wellbeing", tags=["wellbeing"])

@router.get("/projects")
def projects() -> dict:
    return get_project_burnout_overview()

@router.get("/employees")
def employees() -> dict:
    return get_employee_burnout_overview()
