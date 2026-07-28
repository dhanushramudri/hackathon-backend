from fastapi import APIRouter, HTTPException, Query

from app.engines.experience_engine import get_employee_project_history
from app.services.employee_profile_service import (
    EmployeeNotFound,
    get_employee_headcount_summary,
    get_employee_profile,
    get_overtime_risk_summary,
    list_designations,
    list_employees,
)

router = APIRouter(prefix="/employees", tags=["employees"])

@router.get("")
def list_all() -> list[dict]:
    return list_employees()

@router.get("/designations")
def designations() -> list[str]:
    return list_designations()

@router.get("/headcount-summary")
def headcount_summary() -> dict:
    return get_employee_headcount_summary()

@router.get("/overtime-risk-summary")
def overtime_risk_summary() -> dict:
    return get_overtime_risk_summary()

@router.get("/{employee_id}/profile")
def profile(employee_id: str) -> dict:
    try:
        return get_employee_profile(employee_id)
    except EmployeeNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.get("/{employee_id}/project-history")
def project_history(employee_id: str, category: str | None = Query(default=None)) -> list[dict]:
    """Raw project list for click-to-drill on a recommendation candidate's
    track record (a category chip, the project count, or the client count)."""
    return get_employee_project_history(employee_id, category=category)
