from fastapi import APIRouter, HTTPException, Query

from app.engines.experience_engine import get_employee_project_history
from app.engines.feedback_engine import get_employee_feedback
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

@router.get("/{employee_id}/feedback")
def feedback(
    employee_id: str,
    weeks_back: int | None = Query(default=None, ge=1, description="Only feedback from the last N weeks"),
    coe: str | None = Query(default=None, description="Project's primary tech CoE"),
    project_id: str | None = Query(default=None),
    reviewer_employee_id: str | None = Query(default=None, description="Only feedback written by this real reviewer"),
    theme: str | None = Query(default=None, description="Only reviews that cover this theme"),
    ratings: list[int] | None = Query(default=None, description="Only these specific rating values (1-5), any combination"),
) -> dict:
    """HR/PM feedback for one employee, filterable -- manual review/proof
    surface only. Never used as an input to recommendation scoring."""
    return get_employee_feedback(
        employee_id, weeks_back=weeks_back, coe=coe, project_id=project_id,
        reviewer_employee_id=reviewer_employee_id, theme=theme, ratings=ratings,
    )
