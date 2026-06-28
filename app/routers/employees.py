from fastapi import APIRouter, HTTPException

from app.services.employee_profile_service import EmployeeNotFound, get_employee_headcount_summary, get_employee_profile

router = APIRouter(prefix="/employees", tags=["employees"])

@router.get("/headcount-summary")
def headcount_summary() -> dict:
    return get_employee_headcount_summary()

@router.get("/{employee_id}/profile")
def profile(employee_id: str) -> dict:
    try:
        return get_employee_profile(employee_id)
    except EmployeeNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
