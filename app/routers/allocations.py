from fastapi import APIRouter, HTTPException

from app.services.allocation_report_service import AllocationNotFound, get_allocation_report, get_allocation_timesheet

router = APIRouter(prefix="/allocations", tags=["allocations"])

@router.get("/current")
def current() -> list[dict]:
    return get_allocation_report()

@router.get("/timesheet")
def timesheet(employee_id: str, project_id: str) -> dict:
    try:
        return get_allocation_timesheet(employee_id, project_id)
    except AllocationNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
