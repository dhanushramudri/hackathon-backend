from fastapi import APIRouter, HTTPException
import pandas as pd


from app.services.allocation_report_service import AllocationNotFound, get_allocation_report, get_allocation_timesheet,get_availability_as_of

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



@router.get("/availability")
def availability(as_of_date: str | None = None) -> list[dict]:
    ts = pd.Timestamp(as_of_date) if as_of_date else None
    if ts is not None and ts.normalize() < pd.Timestamp.now().normalize():
        raise HTTPException(status_code=400, detail="as_of_date cannot be in the past")
    return get_availability_as_of(ts)