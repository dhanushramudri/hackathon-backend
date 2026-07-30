from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd


from app.services.allocation_report_service import (
    AllocationNotFound, AllocationRowNotFound, ProjectNotFoundForExtension,
    create_allocation, extend_allocation_end_date, extend_project_end_date,
    get_allocation_report, get_allocation_timesheet, get_availability_as_of,
)

router = APIRouter(prefix="/allocations", tags=["allocations"])

class AssignRequest(BaseModel):
    employee_id: str
    project_id: str
    allocation_pct: float
    start_date: str
    end_date: str
    resourcing_status: str = "BILLABLE"

@router.post("/assign")
def assign(body: AssignRequest) -> dict:
    try:
        return create_allocation(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

class ExtendEndDateRequest(BaseModel):
    extended_end_date: str | None = None
    status: str | None = None

@router.post("/{allocation_id}/extend")
def extend_allocation(allocation_id: str, body: ExtendEndDateRequest) -> dict:
    try:
        return extend_allocation_end_date(allocation_id, body.extended_end_date or "", body.status)
    except AllocationRowNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

class ExtendProjectEndDateRequest(BaseModel):
    extended_end_date: str | None = None
    status: str | None = None

@router.post("/projects/{project_code}/extend")
def extend_project(project_code: str, body: ExtendProjectEndDateRequest) -> dict:
    try:
        return extend_project_end_date(project_code, body.extended_end_date or "", body.status)
    except ProjectNotFoundForExtension as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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