from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel

class Employee(BaseModel):
    employee_id: str
    location: Optional[str] = None
    date_of_join: Optional[datetime] = None
    date_of_resignation: Optional[datetime] = None
    job_name: Optional[str] = None
    department_name: Optional[str] = None
    manager_employee_id: Optional[str] = None
    account_status: Optional[int] = None
    is_active_version: Optional[int] = None

class Project(BaseModel):
    project_key: str
    project_code: str
    type_of_project: Optional[str] = None
    project_status: Optional[str] = None
    reporter_employee_id: Optional[str] = None
    approver_employee_id: Optional[str] = None
    tech_coe: Optional[str] = None
    proposition_coe: Optional[str] = None
    project_start_date: Optional[datetime] = None
    project_end_date: Optional[datetime] = None
    date_source: str

class Allocation(BaseModel):
    project_rolebased_user_id: str
    project_id: Optional[str] = None
    employee_id: str
    resourcing_status: str
    allocated_start_date: Optional[datetime] = None
    allocated_end_date: Optional[datetime] = None
    is_allocation_active: int
    allocation_by_percentage: float

class TimesheetEntry(BaseModel):
    timesheet_surrogate_key: str
    employee_id: Optional[str] = None
    project_id: Optional[str] = None
    time: Optional[float] = None
    status: Optional[str] = None
    job_name: Optional[str] = None
    department_name: Optional[str] = None

class Skill(BaseModel):
    employee_id: str
    designation: Optional[str] = None
    coe: Optional[str] = None
    coe_skill: Optional[str] = None
    skill: Optional[str] = None
    subskill: Optional[str] = None
    experience: Optional[str] = None
    score: Optional[float] = None
    skill_source: str

class Competency(BaseModel):
    employee_id: str
    designation: Optional[str] = None
    coe_dep: Optional[str] = None
    competency_sheet: str
    competency_question: str
    response: Optional[str] = None
    score: Optional[float] = None
    competency_source: str

class WsrReport(BaseModel):
    wsr_key: str
    wsr_id: str
    project_id_masked: str
    scope_status: str
    schedule_status: str
    quality_status: str
    csat_status: str
    team_status: str

class PipelineDemand(BaseModel):
    cluster: Optional[int] = None
    client: Optional[str] = None
    client_priority: Optional[str] = None
    likely_start_date: Optional[date] = None
    solution: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    resources_requested: Optional[str] = None
    requested_pct: Optional[str] = None
    skillset: Optional[str] = None
    sow_signed: Optional[str] = None
    comments: Optional[str] = None
