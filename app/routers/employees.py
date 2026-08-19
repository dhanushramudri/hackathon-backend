from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile

from app.engines.experience_engine import get_employee_project_history
from app.engines.feedback_engine import get_employee_feedback
from app.engines.performance_engine import get_cycle_detail, list_employee_cycles
from app.services.employee_profile_service import (
    EmployeeNotFound,
    get_employee_headcount_summary,
    get_employee_profile,
    get_overtime_risk_summary,
    list_designations,
    list_employee_groups,
    list_employees,
)
from app.services.performance_summary_service import get_performance_ai_summary
from app.services.resume_skill_service import (
    ResumeProcessingError,
    get_document_import_file,
    list_document_imports,
    process_resume,
)
from app.services.timesheet_insights_service import get_employee_timesheet_entries
from app.engines.pulse_engine import get_employee_all_pulse_responses

router = APIRouter(prefix="/employees", tags=["employees"])

@router.get("")
def list_all() -> list[dict]:
    return list_employees()

@router.get("/designations")
def designations() -> list[str]:
    return list_designations()

@router.get("/groups")
def employee_groups() -> list[str]:
    """Region/entity/employment-type groups derived from employee_id prefixes
    (JMD/JMG/JML/JMU/INT/TRN/EXT/CRN) -- backs the Employees page group filter."""
    return list_employee_groups()

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

@router.get("/{employee_id}/performance-cycles")
def performance_cycles(employee_id: str) -> list[dict]:
    """Half-yearly KRA/appraisal cycle list for this employee -- manual
    review/proof surface only, mirrors the real "KRA-KPI Forms" list page."""
    return list_employee_cycles(employee_id)

@router.get("/{employee_id}/performance-cycles/{cycle_id}")
def performance_cycle_detail(employee_id: str, cycle_id: str) -> dict:
    """Full detail for one KRA cycle -- stage tracker, total score/grade, and
    the Projects/People/Products/Sales/Overall-Feedback sections."""
    detail = get_cycle_detail(employee_id, cycle_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Performance cycle not found for this employee.")
    return detail

@router.get("/{employee_id}/performance-summary")
def performance_summary(employee_id: str) -> dict:
    """AI summary of this employee's closed KRA cycles, focused on the
    Projects/Products KRAs and Overall Feedback -- a fast read for the RM
    instead of opening every cycle."""
    return get_performance_ai_summary(employee_id)

@router.get("/{employee_id}/timesheet")
def timesheet(
    employee_id: str,
    start_date: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
    end_date: str | None = Query(default=None, description="YYYY-MM-DD, inclusive"),
    project_id: str | None = Query(default=None),
    billing_status: str | None = Query(default=None),
) -> dict:
    """Real per-day timesheet rows for this employee, filterable by date range,
    project, and billing status -- the raw proof surface for the profile modal."""
    return get_employee_timesheet_entries(
        employee_id, start_date=start_date, end_date=end_date, project_id=project_id, billing_status=billing_status,
    )

@router.post("/{employee_id}/resume")
async def upload_resume(employee_id: str, file: UploadFile = File(...), channel: str = Form(default="resume")) -> dict:
    """Upload a real resume or LinkedIn-exported PDF; extracts a real skills
    list via this app's own LLM provider chain and appends any new skills to
    this employee's real skill record (tagged skill_source="{channel}_extracted",
    never overwriting or duplicating existing skill rows)."""
    content = await file.read()
    try:
        return process_resume(employee_id, file.filename or "", content, channel=channel)
    except ResumeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.get("/{employee_id}/document-imports")
def document_imports(employee_id: str) -> list[dict]:
    """Every past successful resume/LinkedIn-PDF import for this employee --
    lets the Skills tab show "already imported" state instead of re-asking
    blind, with a reupload option."""
    return list_document_imports(employee_id)

@router.get("/{employee_id}/document-imports/{import_id}/file")
def document_import_file(employee_id: str, import_id: str) -> Response:
    """Raw bytes of one past import's original file, for an in-browser PDF
    preview or a Word-doc download."""
    try:
        content, filename, content_type = get_document_import_file(employee_id, import_id)
    except ResumeProcessingError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(content=content, media_type=content_type, headers={"Content-Disposition": f'inline; filename="{filename}"'})

@router.get("/{employee_id}/pulse")
def employee_pulse(employee_id: str) -> list[dict]:
    """Every real Weekly Pulse response this employee has submitted -- backs
    the Timesheet tab's per-week "Weekly Pulse" button."""
    return get_employee_all_pulse_responses(employee_id)
