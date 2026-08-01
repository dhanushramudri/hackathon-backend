from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.email_service import EmailNotConfigured, send_email
from app.services.leave_service import build_support_request, get_leave_impact, get_project_alumni_candidates

router = APIRouter(prefix="/leave", tags=["leave"])

class SupportRequestBody(BaseModel):
    employee_id: str
    project_id: str
    start_date: str
    end_date: str

@router.get("/impact")
def impact(
    include_skill: bool = Query(default=True),
    include_competency: bool = Query(default=True),
    include_availability: bool = Query(default=True),
    include_category_match: bool = Query(default=False),
    include_project_count: bool = Query(default=False),
    include_coe_affinity: bool = Query(default=True),
    include_cost_efficiency: bool = Query(default=False),
    include_below_capacity: bool = Query(default=False),
    near_capacity_tolerance_pct: float = Query(default=25.0, ge=0, le=100),
) -> list[dict]:
    return get_leave_impact(
        include_skill=include_skill, include_competency=include_competency, include_availability=include_availability,
        include_category_match=include_category_match, include_project_count=include_project_count,
        include_coe_affinity=include_coe_affinity, include_cost_efficiency=include_cost_efficiency,
        include_below_capacity=include_below_capacity, near_capacity_tolerance_pct=near_capacity_tolerance_pct,
    )

@router.get("/project-alumni")
def project_alumni(project_code: str, exclude_employee_id: str | None = None) -> list[dict]:
    return get_project_alumni_candidates(project_code, exclude_employee_id)

@router.post("/request-support")
def request_support(body: SupportRequestBody) -> dict:
    """Email a backfill candidate asking for their availability, CC'ing the
    project's real manager and the candidate's real reporting manager (CDM
    proxy) -- a manual outreach nudge only, no allocation is created here."""
    req = build_support_request(body.employee_id, body.project_id, body.start_date, body.end_date)
    try:
        send_email(req["employee_email"], req["subject"], req["html"], cc=req["cc_emails"])
    except EmailNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send via Mailtrap: {exc}") from exc
    return {
        "sent_to": req["employee_email"],
        "cc": req["cc_emails"],
        "project_manager_employee_id": req["project_manager_employee_id"],
        "cdm_employee_id": req["cdm_employee_id"],
        "subject": req["subject"],
    }
