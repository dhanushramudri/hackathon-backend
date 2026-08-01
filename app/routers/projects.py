from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.deal_project_link_service import get_project_for_deal, link_deal_to_project
from app.services.project_budget_service import get_budget, get_day_rate, save_budget
from app.services.project_gdpr_service import get_gdpr, save_gdpr
from app.services.project_kickoff_service import get_kickoff, save_kickoff
from app.services.project_service import (
    create_project, list_client_ids, project_code_exists, suggest_project_code, update_project,
)
from app.services.project_sow_service import get_sow_file_path, list_sow_files, save_sow_file

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("/suggest-code")
def suggest_code(name: str) -> dict:
    return {"suggested_code": suggest_project_code(name)}

@router.get("/code-exists")
def code_exists(project_code: str) -> dict:
    return {"exists": project_code_exists(project_code.strip().upper())}

@router.get("/clients")
def clients() -> list[str]:
    return list_client_ids()

@router.get("/deal-link")
def deal_link_get(deal_key: str) -> dict:
    return {"project_code": get_project_for_deal(deal_key)}

@router.post("/deal-link")
def deal_link_set(deal_key: str, project_code: str) -> dict:
    return link_deal_to_project(deal_key, project_code)

class CreateProjectRequest(BaseModel):
    project_code: str
    client_id: str
    type_of_project: str
    start_date: str
    end_date: str
    tech_coe: str | None = None
    proposition_coe: str | None = None
    project_status: str = "ACTIVE"

@router.post("/create")
def create(body: CreateProjectRequest) -> dict:
    try:
        return create_project(**body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

class UpdateProjectRequest(BaseModel):
    client_id: str | None = None
    type_of_project: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    tech_coe: str | None = None
    proposition_coe: str | None = None
    project_status: str | None = None

@router.patch("/{project_code}")
def update(project_code: str, body: UpdateProjectRequest) -> dict:
    try:
        return update_project(project_code, **body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# ── Project GDPR (wizard step 2) ────────────────────────────────────────────

class GdprRequest(BaseModel):
    fields: dict

@router.get("/{project_code}/gdpr")
def gdpr_get(project_code: str) -> dict | None:
    return get_gdpr(project_code)

@router.post("/{project_code}/gdpr")
def gdpr_save(project_code: str, body: GdprRequest) -> dict:
    return save_gdpr(project_code, body.fields)

# ── Budget Creation (wizard step 3) ─────────────────────────────────────────

@router.get("/day-rate")
def day_rate(designation: str, hours_per_day: float = 8.0) -> dict:
    return {"designation": designation, "base_day_rate": get_day_rate(designation, hours_per_day)}

class BudgetRequest(BaseModel):
    header: dict
    line_items: list[dict]

@router.get("/{project_code}/budget")
def budget_get(project_code: str) -> dict | None:
    return get_budget(project_code)

@router.post("/{project_code}/budget")
def budget_save(project_code: str, body: BudgetRequest) -> dict:
    return save_budget(project_code, body.header, body.line_items)

# ── SOW Creation (wizard step 4) ────────────────────────────────────────────

@router.get("/{project_code}/sow")
def sow_list(project_code: str) -> list[dict]:
    return list_sow_files(project_code)

@router.post("/{project_code}/sow")
async def sow_upload(project_code: str, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    return save_sow_file(project_code, file.filename, content)

@router.get("/{project_code}/sow/{filename}")
def sow_download(project_code: str, filename: str) -> FileResponse:
    path = get_sow_file_path(project_code, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=filename)

# ── Project Kickoff (wizard step 6) ─────────────────────────────────────────

class KickoffRequest(BaseModel):
    fields: dict

@router.get("/{project_code}/kickoff")
def kickoff_get(project_code: str) -> dict | None:
    return get_kickoff(project_code)

@router.post("/{project_code}/kickoff")
def kickoff_save(project_code: str, body: KickoffRequest) -> dict:
    return save_kickoff(project_code, body.fields)
