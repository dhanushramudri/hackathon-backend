from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.services.deal_project_link_service import get_project_for_deal, link_deal_to_project
from app.services.project_budget_service import get_budget, get_day_rate, save_budget
from app.services.project_gdpr_service import get_gdpr, save_gdpr
from app.services.project_kickoff_service import get_kickoff, save_kickoff
from app.services.project_service import (
    create_project, list_client_ids, project_code_exists, suggest_project_code, update_project,
)
from app.services.docx_conversion_service import DocxConversionError, get_or_convert_pdf
from app.services.project_sow_service import get_sow_file_path, list_sow_files, save_sow_file
from app.services.sow_chat_service import SowChatError, ask_sow_question
from app.services.sow_extraction_service import SowExtractionError, extract_sow_requirements, get_cached_extraction

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

@router.get("/{project_code}/sow/{filename}/view")
def sow_view(project_code: str, filename: str) -> Response:
    """Same file, served inline (not as a forced download) -- for an <iframe>
    PDF preview (with a #page=N jump) or a client-side docx-to-HTML render."""
    path = get_sow_file_path(project_code, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}.get(ext, "application/octet-stream")
    return Response(content=path.read_bytes(), media_type=content_type, headers={"Content-Disposition": f'inline; filename="{filename}"'})

@router.get("/{project_code}/sow/{filename}/preview-pdf")
def sow_preview_pdf(project_code: str, filename: str) -> Response:
    """A real PDF for this SOW, pixel-perfect -- the original file if it's
    already a PDF, or a real LibreOffice-rendered conversion (cached) if it's
    a .docx. 503 if conversion is needed but unavailable on this machine, so
    the frontend can fall back to the lower-fidelity HTML preview."""
    try:
        pdf_path = get_or_convert_pdf(project_code, filename)
    except DocxConversionError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return Response(content=pdf_path.read_bytes(), media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{filename}.pdf"'})

@router.get("/{project_code}/sow/{filename}/extract")
def sow_extraction_get(project_code: str, filename: str) -> dict:
    """Cached extraction result if this SOW has already been extracted --
    lets the UI show prior results without re-calling the AI on every visit."""
    cached = get_cached_extraction(project_code, filename)
    return cached or {"available": False}

@router.post("/{project_code}/sow/{filename}/extract")
def sow_extraction_run(project_code: str, filename: str) -> dict:
    """Run AI extraction on an already-uploaded SOW: roles, skills, and
    competency areas required, grounded strictly in the document's own text.
    On-demand only -- never runs automatically on upload."""
    try:
        return extract_sow_requirements(project_code, filename)
    except SowExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

class SowChatMessage(BaseModel):
    role: str
    content: str

class SowChatRequest(BaseModel):
    message: str
    history: list[SowChatMessage] = []

@router.post("/{project_code}/sow/{filename}/chat")
def sow_chat(project_code: str, filename: str, body: SowChatRequest) -> dict:
    """Ask a question scoped strictly to this one SOW document -- every
    citation returned is verified to be a real, exact quote from the
    document before being shown."""
    try:
        return ask_sow_question(project_code, filename, body.message, [m.model_dump() for m in body.history])
    except SowChatError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

# ── Project Kickoff (wizard step 6) ─────────────────────────────────────────

class KickoffRequest(BaseModel):
    fields: dict

@router.get("/{project_code}/kickoff")
def kickoff_get(project_code: str) -> dict | None:
    return get_kickoff(project_code)

@router.post("/{project_code}/kickoff")
def kickoff_save(project_code: str, body: KickoffRequest) -> dict:
    return save_kickoff(project_code, body.fields)
