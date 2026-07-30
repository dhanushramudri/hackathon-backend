from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.project_service import create_project, project_code_exists, suggest_project_code

router = APIRouter(prefix="/projects", tags=["projects"])

@router.get("/suggest-code")
def suggest_code(name: str) -> dict:
    return {"suggested_code": suggest_project_code(name)}

@router.get("/code-exists")
def code_exists(project_code: str) -> dict:
    return {"exists": project_code_exists(project_code.strip().upper())}

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
