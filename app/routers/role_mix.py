from fastapi import APIRouter

from app.engines.coe_skill_engine import derive_skills_for_coes
from app.engines.role_mix_engine import get_role_mix, list_coes, list_docx_categories, list_role_mix_templates

router = APIRouter(prefix="/role-mix", tags=["role-mix"])

@router.get("/templates")
def templates() -> list[dict]:
    return list_role_mix_templates()

@router.get("/lookup")
def lookup(type_of_project: str, tech_coe: str | None = None) -> dict:
    return get_role_mix(type_of_project, tech_coe)

@router.get("/categories")
def categories() -> list[dict]:
    return list_docx_categories()

@router.get("/coes")
def coes() -> list[dict]:
    return list_coes()

@router.get("/coe-skills")
def coe_skills(coes: str) -> dict:
    coe_list = [c.strip() for c in coes.split(",") if c.strip()]
    return derive_skills_for_coes(coe_list)
