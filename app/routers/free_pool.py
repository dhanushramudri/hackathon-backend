from fastapi import APIRouter

from app.services.free_pool_service import get_free_pool
from app.services.recommendation_service import get_redeploy_matches_for_employee

router = APIRouter(prefix="/free-pool", tags=["free-pool"])

@router.get("")
def free_pool() -> list[dict]:
    return get_free_pool()

@router.get("/{employee_id}/matches")
def free_pool_matches(employee_id: str, top_n: int = 20) -> list[dict]:
    return get_redeploy_matches_for_employee(employee_id, top_n=top_n)
