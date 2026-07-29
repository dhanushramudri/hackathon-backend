from fastapi import APIRouter, Query

from app.services.free_pool_service import get_free_pool
from app.services.recommendation_service import get_redeploy_matches_for_employee

router = APIRouter(prefix="/free-pool", tags=["free-pool"])

@router.get("")
def free_pool() -> list[dict]:
    return get_free_pool()

@router.get("/{employee_id}/matches")
def free_pool_matches(
    employee_id: str,
    top_n: int = 20,
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
    return get_redeploy_matches_for_employee(
        employee_id, top_n=top_n,
        include_skill=include_skill, include_competency=include_competency, include_availability=include_availability,
        include_category_match=include_category_match, include_project_count=include_project_count,
        include_coe_affinity=include_coe_affinity, include_cost_efficiency=include_cost_efficiency,
        include_below_capacity=include_below_capacity, near_capacity_tolerance_pct=near_capacity_tolerance_pct,
    )
