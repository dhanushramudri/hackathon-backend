from fastapi import APIRouter, Query

from app.services.leave_service import get_leave_impact

router = APIRouter(prefix="/leave", tags=["leave"])

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
