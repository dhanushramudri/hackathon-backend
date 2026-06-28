from fastapi import APIRouter

from app.services.leave_service import get_leave_impact

router = APIRouter(prefix="/leave", tags=["leave"])

@router.get("/impact")
def impact() -> list[dict]:
    return get_leave_impact()
