from fastapi import APIRouter

from app.services.revenue_service import get_revenue_trend

router = APIRouter(prefix="/revenue", tags=["revenue"])

@router.get("/trend")
def trend() -> list[dict]:
    return get_revenue_trend()
