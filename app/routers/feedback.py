from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.feedback_service import list_feedback, submit_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    name: str | None = None
    category: str
    message: str


@router.get("")
def get_feedback() -> list[dict]:
    return list_feedback()


@router.post("")
def post_feedback(req: FeedbackRequest) -> dict:
    try:
        return submit_feedback(req.name, req.category, req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
