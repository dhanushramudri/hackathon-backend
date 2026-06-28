from fastapi import APIRouter
from pydantic import BaseModel

from app.services.copilot_service import ask

router = APIRouter(prefix="/buddy", tags=["buddy"])

class HistoryTurn(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    message: str
    history: list[HistoryTurn] = []

@router.post("/ask")
def buddy_ask(req: AskRequest) -> dict:
    return ask(req.message, [h.model_dump() for h in req.history])
