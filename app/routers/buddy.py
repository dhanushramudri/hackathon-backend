import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.copilot_service import ask, ask_stream

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

@router.post("/ask/stream")
def buddy_ask_stream(req: AskRequest) -> StreamingResponse:
    history = [h.model_dump() for h in req.history]

    def event_stream():
        for event in ask_stream(req.message, history):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
