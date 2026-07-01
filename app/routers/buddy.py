import json
import math

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.copilot_service import ask, ask_stream

def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

router = APIRouter(prefix="/buddy", tags=["buddy"])

class HistoryTurn(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    message: str
    history: list[HistoryTurn] = []

@router.post("/ask")
def buddy_ask(req: AskRequest) -> dict:
    return _sanitize(ask(req.message, [h.model_dump() for h in req.history]))

@router.post("/ask/stream")
def buddy_ask_stream(req: AskRequest) -> StreamingResponse:
    history = [h.model_dump() for h in req.history]

    def event_stream():
        for event in ask_stream(req.message, history):
            yield f"data: {json.dumps(_sanitize(event), default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
