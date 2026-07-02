import json
import math
import os
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.copilot_service import ask, ask_stream

RATINGS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "buddy_ratings.jsonl")
)

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
    prior_context: str | None = None

@router.post("/ask")
def buddy_ask(req: AskRequest) -> dict:
    return _sanitize(ask(req.message, [h.model_dump() for h in req.history], req.prior_context))

@router.post("/ask/stream")
def buddy_ask_stream(req: AskRequest) -> StreamingResponse:
    history = [h.model_dump() for h in req.history]

    def event_stream():
        for event in ask_stream(req.message, history, req.prior_context):
            yield f"data: {json.dumps(_sanitize(event), default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class RatingRequest(BaseModel):
    session_id: str
    message_index: int
    question: str
    answer_snippet: str
    rating: str  # "up" or "down"
    timestamp: str | None = None


@router.post("/rate")
def buddy_rate(req: RatingRequest) -> dict:
    os.makedirs(os.path.dirname(RATINGS_FILE), exist_ok=True)
    entry = {
        "session_id": req.session_id,
        "message_index": req.message_index,
        "question": req.question[:300],
        "answer_snippet": req.answer_snippet[:300],
        "rating": req.rating,
        "timestamp": req.timestamp or datetime.now(timezone.utc).isoformat(),
    }
    with open(RATINGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True}


@router.get("/ratings")
def buddy_ratings() -> dict:
    if not os.path.exists(RATINGS_FILE):
        return {"total": 0, "positive": 0, "negative": 0, "pct_helpful": None, "recent": []}
    entries = []
    with open(RATINGS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    total = len(entries)
    positive = sum(1 for e in entries if e.get("rating") == "up")
    negative = total - positive
    pct = round(positive / total * 100) if total > 0 else None
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "pct_helpful": pct,
        "recent": list(reversed(entries[-20:])),
    }
