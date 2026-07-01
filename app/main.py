import logging
import os
import threading

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import CORS_ORIGINS
from app.core.db import get_connection, table_counts
from app.core.safe_json import SafeJSONResponse
from app.routers import allocations, buddy, digest, employees, forecast, free_pool, leave, pipeline, recommendations, revenue, role_mix, wellbeing
from app.routers import health as health_monitor_router
from app.services.digest_service import build_digest
from app.services.email_service import render_digest_html, send_email

logger = logging.getLogger("resourceiq.scheduler")
logger_warmup = logging.getLogger("resourceiq.warmup")
scheduler = BackgroundScheduler()


def _warmup_embedding_model() -> None:
    """Load the SentenceTransformer model and employee embeddings into this worker's RAM.

    Runs in a daemon thread on startup so uvicorn finishes binding immediately.
    By the time a user clicks the first recommendation row, the model is already
    warm in every worker — no 30-second wait on first request.

    The employee vectors are persisted in .embedding_cache/emb_*.npz so they are
    not re-encoded on restart — only the model weights are loaded into RAM (~5-15s).
    """
    try:
        from app.core.adapter import get_adapter
        from app.engines.embedding_engine import build_employee_embedding_index
        logger_warmup.info("Warming up embedding model…")
        skills = get_adapter().get_skills()
        build_employee_embedding_index(skills)
        logger_warmup.info("Embedding model warm — worker ready for semantic matching.")
    except Exception:
        logger_warmup.warning("Embedding warmup failed (non-fatal — word-token matching still works).", exc_info=True)

def _send_scheduled_digest(period_label: str) -> None:
    recipient = os.environ.get("DIGEST_RECIPIENT_EMAIL", "")
    if not recipient:
        logger.warning("Skipping scheduled digest -- DIGEST_RECIPIENT_EMAIL not set.")
        return
    try:
        digest = build_digest()
        html = render_digest_html(digest, period_label)
        send_email(recipient, f"ResourceIQ Digest — {period_label}", html)
    except Exception:
        logger.exception("Scheduled digest send failed")

app = FastAPI(
    title="ResourceIQ API",
    description="JMAN resourcing co-pilot -- backend for the 5 use-case engines.",
    version="0.1.0",
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_data() -> None:
    get_connection()
    # Warm up the embedding model in the background so the first recommendation
    # request is instant instead of waiting 30s for PyTorch to initialise.
    threading.Thread(target=_warmup_embedding_model, daemon=True).start()
    # Friday EOD: what's still unresolved before the weekend. Monday AM: what to
    # tackle first thing this week. Same digest content, different framing.
    scheduler.add_job(_send_scheduled_digest, "cron", day_of_week="fri", hour=18, minute=0, args=["this weekend"], id="friday_eod_digest")
    scheduler.add_job(_send_scheduled_digest, "cron", day_of_week="mon", hour=8, minute=0, args=["this week"], id="monday_am_digest")
    scheduler.start()

@app.on_event("shutdown")
def stop_scheduler() -> None:
    scheduler.shutdown(wait=False)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/meta/tables")
def meta_tables() -> dict[str, int]:
    return table_counts()

app.include_router(role_mix.router)
app.include_router(allocations.router)
app.include_router(recommendations.router)
app.include_router(health_monitor_router.router)
app.include_router(forecast.router)
app.include_router(pipeline.router)
app.include_router(buddy.router)
app.include_router(free_pool.router)
app.include_router(revenue.router)
app.include_router(leave.router)
app.include_router(employees.router)
app.include_router(digest.router)
app.include_router(wellbeing.router)
