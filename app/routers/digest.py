import os

from fastapi import APIRouter, HTTPException

from app.services.digest_service import build_digest
from app.services.email_service import EmailNotConfigured, render_digest_html, send_email

router = APIRouter(prefix="/digest", tags=["digest"])

@router.get("/preview")
def preview_digest(period_label: str = "this week") -> dict:
    digest = build_digest()
    return {"digest": digest, "html": render_digest_html(digest, period_label)}

@router.post("/send")
def send_digest_now(period_label: str = "right now") -> dict:
    recipient = os.environ.get("DIGEST_RECIPIENT_EMAIL", "")
    if not recipient:
        raise HTTPException(status_code=400, detail="DIGEST_RECIPIENT_EMAIL is not set in the environment.")
    digest = build_digest()
    html = render_digest_html(digest, period_label)
    try:
        send_email(recipient, f"ResourceIQ Digest — {period_label}", html)
    except EmailNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send via Mailtrap: {exc}") from exc
    return {
        "sent_to": recipient,
        "no_backfill_count": digest["no_backfill_count"],
        "high_risk_total_count": digest["high_risk_total_count"],
    }
