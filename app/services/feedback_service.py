import pandas as pd

from app.core.config import APP_STATE_DIR

FEEDBACK_CSV = APP_STATE_DIR / "user_feedback.csv"
_COLUMNS = ["submitted_at", "name", "category", "message"]
_VALID_CATEGORIES = {"Bug", "Feature request", "General"}


def submit_feedback(name: str | None, category: str, message: str) -> dict:
    message = (message or "").strip()
    if not message:
        raise ValueError("Feedback message cannot be empty.")
    category = category if category in _VALID_CATEGORIES else "General"

    row = {
        "submitted_at": pd.Timestamp.now().isoformat(),
        "name": (name or "").strip() or "Anonymous",
        "category": category,
        "message": message,
    }
    if FEEDBACK_CSV.exists():
        df = pd.read_csv(FEEDBACK_CSV, dtype=str)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(FEEDBACK_CSV, index=False)
    return row


def list_feedback() -> list[dict]:
    if not FEEDBACK_CSV.exists():
        return []
    df = pd.read_csv(FEEDBACK_CSV, dtype=str).fillna("")
    rows = df.sort_values("submitted_at", ascending=False)
    return rows.to_dict(orient="records")
