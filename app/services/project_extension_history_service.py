"""Append-only log of every project end-date extension change -- unlike the
single mutable extended_end_date/extended_end_status columns on the projects
CSV (which only ever hold the CURRENT value), this keeps every past change as
its own record so the Extensions tab can show a real history, not just the
latest state."""
import pandas as pd

from app.core.config import APP_STATE_DIR

EXTENSION_HISTORY_CSV = APP_STATE_DIR / "project_extensions.csv"
_FIELDS = ["project_code", "recorded_at", "from_end_date", "to_end_date", "status"]


def record_extension(project_code: str, from_end_date: str | None, to_end_date: str | None, status: str | None) -> dict:
    row = {
        "project_code": project_code,
        "recorded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "from_end_date": from_end_date or "",
        "to_end_date": to_end_date or "",
        "status": status or "",
    }
    if EXTENSION_HISTORY_CSV.exists():
        df = pd.read_csv(EXTENSION_HISTORY_CSV, dtype=str).fillna("")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=_FIELDS)
    df.to_csv(EXTENSION_HISTORY_CSV, index=False)
    return row


def get_extension_history(project_code: str) -> list[dict]:
    if not EXTENSION_HISTORY_CSV.exists():
        return []
    df = pd.read_csv(EXTENSION_HISTORY_CSV, dtype=str).fillna("")
    rows = df[df["project_code"] == project_code].to_dict("records")
    rows.sort(key=lambda r: r["recorded_at"], reverse=True)
    return [{k: (v if v != "" else None) for k, v in r.items()} for r in rows]
