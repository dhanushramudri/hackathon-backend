import json

from app.core.config import APP_STATE_DIR
from app.services.project_appstate_service import get_row, upsert_row
from app.services.rate_card_service import get_hourly_rate

BUDGET_CSV = APP_STATE_DIR / "project_budget.csv"

HEADER_FIELDS = ["billing_currency", "engagement_style", "proposition_coe", "payment_term", "is_billable"]

# This app's existing rate authority (already used by demand_forecast_service /
# revenue_engine) is a flat, designation-only, illustrative USD hourly band --
# not the real location-adjusted rate card JIN shows. Reused as-is rather than
# fabricated to look like a precise match.
def get_day_rate(designation: str, hours_per_day: float = 8.0) -> float | None:
    hourly = get_hourly_rate(designation)
    return round(hourly * hours_per_day, 2) if hourly is not None else None

def save_budget(project_code: str, header: dict, line_items: list[dict]) -> dict:
    fields = {k: header.get(k) for k in HEADER_FIELDS}
    fields["line_items"] = json.dumps(line_items)
    upsert_row(BUDGET_CSV, project_code, fields)
    return {"project_code": project_code, **{k: header.get(k) for k in HEADER_FIELDS}, "line_items": line_items}

def get_budget(project_code: str) -> dict | None:
    row = get_row(BUDGET_CSV, project_code)
    if row is None:
        return None
    row = dict(row)
    row["line_items"] = json.loads(row["line_items"]) if row.get("line_items") else []
    return row
