from app.core.config import APP_STATE_DIR
from app.services.project_appstate_service import get_row, upsert_row

GDPR_CSV = APP_STATE_DIR / "project_gdpr.csv"

GDPR_FIELDS = [
    "personal_data_collected", "purpose", "retention_period", "special_category_data",
    "special_category_conditions", "legal_basis", "under_13_data", "data_processed",
    "data_storage_location", "dpa_signed", "transfer_to_jman_digital",
    "transfer_to_third_parties", "third_parties",
]

def save_gdpr(project_code: str, fields: dict) -> dict:
    return upsert_row(GDPR_CSV, project_code, {k: fields.get(k) for k in GDPR_FIELDS})

def get_gdpr(project_code: str) -> dict | None:
    return get_row(GDPR_CSV, project_code)
