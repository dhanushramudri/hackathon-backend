from app.core.config import APP_STATE_DIR
from app.services.project_appstate_service import get_row, upsert_row

KICKOFF_CSV = APP_STATE_DIR / "project_kickoff.csv"

KICKOFF_FIELDS = [
    "held_internal_session", "used_internal_materials",
    "covered_client_background", "covered_proposal_review", "covered_problem_statement",
    "covered_stakeholders_plan", "covered_client_kickoff_prep", "covered_team_roles",
    "covered_development_goals", "covered_ways_of_working",
    "session_hours", "comments",
]

def save_kickoff(project_code: str, fields: dict) -> dict:
    return upsert_row(KICKOFF_CSV, project_code, {k: fields.get(k) for k in KICKOFF_FIELDS})

def get_kickoff(project_code: str) -> dict | None:
    return get_row(KICKOFF_CSV, project_code)
