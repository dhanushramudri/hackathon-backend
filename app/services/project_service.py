import re
import uuid
import pandas as pd

from app.core.adapter import get_adapter
from app.core.config import TRANSFORMED_DIR
from app.core import db as db_module

PROJECTS_CSV = TRANSFORMED_DIR / "02_Project_Details_clean.csv"
PROJECT_CODE_RE = re.compile(r"^[A-Z]{3}_\d{3}$")

# ponytail: word list is a heuristic stoplist, not exhaustive -- good enough
# for a suggestion the user edits anyway, not a source of truth.
_STOPWORDS = {
    "group", "capital", "partners", "holdings", "limited", "ltd", "llp", "llc",
    "inc", "pe", "fp&a", "fp", "a", "the", "and", "of",
}

def _clean_words(name: str) -> list[str]:
    # Split on non-letter separators (-, &, /, whitespace), drop generic/short
    # abbreviation-like tokens, keep whatever looks like a real proper noun.
    raw = re.split(r"[^A-Za-z]+", name)
    return [w for w in raw if w and w.lower() not in _STOPWORDS]

def _prefix_from_name(name: str) -> str:
    words = _clean_words(name)
    if not words:
        return "GEN"
    longest = max(words, key=len)
    letters = re.sub(r"[^A-Za-z]", "", longest).upper()
    if len(letters) >= 3:
        return letters[:3]
    # too short on its own -- pad from the next-longest word
    rest = "".join(w.upper() for w in sorted(words, key=len, reverse=True)[1:])
    return (letters + rest + "XXX")[:3]

def suggest_project_code(name: str) -> str:
    """3 letters derived from the (client/deal) name + the next available
    _NNN phase number for that prefix, e.g. "Kasaya" -> KYA_001. A starting
    point only -- the caller can edit the prefix/number before creating."""
    prefix = _prefix_from_name(name)
    adapter = get_adapter()
    existing = adapter.get_projects()["project_code"].dropna().astype(str)
    used_numbers = set()
    for code in existing:
        m = re.match(rf"^{prefix}_(\d{{3}})$", code)
        if m:
            used_numbers.add(int(m.group(1)))
    n = 1
    while n in used_numbers:
        n += 1
    return f"{prefix}_{n:03d}"

def project_code_exists(project_code: str) -> bool:
    adapter = get_adapter()
    existing = adapter.get_projects()["project_code"].dropna().astype(str)
    return project_code in set(existing)

def create_project(
    project_code: str, client_id: str, type_of_project: str,
    start_date: str, end_date: str,
    tech_coe: str | None = None, proposition_coe: str | None = None,
    project_status: str = "ACTIVE",
) -> dict:
    """Turn a pipeline deal into a real project -- appends a row to the source
    CSV (same source-of-truth pattern as create_allocation) and reloads the DB
    so it's immediately assignable-against."""
    project_code = project_code.strip().upper()
    if not PROJECT_CODE_RE.match(project_code):
        raise ValueError("project_code must be 3 letters + underscore + 3 digits, e.g. KYA_001")
    if project_code_exists(project_code):
        raise ValueError(f"Project code {project_code!r} already exists")
    if pd.to_datetime(end_date) < pd.to_datetime(start_date):
        raise ValueError("end_date cannot be before start_date")

    df = pd.read_csv(PROJECTS_CSV)
    df.columns = [c.strip() for c in df.columns]
    new_row = {
        "project_key": uuid.uuid4().hex,
        "project_code": project_code,
        "project_start_date": start_date,
        "project_end_date": end_date,
        "type_of_project": type_of_project,
        "project_status": project_status,
        "reporter_employee_id": None,
        "approver_employee_id": None,
        "client_id": client_id,
        "tech_coe": tech_coe,
        "proposition_coe": proposition_coe,
        "is_active_version": 1,
        # "given" (not a made-up marker) so every real-project filter elsewhere
        # in the app (role_mix_engine, revenue_engine, health monitor) treats
        # this exactly like any other real project -- see date_source usage.
        "date_source": "given",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(PROJECTS_CSV, index=False)
    db_module.reload()
    return new_row

if __name__ == "__main__":
    assert _prefix_from_name("Kasaya") == "KAS"
    assert _prefix_from_name("GlAS") == "GLA"
    assert _prefix_from_name("Hg Capital - Caseware") == "CAS"
    assert _prefix_from_name("H&F - Baker Tilly - FP&A") == "BAK"
    assert PROJECT_CODE_RE.match("KYA_001") and not PROJECT_CODE_RE.match("KY_001")
    print("project_service self-check OK")
