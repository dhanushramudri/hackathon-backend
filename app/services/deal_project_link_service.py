import pandas as pd

from app.core.config import APP_STATE_DIR

# So re-opening the Project Wizard on a deal later (even after a page refresh
# or a new session) resumes at Step 5 instead of offering to create a second,
# duplicate project for the same deal.
LINK_CSV = APP_STATE_DIR / "deal_project_link.csv"

def link_deal_to_project(deal_key: str, project_code: str) -> dict:
    row = {"deal_key": deal_key, "project_code": project_code}
    if LINK_CSV.exists():
        df = pd.read_csv(LINK_CSV, dtype=str)
        df = df[df["deal_key"] != deal_key]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(LINK_CSV, index=False)
    return row

def get_project_for_deal(deal_key: str) -> str | None:
    if not LINK_CSV.exists():
        return None
    df = pd.read_csv(LINK_CSV, dtype=str)
    match = df[df["deal_key"] == deal_key]
    return match.iloc[-1]["project_code"] if not match.empty else None
