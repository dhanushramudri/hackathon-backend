"""Which delivery cluster each live project belongs to. A real `cluster`
column DOES exist in the fuller JDWH project-table export (260815_Tables
Schema.xlsx's `project` sheet -- confirmed live: 45 of 92 currently-active
projects carry a genuine "Cluster N" value there) and is now mapped through
by jdwh_table_mapping.map_project_table into 02_Project_Details_clean.csv's
`cluster_number` column. The remaining projects (about half) genuinely have
no cluster in the source itself -- not a mapping gap -- so a manual,
app-authored override/fallback is still needed for those, and for any new
project the wizard creates before it round-trips through a JDWH load. A
manual assignment here always wins over the real source value, so the
governance team can still correct a wrong or missing tag from the UI.
Manual storage follows the same single-row-per-project overwrite pattern as
GDPR/Budget/Kickoff, via project_appstate_service's upsert_row/get_row."""
import pandas as pd

from app.core.config import APP_STATE_DIR
from app.services.project_appstate_service import get_row, upsert_row

CLUSTER_CSV = APP_STATE_DIR / "project_cluster_assignment.csv"

# Real cluster names from the JQA Governance weekly deck -- kept as a fixed
# lookup rather than free-text so every cluster view/ball reads consistently.
CLUSTER_NAMES: dict[int, str] = {
    1: "Ganges",
    2: "Tigris",
    3: "Kauveri",
    4: "Patapsco",
    5: "Mississippi",
}

def set_cluster(project_code: str, cluster_number: int) -> dict:
    if cluster_number not in CLUSTER_NAMES:
        raise ValueError(f"cluster_number must be one of {sorted(CLUSTER_NAMES)}")
    return upsert_row(CLUSTER_CSV, project_code, {"cluster_number": cluster_number})

def get_cluster(project_code: str, real_cluster_number: int | None = None) -> int | None:
    row = get_row(CLUSTER_CSV, project_code)
    if row and row.get("cluster_number"):
        return int(row["cluster_number"])
    return real_cluster_number

def _manual_assignments() -> dict[str, int]:
    if not CLUSTER_CSV.exists():
        return {}
    df = pd.read_csv(CLUSTER_CSV, dtype=str).fillna("")
    return {r["project_code"]: int(r["cluster_number"]) for _, r in df.iterrows() if r["cluster_number"]}

def _real_assignments(projects_df: pd.DataFrame) -> dict[str, int]:
    """project_code -> cluster_number straight from the real JDWH-sourced
    column, for whichever rows actually have one set."""
    if "cluster_number" not in projects_df.columns:
        return {}
    real = projects_df.dropna(subset=["cluster_number"])
    return {r["project_code"]: int(r["cluster_number"]) for _, r in real.iterrows()}

def effective_assignments(projects_df: pd.DataFrame) -> dict[str, int]:
    """Real source value, with any manual override applied on top."""
    combined = _real_assignments(projects_df)
    combined.update(_manual_assignments())
    return combined

def list_projects_by_cluster(cluster_number: int, projects_df: pd.DataFrame) -> list[str]:
    assignments = effective_assignments(projects_df)
    active_set = set(projects_df["project_code"])
    return [code for code, n in assignments.items() if n == cluster_number and code in active_set]

def list_unassigned_projects(projects_df: pd.DataFrame) -> list[str]:
    assignments = effective_assignments(projects_df)
    return [code for code in projects_df["project_code"] if code not in assignments]

def cluster_counts(projects_df: pd.DataFrame) -> dict[int, int]:
    assignments = effective_assignments(projects_df)
    active_set = set(projects_df["project_code"])
    counts = {n: 0 for n in CLUSTER_NAMES}
    for code, n in assignments.items():
        if code in active_set:
            counts[n] += 1
    return counts
