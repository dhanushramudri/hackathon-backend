"""Per-project risk/issue log for the Cluster Governance view -- the live
equivalent of the weekly JQA deck's "Risks & Key Projects" table (Project |
Risk Description | Risk Type | Suggested Mitigation Steps). No real source-
system data carries this (it's pure narrative the governance team currently
types into a PowerPoint), so it's app-authored, append-only -- unlike the
single-row-per-project AppState pattern, a project can have more than one
open risk at once, and resolved risks stay on record rather than being
overwritten/deleted, mirroring project_extension_history_service's shape."""
import uuid

import pandas as pd

from app.core.config import APP_STATE_DIR

RISK_CSV = APP_STATE_DIR / "project_governance_risk.csv"
_FIELDS = [
    "risk_id", "project_code", "risk_description", "risk_type", "mitigation_steps",
    "created_at", "resolved", "resolved_at",
]

def _read_all() -> pd.DataFrame:
    if not RISK_CSV.exists():
        return pd.DataFrame(columns=_FIELDS)
    return pd.read_csv(RISK_CSV, dtype=str).fillna("")

def _write_all(df: pd.DataFrame) -> None:
    df.to_csv(RISK_CSV, index=False)

def add_risk(project_code: str, risk_description: str, risk_type: str | None, mitigation_steps: str | None) -> dict:
    row = {
        "risk_id": uuid.uuid4().hex,
        "project_code": project_code,
        "risk_description": risk_description,
        "risk_type": risk_type or "",
        "mitigation_steps": mitigation_steps or "",
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "resolved": "false",
        "resolved_at": "",
    }
    df = pd.concat([_read_all(), pd.DataFrame([row])], ignore_index=True)
    _write_all(df)
    return row

def resolve_risk(risk_id: str) -> dict | None:
    df = _read_all()
    match = df["risk_id"] == risk_id
    if not match.any():
        return None
    df.loc[match, "resolved"] = "true"
    df.loc[match, "resolved_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    _write_all(df)
    row = df[match].iloc[-1].to_dict()
    return {k: (v if v != "" else None) for k, v in row.items()}

def list_risks(project_codes: list[str], include_resolved: bool = False) -> list[dict]:
    df = _read_all()
    if df.empty:
        return []
    rows = df[df["project_code"].isin(project_codes)]
    if not include_resolved:
        rows = rows[rows["resolved"] != "true"]
    rows = rows.sort_values("created_at", ascending=False)
    out = rows.to_dict("records")
    for r in out:
        r["resolved"] = r["resolved"] == "true"
        for k in ("risk_type", "mitigation_steps", "resolved_at"):
            if r.get(k) == "":
                r[k] = None
    return out
