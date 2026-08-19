"""This week's "Top Projects -- End to End Review" for a cluster (the JQA
deck's spotlight table: Project | DevOps Visibility | Milestone Visibility |
Comments | Action Plan). The deck's Score Card column (a numeric rating like
"DevOps - 4.5/5") was dropped -- it mostly restated the WSR badge/Comments
already shown, so it wasn't worth its own manual field.

Defaults to the top 5 projects in the cluster by real risk_score (same
score already driving the main Health table's default sort) -- the real deck
doesn't pick these by a visible rule, but showing nothing until someone
manually curates it isn't a useful default either, and "highest real risk
first" is the closest honest proxy: the projects most likely to actually
need this week's governance attention. Fully overridable per project per
week: an explicit "excluded" mark drops an auto-picked project out, an
explicit "included" mark (via Add Project) pulls in one that isn't in the
real top 5. Action Plan (the only genuinely manual field -- see
governance_service.compute_delivery_signals for why DevOps Visibility/
Milestone Visibility/Comments aren't stored here at all) can be attached to
either kind of row.

Composite-keyed on (project_code, week_start_date) so each week's overrides/
notes stand on their own and past weeks are preserved automatically, even
though this first version only ever reads/writes the current week."""
import pandas as pd

from app.core.config import APP_STATE_DIR

SPOTLIGHT_CSV = APP_STATE_DIR / "project_governance_spotlight.csv"
_KEY_FIELDS = ["project_code", "week_start_date"]
_NOTE_FIELDS = ["action_plan"]
_FIELDS = _KEY_FIELDS + ["excluded"] + _NOTE_FIELDS

DEFAULT_SPOTLIGHT_SIZE = 5

def _read_all() -> pd.DataFrame:
    if not SPOTLIGHT_CSV.exists():
        return pd.DataFrame(columns=_FIELDS)
    return pd.read_csv(SPOTLIGHT_CSV, dtype=str).fillna("")

def _upsert(project_code: str, week_start_date: str, excluded: bool, note_fields: dict | None = None) -> dict:
    df = _read_all()
    existing = df[(df["project_code"] == project_code) & (df["week_start_date"] == week_start_date)]
    row = {"project_code": project_code, "week_start_date": week_start_date, "excluded": str(excluded)}
    prior_notes = existing.iloc[-1].to_dict() if not existing.empty else {}
    for k in _NOTE_FIELDS:
        if note_fields is not None and k in note_fields:
            row[k] = note_fields.get(k) or ""
        else:
            row[k] = prior_notes.get(k, "")
    df = df[~((df["project_code"] == project_code) & (df["week_start_date"] == week_start_date))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(SPOTLIGHT_CSV, index=False)
    return row

def add_or_update_spotlight(project_code: str, week_start_date: str, fields: dict) -> dict:
    """Explicitly include this project this week (overriding an auto-
    exclusion if one existed) and/or save its Action Plan."""
    return _upsert(project_code, week_start_date, excluded=False, note_fields=fields)

def remove_from_spotlight(project_code: str, week_start_date: str) -> None:
    """Explicitly exclude this project this week -- for an auto-picked (top
    risk_score) project this is the only way to drop it, since there's no
    stored "add" row to delete; for a manually-added one it simply reverses
    that addition. Notes are preserved in case it's re-included later."""
    _upsert(project_code, week_start_date, excluded=True)

def list_manual_overrides(project_codes: list[str], week_start_date: str) -> dict[str, dict]:
    """{project_code: {"excluded": bool, "action_plan": str|None}} for every
    project with an explicit action or note this week."""
    df = _read_all()
    if df.empty:
        return {}
    rows = df[(df["project_code"].isin(project_codes)) & (df["week_start_date"] == week_start_date)]
    result: dict[str, dict] = {}
    for _, r in rows.iterrows():
        result[r["project_code"]] = {
            "excluded": r.get("excluded") == "True",
            "action_plan": r.get("action_plan") or None,
        }
    return result
