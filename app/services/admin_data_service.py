import io
import shutil
from datetime import datetime

import pandas as pd

from app.core import adapter, db
from app.core.config import APP_STATE_DIR, PIPELINE_XLSX, TRANSFORMED_DIR
from app.services.skill_matrix_transform import (
    is_wide_survey_export,
    read_csv_robust,
    transform_wide_skill_matrix,
)

# Where a replaced file's PREVIOUS version is kept before being overwritten --
# an upload here replaces a real data file in place, so a bad upload (wrong
# file, corrupted export) needs a way back without re-pulling from the
# original source.
UPLOAD_BACKUPS_DIR = APP_STATE_DIR / "upload_backups"
UPLOAD_BACKUPS_DIR.mkdir(exist_ok=True)

_PIPELINE_REQUIRED_SHEETS = ["Forecast", "Skillset", "Hierarchy", "6 Months Revenue"]

# These three are the real datasets the Resourcing/RMG team maintains OUTSIDE
# the JIN data warehouse (skill matrix and competency live in separately
# managed spreadsheets; pipeline data comes from the HubSpot-sourced workbook)
# -- see app/core/db.py's _CSV_TABLES/_PIPELINE_SHEETS for the exact same
# files this app already loads from disk. Employees/projects/allocations/
# timesheets/WSR are NOT here -- those come from the JIN warehouse itself
# (get_adapter()/JinApiAdapter), not a manual upload.
DATASET_REGISTRY = {
    "skill_matrix": {
        "label": "Skill Matrix",
        "description": "Per-employee skill ratings -- managed separately from the JIN data warehouse.",
        "file_path": TRANSFORMED_DIR / "05_Skill_Details_clean.csv",
        "file_type": "csv",
        "tables": [{"table": "skills", "sheet_name": None, "label": "Skill Matrix"}],
        "required_columns": ["employee_id"],
    },
    "competency": {
        "label": "Competency",
        "description": "Per-employee competency scores -- managed separately from the JIN data warehouse.",
        "file_path": TRANSFORMED_DIR / "06_Competency_Details_clean.csv",
        "file_type": "csv",
        "tables": [{"table": "competencies", "sheet_name": None, "label": "Competency"}],
        "required_columns": ["employee_id"],
    },
    "pipeline_data": {
        "label": "Pipeline Data",
        "description": "HubSpot-sourced pipeline workbook (Forecast, Skillset, Hierarchy, 6 Months Revenue sheets).",
        "file_path": PIPELINE_XLSX,
        "file_type": "xlsx",
        "tables": [
            {"table": "pipeline_forecast", "sheet_name": "Forecast", "label": "Forecast"},
            {"table": "pipeline_skillset", "sheet_name": "Skillset", "label": "Skillset"},
            {"table": "pipeline_hierarchy", "sheet_name": "Hierarchy", "label": "Hierarchy"},
            {"table": "pipeline_revenue", "sheet_name": "6 Months Revenue", "label": "6 Months Revenue"},
        ],
        "required_columns": [],
    },
}


class DatasetValidationError(Exception):
    pass


def _row_count(table: str) -> int | None:
    try:
        return db.table_counts().get(table)
    except Exception:
        return None


def list_data_sources() -> list[dict]:
    out = []
    for key, meta in DATASET_REGISTRY.items():
        path = meta["file_path"]
        exists = path.exists()
        out.append({
            "key": key,
            "label": meta["label"],
            "description": meta["description"],
            "file_type": meta["file_type"],
            "current_filename": path.name if exists else None,
            "row_count": _row_count(meta["tables"][0]["table"]),
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
            "source": "Manual upload (outside JIN warehouse)",
        })
    return out


def get_connection_status() -> dict:
    """Honest status, not a simulated connection -- see app/core/adapter.py's
    JinApiAdapter, which is a real production-contract stub that raises until
    real credentials/base URL are wired in. Switching to JIN mode before that
    happens would break every page, on purpose -- there is no fake-success
    state here."""
    cfg = adapter.get_connection_config()
    jin_configured = bool(cfg["base_url"] and cfg["has_api_key"])
    jin_connected = cfg["mode"] == "jin" and jin_configured
    return {
        "mode": cfg["mode"],
        "jin_base_url": cfg["base_url"],
        "jin_has_api_key": cfg["has_api_key"],
        "jin_configured": jin_configured,
        "jin_connected": jin_connected,
        "message": (
            "Connected to the JIN Data Warehouse."
            if jin_connected
            else "Not connected -- awaiting JIN Data Warehouse credentials/access. Running on local files (uploaded above + transformed exports)."
        ),
    }


def save_connection(mode: str, base_url: str | None, api_key: str | None) -> dict:
    adapter.save_connection_config(mode, base_url, api_key)
    return get_connection_status()


def test_connection() -> dict:
    return adapter.test_connection()


def get_dataset_preview(key: str, max_rows: int = 20) -> dict:
    """The REAL, currently-loaded data for this dataset -- straight from the
    same DuckDB tables every other page in this app reads from (via
    db.run_readonly_query, which already handles JSON-safe date/NaN
    conversion), not a re-read of the raw file. This is "what we have right
    now", so Siva/the RM can see exactly what's live before deciding whether
    a replacement is even needed."""
    meta = DATASET_REGISTRY.get(key)
    if meta is None:
        raise DatasetValidationError(f"Unknown dataset '{key}'.")
    tables = []
    for t in meta["tables"]:
        try:
            result = db.run_readonly_query(f"SELECT * FROM {t['table']}", max_rows=max_rows)
        except db.ReadOnlyQueryError as exc:
            result = {"columns": [], "rows": [], "total_row_count": 0, "truncated": False, "error": str(exc)}
        tables.append({"label": t["label"], **result})
    return {"key": key, "label": meta["label"], "tables": tables}


def _json_safe(v):
    if isinstance(v, (list, dict)):
        return v
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _sample_rows(df: pd.DataFrame, n: int = 3) -> list[dict]:
    return [{str(c): _json_safe(row[c]) for c in df.columns} for _, row in df.head(n).iterrows()]


def get_dataset_schema(key: str) -> dict:
    """The expected format for a REPLACEMENT upload -- read live from the file
    currently on disk (the file this app is actually running on right now is,
    by definition, an accepted format), shown as column names + a few real
    sample rows so Siva/the RM can match column names, casing, and date
    format exactly rather than guessing."""
    meta = DATASET_REGISTRY.get(key)
    if meta is None:
        raise DatasetValidationError(f"Unknown dataset '{key}'.")
    if not meta["file_path"].exists():
        raise DatasetValidationError(f"No current file exists yet for '{key}' to derive an expected format from.")

    sheets = []
    if meta["file_type"] == "csv":
        df = pd.read_csv(meta["file_path"], nrows=5)
        sheets.append({"sheet_name": None, "columns": [str(c) for c in df.columns], "sample_rows": _sample_rows(df)})
    else:
        xl = pd.ExcelFile(meta["file_path"])
        for t in meta["tables"]:
            sheet_name = t["sheet_name"]
            if sheet_name in xl.sheet_names:
                sdf = xl.parse(sheet_name, nrows=5)
                sheets.append({"sheet_name": sheet_name, "columns": [str(c) for c in sdf.columns], "sample_rows": _sample_rows(sdf)})
    return {"key": key, "label": meta["label"], "file_type": meta["file_type"], "sheets": sheets}


def _build_employee_lookup() -> dict[str, dict]:
    """employee_id (and, once a real email/name field exists in the employee
    master data, email) -> {employee_id, job_name, department_name}, for
    resolving a Skills Matrix survey respondent to a real employee. Keyed by
    employee_id today since that's the only real identifier this app's
    employee master data carries -- see skill_matrix_transform.py's module
    docstring for why email/name can't be matched yet. Always the LOCAL
    employee file regardless of the active connection mode -- this manual
    upload feature exists specifically for data that lives outside JIN, so
    it shouldn't depend on whatever get_adapter() currently resolves to."""
    employees = adapter.LocalAdapter().get_employees()
    lookup: dict[str, dict] = {}
    for _, r in employees.iterrows():
        emp = {"employee_id": r["employee_id"], "job_name": r.get("job_name"), "department_name": r.get("department_name")}
        lookup[str(r["employee_id"]).strip().lower()] = emp
        email = r.get("email") if "email" in employees.columns else None
        if email and pd.notna(email):
            lookup[str(email).strip().lower()] = emp
    return lookup


def _handle_wide_skill_matrix_upload(df: pd.DataFrame) -> dict:
    lookup = _build_employee_lookup()
    result = transform_wide_skill_matrix(df, lookup)

    if not result["rows"]:
        sample = ", ".join(result["unmatched_emails"][:5])
        raise DatasetValidationError(
            f"This looks like the real Skills Matrix survey export -- it identifies "
            f"{result['respondent_count']} respondent(s) by email, but none could be matched to a "
            f"real employee_id: this app's employee master data has no email or name field to match "
            f"against, only employee_id. Fastest fix: add an 'Employee ID' column to this export "
            f"(e.g. EMP123, cross-referenced by HR/Siva against the email) and re-upload -- this file "
            f"will ingest automatically once that column exists. Sample respondent email(s) from this "
            f"file: {sample}."
        )

    long_df = pd.DataFrame(result["rows"])
    path = DATASET_REGISTRY["skill_matrix"]["file_path"]
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, UPLOAD_BACKUPS_DIR / f"{path.stem}.{stamp}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(path, index=False)
    db.reload()

    updated = next(d for d in list_data_sources() if d["key"] == "skill_matrix")
    updated["unmatched_respondent_count"] = len(result["unmatched_emails"])
    updated["matched_respondent_count"] = result["matched_count"]
    return updated


def _validate_upload(meta: dict, content: bytes) -> None:
    if meta["file_type"] == "csv":
        try:
            df = read_csv_robust(io.BytesIO(content), nrows=5)
        except Exception as exc:
            raise DatasetValidationError(f"Could not parse this file as a CSV: {exc}") from exc
        if len(df.columns) == 0:
            raise DatasetValidationError("This file has no columns -- is it the right file?")
        cols_lower = {c.strip().lower() for c in df.columns}
        missing = [c for c in meta["required_columns"] if c.lower() not in cols_lower]
        if missing:
            raise DatasetValidationError(
                f"This file is missing required column(s): {', '.join(missing)}. Found: {', '.join(df.columns)}."
            )
    else:
        try:
            xl = pd.ExcelFile(io.BytesIO(content))
        except Exception as exc:
            raise DatasetValidationError(f"Could not parse this file as an Excel workbook: {exc}") from exc
        missing_sheets = [s for s in _PIPELINE_REQUIRED_SHEETS if s not in xl.sheet_names]
        if missing_sheets:
            raise DatasetValidationError(
                f"This workbook is missing required sheet(s): {', '.join(missing_sheets)}. "
                f"Found: {', '.join(xl.sheet_names)}."
            )


def upload_dataset(key: str, filename: str, content: bytes) -> dict:
    meta = DATASET_REGISTRY.get(key)
    if meta is None:
        raise DatasetValidationError(f"Unknown dataset '{key}'.")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    expected_ext = meta["file_type"]
    if ext != expected_ext:
        raise DatasetValidationError(f"{meta['label']} expects a .{expected_ext} file, got .{ext or '?'}.")

    if key == "skill_matrix":
        try:
            probe_df = read_csv_robust(io.BytesIO(content))
        except Exception as exc:
            raise DatasetValidationError(f"Could not parse this file as a CSV: {exc}") from exc
        if is_wide_survey_export(probe_df.columns):
            return _handle_wide_skill_matrix_upload(probe_df)

    _validate_upload(meta, content)

    path = meta["file_path"]
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, UPLOAD_BACKUPS_DIR / f"{path.stem}.{stamp}{path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    db.reload()

    return next(d for d in list_data_sources() if d["key"] == key)
