"""Alternate real data source for the JIN Data Warehouse tables: an uploaded
Excel export instead of a live SQL connection -- for exactly the situation
the Resource Manager is in right now (no JMAN VPN on this machine, so the
live connection in jdwh_connection_service.py can't reach the server).

Feeds the SAME column-mapping (jdwh_table_mapping.map_all_tables) and the
SAME write/backup/reload step (jdwh_connection_service.write_mapped_tables)
as the live SQL path -- the only thing that differs is where the 5 real
tables' raw rows come from. This means whichever source is used, the result
is identical: this app's local employees/projects/allocations/timesheets/WSR
data gets replaced with the mapped real data.

Two upload shapes, both real (not a live query, so no MFA/VPN needed):
  - One workbook with all 6 tables as separate sheets, named exactly like
    the real tables (employee, designation_history, project,
    project_allocation, timesheet, weekly_status_report) -- e.g. the same
    shape as the jin_uat_data.ods sample the data-warehouse team already
    shared.
  - 6 separate files, one table per file (its first sheet, if it's a
    workbook with more than one).
designation_history is accepted in both shapes for consistency with the rest
of the JDWH UI (Connect's table discovery lists all 6), but genuinely unused
-- see jdwh_table_mapping.py's module docstring for why.

project_rolebased_user is a newer, optional 7th table/sheet -- when present,
it's combined with project_allocation's own rows rather than replacing them
(see jdwh_table_mapping.map_rolebased_user_table for the evidence on why
this table exists: project_allocation stopped reflecting real current
staffing on non-BAU work).
"""
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from app.services.jdwh_connection_service import write_mapped_tables
from app.services.jdwh_table_mapping import map_all_tables

REQUIRED_TABLES = ["employee", "project", "project_allocation", "timesheet", "weekly_status_report"]
# project_rolebased_user is optional -- a newer table that, when present, is
# combined with project_allocation's own rows (see
# jdwh_table_mapping.map_rolebased_user_table for why: project_allocation
# itself stopped reflecting real current staffing on non-BAU work).
OPTIONAL_TABLES = ["designation_history", "project_rolebased_user"]
ALL_TABLES = REQUIRED_TABLES + OPTIONAL_TABLES


class JdwhUploadError(Exception):
    pass


def _find_soffice() -> str | None:
    path = shutil.which("soffice")
    if path:
        return path
    for candidate in (r"C:\Program Files\LibreOffice\program\soffice.exe",):
        if Path(candidate).exists():
            return candidate
    return None


def _read_excel_file(content: bytes, filename: str) -> pd.ExcelFile:
    """Real .xlsx/.xls files read directly. A real .ods is ALWAYS converted
    via LibreOffice first rather than read directly through pandas' odfpy
    engine -- confirmed (twice, against the data team's own real UAT sample)
    that odfpy's own cell reader crashes on certain malformed datetime-shaped
    values *while parsing a sheet's rows*, not while opening the file, so a
    try/read-then-fallback approach doesn't actually catch it in time. Same
    LibreOffice tool already used elsewhere in this app for pixel-perfect SOW
    previews."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "ods":
        soffice = _find_soffice()
        if not soffice:
            raise JdwhUploadError(f"Could not read '{filename}': LibreOffice (needed to convert .ods files) isn't installed.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_path = Path(tmp_dir) / filename
            src_path.write_bytes(content)
            result = subprocess.run(
                [soffice, "--headless", "--norestore", "--convert-to", "xlsx", "--outdir", tmp_dir, str(src_path)],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                raise JdwhUploadError(f"Could not convert '{filename}' to a readable format: {result.stderr.decode(errors='replace')}")
            converted_path = src_path.with_suffix(".xlsx")
            # Read fully into memory before the temp dir is cleaned up (pandas
            # would otherwise still be lazily holding the file open past this
            # `with` block's exit).
            return pd.ExcelFile(io.BytesIO(converted_path.read_bytes()))
    try:
        return pd.ExcelFile(io.BytesIO(content))
    except Exception as exc:
        raise JdwhUploadError(f"Could not read '{filename}' as an Excel file: {exc}") from exc


def preview_workbook(content: bytes, filename: str) -> dict:
    """Confirms the file was actually read and shows what's in it -- sheet
    names + row/column counts only, never real cell values -- so the RM gets
    real feedback the moment a file is picked, before ever reaching the
    destructive Load step."""
    xl = _read_excel_file(content, filename)
    sheets = []
    for name in xl.sheet_names:
        if name == "Schema":
            continue  # metadata sheet, not one of the 6 real tables
        try:
            df = xl.parse(name, dtype=str)
            row_count, column_count = len(df), len(df.columns)
        except Exception:
            row_count, column_count = None, None
        sheets.append({
            "sheet_name": name,
            "row_count": row_count,
            "column_count": column_count,
            "is_required_table": name in REQUIRED_TABLES,
            "is_optional_table": name in OPTIONAL_TABLES,
        })
    missing = [t for t in REQUIRED_TABLES if t not in xl.sheet_names]
    return {"sheets": sheets, "missing_required": missing}


def preview_file(content: bytes, filename: str) -> dict:
    """Same idea as preview_workbook, for one single-table file (the "6
    separate files" upload shape) -- its first sheet only."""
    xl = _read_excel_file(content, filename)
    if not xl.sheet_names:
        return {"sheet_name": None, "row_count": 0, "column_count": 0}
    name = xl.sheet_names[0]
    df = xl.parse(name, dtype=str)
    return {"sheet_name": name, "row_count": len(df), "column_count": len(df.columns)}


def load_tables_from_workbook(content: bytes, filename: str) -> dict:
    """One workbook, one sheet per table (see module docstring for the
    expected sheet names)."""
    xl = _read_excel_file(content, filename)
    missing = [t for t in REQUIRED_TABLES if t not in xl.sheet_names]
    if missing:
        raise JdwhUploadError(
            f"This workbook is missing required sheet(s): {', '.join(missing)}. "
            f"Found sheets: {', '.join(xl.sheet_names)}. Expected one sheet per table, named exactly: "
            f"{', '.join(REQUIRED_TABLES)}."
        )
    raw = {t: xl.parse(t, dtype=str) for t in REQUIRED_TABLES}
    if "project_rolebased_user" in xl.sheet_names:
        raw["project_rolebased_user"] = xl.parse("project_rolebased_user", dtype=str)
    mapped = map_all_tables(raw)
    return write_mapped_tables(mapped)


def load_tables_from_files(files: dict[str, tuple[bytes, str]]) -> dict:
    """6 (or 7, with the optional project_rolebased_user table) separate
    files, keyed by table name -- each file's first sheet is used. `files`
    maps table name -> (content, filename); only REQUIRED_TABLES keys need
    to be present (designation_history is accepted but ignored;
    project_rolebased_user is accepted and used when present)."""
    missing = [t for t in REQUIRED_TABLES if t not in files]
    if missing:
        raise JdwhUploadError(f"Missing file(s) for required table(s): {', '.join(missing)}.")

    raw = {}
    for table in REQUIRED_TABLES + (["project_rolebased_user"] if "project_rolebased_user" in files else []):
        content, filename = files[table]
        xl = _read_excel_file(content, filename)
        if not xl.sheet_names:
            raise JdwhUploadError(f"'{filename}' (for {table}) has no sheets.")
        raw[table] = xl.parse(xl.sheet_names[0], dtype=str)

    mapped = map_all_tables(raw)
    return write_mapped_tables(mapped)
