import re
import threading
from functools import lru_cache

import duckdb
import pandas as pd

from app.core.config import BACKEND_ROOT, DUCKDB_PATH, PIPELINE_XLSX, TRANSFORMED_DIR

_DATE_COLUMNS = {
    "projects": ["project_start_date", "project_end_date", "extended_end_date"],
    "allocations": ["allocated_start_date", "allocated_end_date", "extended_end_date", "extended_start_date"],
    "leaves": ["leave_start_date", "leave_end_date"],
    "timesheets": ["date", "created_at", "updated_at"],
    "wsr_reports": ["week_start_date", "week_end_date"],
    "hr_feedback": ["feedback_date"],
}

_EXPLICIT_FORMAT_DATE_COLUMNS = {
    "employees": (["date_of_join", "date_of_resignation"], "%d-%m-%Y"),
    "weekly_pulse": (["week_start_date", "week_end_date", "submitted_on", "created_at", "updated_at", "data_loaded_at"], "%d-%m-%Y"),
}

_CSV_TABLES = {
    "employees": "01_Employee_Details_clean.csv",
    "projects": "02_Project_Details_clean.csv",
    "allocations": "03_Project_Allocation_clean.csv",
    "timesheets": "04_Timesheet_Details_clean.csv",
    "skills": "05_Skill_Details_clean.csv",
    "competencies": "06_Competency_Details_clean.csv",
    "wsr_reports": "08_WSR_Report_clean.csv",
    "leaves": "09_Leave_Details_synthetic.csv",
    # Hackathon team didn't provide real weekly-pulse survey data -- generated
    # synthetic set (see weekly_pulse_generator.py) so this feature can exist.
    "weekly_pulse": "10_Weekly_Pulse_dummy.csv",
    # Hackathon team didn't provide real HR feedback records either -- generated
    # synthetic set grounded in real (employee, project) allocation pairs (see
    # app/scripts/generate_hr_feedback_data.py) purely for manual RM review/proof,
    # never wired into recommendation scoring.
    "hr_feedback": "11_HR_Feedback_dummy.csv",
}

_PIPELINE_SHEETS = {
    "Forecast": "pipeline_forecast",
    "Skillset": "pipeline_skillset",
    "Hierarchy": "pipeline_hierarchy",
    "6 Months Revenue": "pipeline_revenue",
}

# Loaded from the backend root (not data/Transformed) since it's a curated
# reference file, not a raw pipeline/HR export -- a real per-CoE
# required-skills list used as a fallback source when a pipeline deal has no
# skillset text of its own (see app/engines/pipeline_skill_inference.py).
_ROOT_CSV_TABLES = {
    "coe_skills_mapping": "COE_Skills_Mapping.csv",
}

_PIPELINE_FORECAST_FFILL_COLUMNS = [
    "request_received",
    "original_requested_start_date",
    "request_type",
    "client_priority",
    "client",
    "em",
    "start_date_confirmed",
    "number_of_weeks",
    "deal_stage_hubspot",
    "solution",
    "sow_signed",
]

def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    # Source CSVs are occasionally regenerated in a fixed-width-padded style
    # (e.g. allocations' "employee_id" arriving as " EMP233     " instead of
    # "EMP1"), which silently breaks every join against a clean id column in
    # another table. Strip all string cells so ids/status codes compare equal
    # regardless of padding in the raw file.
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df

def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def clean(col: str) -> str:
        col = col.strip().lower()
        col = re.sub(r"[^a-z0-9]+", "_", col)
        return col.strip("_")

    df = df.copy()
    cleaned = [clean(c) for c in df.columns]
    seen: dict[str, int] = {}
    final = []
    for i, name in enumerate(cleaned):
        name = name or f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        final.append(name)
    df.columns = final
    return df

# A write (create_project/create_allocation/etc.) calls reload(), which clears
# this cache so the next get_connection() re-reads every CSV from disk into a
# fresh in-memory duckdb connection. Without the lock below, a request
# arriving concurrently with that rebuild could see the connection's tables
# mid CREATE-OR-REPLACE (duckdb doesn't guarantee that's safe to query from a
# second thread while it's happening) and 500 -- rare, but real, and much more
# likely right after a write that's immediately followed by a burst of reads
# (e.g. every wizard step's queries turning "enabled" the instant a project is
# created). The lock just serializes "rebuild the connection" against "hand
# out the connection" -- reads block briefly during a reload instead of racing it.
_reload_lock = threading.Lock()

@lru_cache(maxsize=1)
def _build_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    _load_all(con)
    return con

def get_connection() -> duckdb.DuckDBPyConnection:
    with _reload_lock:
        return _build_connection()

def _load_all(con: duckdb.DuckDBPyConnection) -> None:
    for table, filename in _CSV_TABLES.items():
        path = TRANSFORMED_DIR / filename
        # Don't pass parse_dates to read_csv -- it needs an exact header-name match,
        # and source CSVs occasionally carry stray whitespace padding in the header
        # row (e.g. "allocated_start_date" stored as " allocated_start_date"), which
        # makes parse_dates raise before _sanitize_columns ever gets a chance to
        # clean it up. Sanitize first, then parse dates by the clean column name --
        # robust regardless of whitespace/casing in the raw file.
        df = pd.read_csv(path, low_memory=False)
        df = _sanitize_columns(df)
        df = _strip_string_values(df)
        for col in _DATE_COLUMNS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if table in _EXPLICIT_FORMAT_DATE_COLUMNS:
            cols, fmt = _EXPLICIT_FORMAT_DATE_COLUMNS[table]
            for col in cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_tmp")
        con.unregister("df_tmp")

    sheets = pd.read_excel(PIPELINE_XLSX, sheet_name=None)
    for sheet_name, table in _PIPELINE_SHEETS.items():
        df = _sanitize_columns(sheets[sheet_name])
        df = _strip_string_values(df)
        if table == "pipeline_forecast":
            df = df.rename(columns={"col_15": "requested_pct"})
            df["original_requested_start_date"] = pd.to_datetime(df["original_requested_start_date"], errors="coerce")
            df["deal_id"] = df["client"].notna().cumsum()
            df[_PIPELINE_FORECAST_FFILL_COLUMNS] = df.groupby("deal_id")[_PIPELINE_FORECAST_FFILL_COLUMNS].ffill()
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_tmp")
        con.unregister("df_tmp")

    for table, filename in _ROOT_CSV_TABLES.items():
        path = BACKEND_ROOT / filename
        df = pd.read_csv(path)
        df = _sanitize_columns(df)
        df = _strip_string_values(df)
        df = df.dropna(subset=["coe"])  # source file has a trailing blank line
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_tmp")
        con.unregister("df_tmp")

def get_cursor() -> duckdb.DuckDBPyConnection:
    return get_connection().cursor()

def _all_table_names() -> list[str]:
    return list(_CSV_TABLES.keys()) + list(_PIPELINE_SHEETS.values()) + list(_ROOT_CSV_TABLES.keys())

def table_counts() -> dict[str, int]:
    return {t: get_cursor().execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _all_table_names()}

@lru_cache(maxsize=1)
def get_schema_description() -> str:
    """Table(column type, ...) text block for every real table -- the schema
    reference an LLM needs to write a valid ad-hoc SQL query (see
    copilot_service.py's query_database tool). Cached like the connection
    itself; cleared in reload() since a reload can, in principle, change
    column shapes."""
    con = get_cursor()
    lines = []
    for table in _all_table_names():
        try:
            cols = con.execute(f"DESCRIBE {table}").fetchall()
        except duckdb.Error:
            continue
        col_list = ", ".join(f"{c[0]} {c[1]}" for c in cols)
        lines.append(f"{table}({col_list})")
    return "\n".join(lines)

class ReadOnlyQueryError(Exception):
    """Raised when an LLM-generated SQL string is rejected before execution,
    or when DuckDB itself errors while running it. The message is written to
    be genuinely useful as a tool-result: it goes straight back to the LLM so
    it can self-correct on the next attempt (see MAX_SQL_ATTEMPTS in
    copilot_service.py), so it always names exactly what was wrong."""

_LEADING_KEYWORD_RE = re.compile(r"^[a-zA-Z]+")
_ALLOWED_LEADING_KEYWORDS = {"select", "with"}

def _validate_readonly_sql(sql: str) -> str:
    cleaned = (sql or "").strip()
    # A single optional trailing semicolon is fine (models often add one) --
    # anything else semicolon-related is a multi-statement smuggling attempt,
    # caught below via extract_statements.
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        raise ReadOnlyQueryError("Empty SQL query -- write a real SELECT statement.")

    match = _LEADING_KEYWORD_RE.match(cleaned)
    leading = match.group(0).lower() if match else ""
    if leading not in _ALLOWED_LEADING_KEYWORDS:
        raise ReadOnlyQueryError(
            f"Rejected: only SELECT or WITH statements are allowed for read-only queries. "
            f"This statement starts with '{leading or cleaned[:20]}', which this tool never "
            f"permits (no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE or any other write/DDL). "
            f"Rewrite it as a single SELECT/WITH query."
        )

    # Reject anything that isn't exactly one statement -- prevents smuggling a
    # second, mutating statement after a semicolon. duckdb's own parser is the
    # source of truth here (far more reliable than a keyword blocklist, which
    # would also risk false-positives on legitimate column/string content).
    con = get_connection()
    try:
        statements = con.extract_statements(cleaned)
    except duckdb.Error as exc:
        raise ReadOnlyQueryError(f"SQL parse error: {exc}. Fix the syntax and try again.") from exc
    if len(statements) != 1:
        raise ReadOnlyQueryError(
            "Rejected: only a single SQL statement is allowed per call -- remove any "
            "semicolon-separated extra statements and submit one SELECT/WITH query at a time."
        )
    return cleaned

def run_readonly_query(sql: str, max_rows: int = 200) -> dict:
    """Execute an LLM-generated, validated read-only SQL query against the
    real shared DuckDB connection (same data every other tool/adapter reads
    from) and return a JSON-safe {columns, rows, total_row_count, truncated}
    payload. Raises ReadOnlyQueryError -- with a message meant to be read by
    the LLM itself -- on anything rejected before execution or any DuckDB
    error during execution (bad column/table name, syntax error, etc)."""
    cleaned = _validate_readonly_sql(sql)
    cur = get_cursor()
    try:
        result_df = cur.execute(cleaned).fetchdf()
    except duckdb.Error as exc:
        raise ReadOnlyQueryError(f"SQL error: {exc}. Fix the query (check table/column names against the schema) and try again.") from exc

    total_row_count = len(result_df)
    truncated_df = result_df.head(max_rows)
    columns = [str(c) for c in truncated_df.columns]
    rows = []
    for record in truncated_df.itertuples(index=False, name=None):
        row = []
        for v in record:
            if isinstance(v, (list, dict, tuple, set)):
                row.append(v)
            elif pd.isna(v):
                row.append(None)
            elif hasattr(v, "isoformat"):
                row.append(v.isoformat())
            else:
                row.append(v)
        rows.append(row)

    return {
        "columns": columns,
        "rows": rows,
        "total_row_count": total_row_count,
        "truncated": total_row_count > max_rows,
    }

def reload() -> None:
    with _reload_lock:
        _build_connection.cache_clear()
        get_schema_description.cache_clear()
        from app.core.adapter import _cached_query

        _cached_query.cache_clear()
