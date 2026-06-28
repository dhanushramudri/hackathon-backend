import re
from functools import lru_cache

import duckdb
import pandas as pd

from app.core.config import DUCKDB_PATH, PIPELINE_XLSX, TRANSFORMED_DIR

_DATE_COLUMNS = {
    "projects": ["project_start_date", "project_end_date"],
    "allocations": ["allocated_start_date", "allocated_end_date"],
    "leaves": ["leave_start_date", "leave_end_date"],
    "timesheets": ["date", "created_at", "updated_at"],
    "wsr_reports": ["week_start_date", "week_end_date"],
}

_EXPLICIT_FORMAT_DATE_COLUMNS = {
    "employees": (["date_of_join", "date_of_resignation"], "%d-%m-%Y"),
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
}

_PIPELINE_SHEETS = {
    "Forecast": "pipeline_forecast",
    "Skillset": "pipeline_skillset",
    "Hierarchy": "pipeline_hierarchy",
    "6 Months Revenue": "pipeline_revenue",
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

@lru_cache(maxsize=1)
def get_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DUCKDB_PATH))
    _load_all(con)
    return con

def _load_all(con: duckdb.DuckDBPyConnection) -> None:
    for table, filename in _CSV_TABLES.items():
        path = TRANSFORMED_DIR / filename
        df = pd.read_csv(path, low_memory=False, parse_dates=_DATE_COLUMNS.get(table))
        if table in _EXPLICIT_FORMAT_DATE_COLUMNS:
            cols, fmt = _EXPLICIT_FORMAT_DATE_COLUMNS[table]
            for col in cols:
                df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
        df = _sanitize_columns(df)
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_tmp")
        con.unregister("df_tmp")

    sheets = pd.read_excel(PIPELINE_XLSX, sheet_name=None)
    for sheet_name, table in _PIPELINE_SHEETS.items():
        df = _sanitize_columns(sheets[sheet_name])
        if table == "pipeline_forecast":
            df = df.rename(columns={"col_15": "requested_pct"})
            df["original_requested_start_date"] = pd.to_datetime(df["original_requested_start_date"], errors="coerce")
            df["deal_id"] = df["client"].notna().cumsum()
            df[_PIPELINE_FORECAST_FFILL_COLUMNS] = df.groupby("deal_id")[_PIPELINE_FORECAST_FFILL_COLUMNS].ffill()
        con.register("df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_tmp")
        con.unregister("df_tmp")

def get_cursor() -> duckdb.DuckDBPyConnection:
    return get_connection().cursor()

def table_counts() -> dict[str, int]:
    all_tables = list(_CSV_TABLES.keys()) + list(_PIPELINE_SHEETS.values())
    return {t: get_cursor().execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in all_tables}

def reload() -> None:
    get_connection.cache_clear()
    from app.core.adapter import _cached_query

    _cached_query.cache_clear()
