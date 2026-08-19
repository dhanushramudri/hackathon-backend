"""Real-world Skills Matrix survey ingestion.

The real Skills Matrix RMG will eventually upload isn't shaped like this
app's internal `skills` table -- it's a wide Microsoft-Forms-style export
(one row per respondent, one column per "Category.Skill", ratings 0-6,
identified by Email/Name) -- see the real sample file "Copy of Skills
Matrix_(Sheet1).csv". This module melts that real shape into the long
`employee_id, coe, coe_skill, skill, subskill, experience, score,
skill_source` format `app/core/db.py`/`scoring.py` already expect.

The one thing this module CANNOT paper over: this app's real employee
master data (`01_Employee_Details_clean.csv`) has no email or name field at
all -- only `employee_id`. Matching a survey respondent to a real
employee_id therefore requires either (a) an `Employee ID` column added to
the export itself, or (b) a real email/name field appearing in the employee
master data (expected once the JIN Data Warehouse connection is live -- see
app/core/adapter.py). Until one of those exists, respondents are correctly
reported as unmatched rather than silently guessed or dropped.
"""
import pandas as pd

EMAIL_COLUMN_CANDIDATES = ("email",)
ID_COLUMN_CANDIDATES = ("employee_id", "employee id", "emp id", "empid")

# Real header noise this export carries (trailing non-breaking spaces,
# free-text "please add details" follow-ups, and 3 trailing competency
# self-ratings that belong to Competency, not Skill Matrix) -- anything
# WITHOUT a "." is metadata, never a skill rating column.
_COMPETENCY_COLUMN_HINTS = ("effective communication", "effective problem solving", "project management")

# The real export uses a 0-6 self-rating scale; this app's scoring
# (scoring.py) is normalized against a 0-5 scale everywhere else. Rescaled
# proportionally rather than clipped, so a genuine 6 doesn't silently
# collapse to the same weight as a 5.
_SOURCE_MAX = 6.0
_TARGET_MAX = 5.0


def _find_column(columns, candidates: tuple[str, ...]) -> str | None:
    for c in columns:
        if str(c).strip().lower() in candidates:
            return c
    return None


def is_wide_survey_export(columns) -> bool:
    """True for the real MS-Forms-style export: real 'Category.Skill' rating
    columns, identified by either Email or an Employee ID column. Deliberately
    NOT conditioned on the ABSENCE of an employee_id column -- once someone
    adds one (the fix this module's own error message recommends), the file
    is still wide-shaped and must still route through this transform, not
    fall through to the long-format validator, which would just reject it
    for a completely different reason."""
    has_email = _find_column(columns, EMAIL_COLUMN_CANDIDATES) is not None
    has_employee_id = _find_column(columns, ID_COLUMN_CANDIDATES) is not None
    has_rating_columns = any("." in str(c) for c in columns)
    return (has_email or has_employee_id) and has_rating_columns


def read_csv_robust(content_or_path, **kwargs) -> pd.DataFrame:
    """Real exports from this org are inconsistently UTF-8 -- this exact real
    file is Windows-1252 (mojibake'd names like "Name and Surname" carry a
    stray byte that raises UnicodeDecodeError under a strict utf-8 read).
    Try utf-8 first (the common case), fall back to cp1252 rather than
    rejecting a real file outright over an encoding technicality."""
    try:
        return pd.read_csv(content_or_path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        if hasattr(content_or_path, "seek"):
            content_or_path.seek(0)
        return pd.read_csv(content_or_path, encoding="cp1252", **kwargs)


def transform_wide_skill_matrix(df: pd.DataFrame, employee_lookup: dict[str, dict]) -> dict:
    """employee_lookup: {lowercased email OR literal employee_id: {"employee_id", "job_name", "department_name"}}
    -- built by the caller from the real employees table (+ a real email map,
    once one exists) so this module stays a pure transform, not a data-access layer.

    Returns {rows, respondent_count, matched_count, unmatched_emails}. `rows`
    is only ever built from respondents that resolved to a real employee_id
    -- there is no partial/guessed attribution path."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    email_col = _find_column(df.columns, EMAIL_COLUMN_CANDIDATES)
    id_col = _find_column(df.columns, ID_COLUMN_CANDIDATES)
    if email_col is None and id_col is None:
        raise ValueError("Could not find an 'Employee ID' or 'Email' column to identify respondents.")

    completion_col = next((c for c in df.columns if c.strip().lower() == "completion time"), None)
    if email_col is not None and completion_col is not None:
        # Real respondents who filled the survey twice -- keep only their
        # latest real submission rather than double-counting or averaging.
        df[completion_col] = pd.to_datetime(df[completion_col], errors="coerce")
        df = df.sort_values(completion_col).drop_duplicates(subset=[email_col], keep="last")

    skill_cols = [
        c for c in df.columns
        if "." in c and not any(c.strip().lower().startswith(h) for h in _COMPETENCY_COLUMN_HINTS)
    ]
    if not skill_cols:
        raise ValueError("Could not find any 'Category.Skill' rating columns in this file.")

    rows: list[dict] = []
    unmatched_emails: list[str] = []
    matched_count = 0

    for _, r in df.iterrows():
        employee_id = None
        lookup_key = None
        if id_col is not None and pd.notna(r.get(id_col)) and str(r[id_col]).strip():
            lookup_key = str(r[id_col]).strip().lower()
        elif email_col is not None:
            lookup_key = str(r.get(email_col) or "").strip().lower()

        emp = employee_lookup.get(lookup_key) if lookup_key else None
        if emp is None:
            if email_col is not None:
                email_val = str(r.get(email_col) or "").strip()
                if email_val:
                    unmatched_emails.append(email_val)
            continue

        matched_count += 1
        employee_id = emp["employee_id"]
        for col in skill_cols:
            raw = r.get(col)
            if pd.isna(raw) or str(raw).strip() == "":
                continue
            try:
                raw_score = float(raw)
            except (TypeError, ValueError):
                continue
            category, _, skill_name = col.partition(".")
            score = round(max(0.0, min(_SOURCE_MAX, raw_score)) * (_TARGET_MAX / _SOURCE_MAX), 2)
            rows.append({
                "employee_id": employee_id,
                "Designation": emp.get("job_name"),
                "COE": emp.get("department_name"),
                "COE Skill": category.strip(),
                "Skill": skill_name.strip(),
                "SubSkill": skill_name.strip(),
                "Experience": None,
                "Score": score,
                "skill_source": "self_assessed_survey",
            })

    return {
        "rows": rows,
        "respondent_count": len(df),
        "matched_count": matched_count,
        "unmatched_emails": sorted(set(unmatched_emails)),
    }
