"""Column-mapping from the real JIN Data Warehouse's 6 provisioned tables
(core.employee, core.designation_history, core.project, core.project_allocation,
core.timesheet, core.weekly_status_report -- see jdwh_connection_service.py)
onto this app's existing local data model (the same shape db.py already
loads from data/Transformed/*.csv, so every downstream engine keeps working
unchanged regardless of which source populated it).

Built and validated against a real (partially masked) UAT sample the data
team provided -- see backend/jin_uat_data.ods -- not against production.
Three real data-quality issues were confirmed in that sample and are handled
defensively here (never crash, coerce to null/best-effort instead):

1. Mixed date formats within the same column ("2025-02-06 00:00:00" next to
   "7/22/2024" in the same employee.date_of_join column) -- _parse_date()
   uses pandas' flexible parser rather than a fixed format.
2. `date_of_resignation` uses a literal sentinel "12/31/2999" for "still
   employed" instead of NULL -- must be converted to a real null, not passed
   through as a fake future resignation date.
3. Several `datetime2(7)`-typed audit columns (employee.updated_at,
   project.project_start_date/end_date, valid_from/valid_to) came through
   the sample export with only a time-of-day fragment and no date at all
   (e.g. "00:52:39", "00:00:00") -- these are dropped as unreliable rather
   than mapped; nothing in this app's engines actually needs them, so the
   safest move is to not carry a corrupted date forward as if it were real.
   Whether this is specific to how the UAT sample was generated or a real
   production quirk is unconfirmed -- worth re-checking once real data flows
   through this path.

Every JDWH id column that's a UUID (employee.employee_id, employee.jin_employee_id,
manager_id, reporter_id, approver_id, project.project_id, etc.) gets resolved
to this app's own code-style identifiers (employee_id like "EMP1",
project_code like "GCC_005") via the lookup tables built below -- this app's
existing engines match on those codes everywhere, not on warehouse surrogate
keys. Employee/project rows this app has never seen before naturally get a
real, new employee_id/project_code (the warehouse's own employee_code/
project_code, passed through as-is) rather than invented ones.
"""
import re

import pandas as pd

RESIGNATION_SENTINEL_DATE = pd.Timestamp("2999-12-31")

# A bare "HH:MM:SS[.f]" string with no date part at all -- what the corrupted
# datetime2(7) columns came through as in the sample (see module docstring).
_BARE_TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}(\.\d+)?\s*$")


def _parse_date(series: pd.Series) -> pd.Series:
    """Flexible parse (handles mixed 'YYYY-MM-DD HH:MM:SS' / 'M/D/YYYY'
    strings in the same column) -- bad/unparseable values become NaT rather
    than raising, since a single corrupted row must never abort the whole
    table's load."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.where(parsed != RESIGNATION_SENTINEL_DATE)


def _parse_date_strict_only(series: pd.Series) -> pd.Series:
    """For the datetime2(7) audit columns confirmed corrupted in the sample
    (time-of-day only, no date, e.g. "00:52:39") -- a bare time string like
    that has no date to recover, but naively parsing it with
    pandas/dateutil silently fills in the date as TODAY (the day this code
    happens to run), which is far more dangerous than an obvious NaT: it
    looks like a real, current date instead of an obviously missing one.
    Detects that shape in the raw string BEFORE parsing and forces NaT."""
    is_bare_time = series.astype(str).str.match(_BARE_TIME_RE)
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.where(~is_bare_time)


def _clean_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"nan": None, "None": None, "": None})


def build_employee_code_lookup(employee_df: pd.DataFrame) -> dict[str, str]:
    """UUID -> this app's employee_id (JDWH's own `employee_code`, e.g.
    "JMD404"). Keyed by BOTH employee.employee_id and employee.jin_employee_id
    (lowercased) since different real tables reference the employee by
    whichever of those two UUID columns they happen to carry -- manager_id/
    reporter_id/approver_id look like employee.employee_id, while
    project_allocation/timesheet's jin_employee_id matches employee.jin_employee_id."""
    lookup: dict[str, str] = {}
    for _, row in employee_df.iterrows():
        code = row.get("employee_code")
        if not code or pd.isna(code):
            continue
        for uuid_col in ("employee_id", "jin_employee_id"):
            uuid_val = row.get(uuid_col)
            if uuid_val and not pd.isna(uuid_val):
                lookup[str(uuid_val).strip().lower()] = str(code).strip()
    return lookup


def build_project_code_lookup(project_df: pd.DataFrame) -> dict[str, str]:
    """UUID (project.project_id) -> this app's project_id (JDWH's own
    `project_code`, e.g. "GCC_005") -- every other real table references a
    project only by this project_id UUID, but this app's existing
    allocations/timesheets/WSR tables all key on project_code, not the raw
    warehouse UUID."""
    lookup: dict[str, str] = {}
    for _, row in project_df.iterrows():
        uuid_val = row.get("project_id")
        code = row.get("project_code")
        if uuid_val and code and not pd.isna(uuid_val) and not pd.isna(code):
            lookup[str(uuid_val).strip().lower()] = str(code).strip()
    return lookup


def _resolve(series: pd.Series, lookup: dict[str, str]) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(lookup)


# Confirmed genuine typos of an EXISTING canonical title in role_hierarchy.py
# (single-character edits, not a plausible distinct real role) -- found by
# diffing every real job_name value against the canonical seniority ladder.
# Deliberately NOT a broad fuzzy-match/auto-merge: most near-matches in the
# real data (e.g. "IT Manager" vs "Manager", "HR Intern" vs "Intern",
# "Solutions Manager" vs "Solutions Enabler") are genuinely different real
# titles, and auto-merging those would destroy real signal, not fix a typo.
# Left uncorrected, these fragment role_mix_engine's historical role-mix
# stats and the Staffing tab's "who's actually here" view into two separate,
# smaller buckets for what's really one role, understating that role's real
# prevalence/headcount and showing a confusing duplicate row in the UI.
_JOB_NAME_TYPO_FIXES: dict[str, str] = {
    "Solutions Consulant": "Solutions Consultant",
    "Senior Associate COnsultant": "Senior Associate Consultant",
}


def map_employee_table(df: pd.DataFrame) -> pd.DataFrame:
    """-> 01_Employee_Details_clean.csv shape (employee_id, location,
    date_of_join, date_of_resignation, job_name, department_name,
    manager_employee_id, account_status, is_active_version).

    core.employee is SCD Type 2 -- valid_from/valid_to/is_active_version
    columns confirm the same employee_code can legitimately appear as
    multiple historical rows (a role change, a location change, etc.), not
    just once. This app's employee table is a current-state master list (one
    row per employee_id), so every downstream table/lookup that keys on
    employee_id assumes it's unique -- confirmed the hard way: with a real,
    larger sample (1,165 raw rows), duplicate employee_code rows made
    map_timesheet_table's department_name lookup crash outright
    (`pandas.errors.InvalidIndexError: Reindexing only valid with uniquely
    valued Index objects`), since a non-unique index can't be used as a
    mapping. Deduplicated here to one row per employee_code -- the active
    version (is_active_version == 1) if there is one, else just the first
    row encountered -- before anything else touches this table."""
    employee_lookup = build_employee_code_lookup(df)
    out = pd.DataFrame({
        "employee_id": _clean_id(df["employee_code"]),
        # Deliberately the ONLY personal-identity field this app ever maps in
        # from JDWH (no email, no first/last name split) -- shown to real
        # Resource Managers using the app, but never something this assistant
        # reads/inspects while building or testing (values aren't printed to
        # logs or verified by content, only by column presence/coverage).
        "employee_full_name": df.get("employee_full_name"),
        "location": df.get("location"),
        "date_of_join": _parse_date(df["date_of_join"]),
        "date_of_resignation": _parse_date(df["date_of_resignation"]),
        "job_name": df["job_name"].replace(_JOB_NAME_TYPO_FIXES),
        "department_name": df.get("department_name"),
        "manager_employee_id": _resolve(df["manager_id"].astype(str), employee_lookup),
        # Missing (not explicitly 0) account_status defaults to active, not
        # inactive -- confirmed against real data: every row where this field
        # came through blank had NO resignation date recorded either, several
        # with 2025/2026 join dates and real job titles (Manager, Finance
        # Executive, etc.), meaning this field is just unpopulated for a
        # large share of the real roster, not a real "inactive" signal.
        # date_of_resignation is the authoritative departure signal
        # everywhere else in this app (see adapter.py's LocalAdapter,
        # `not_yet_departed`) -- defaulting a missing account_status to 0
        # would silently override that with a data gap instead.
        "account_status": pd.to_numeric(df.get("account_status"), errors="coerce").fillna(1).astype(int),
        "is_active_version": pd.to_numeric(df.get("is_active_version"), errors="coerce").fillna(1).astype(int),
    })
    out = out.dropna(subset=["employee_id"])
    out = out.sort_values("is_active_version", ascending=False)
    out = out.drop_duplicates(subset="employee_id", keep="first")
    # A handful of real rows are obvious placeholder/test data (employee_id
    # values like "XXX", "TEMP_991", "Temp001", "Test002" with job_name
    # literally "Test"/"Test IT") -- not a real person under any
    # region/office/entity, confirmed by employee_id prefix analysis showing
    # them as outliers outside the real JMD/JMG/JML/JMU/INT/EXT/TRN groups.
    # Dropped here so they never inflate headcount or pollute any downstream
    # matching, not just the Employees page's total.
    is_junk = out["employee_id"].str.contains(r"test|xxx|temp", case=False, na=False)
    return out[~is_junk]


def _parse_cluster_number(series: pd.Series) -> pd.Series:
    """"Cluster 3" -> 3 -- the real JDWH `cluster` column found in a fuller
    project-table export (260815_Tables Schema.xlsx) than the original sample
    this mapper was built against had. Confirmed real and per-project: 46 of
    89 currently-active projects carry a genuine value here (the rest are
    blank in the source itself, not a mapping gap), matching the same
    Cluster-1..5 segmentation already used for pipeline deals and now for the
    Health page's Cluster Governance view."""
    if series is None:
        return pd.Series(dtype="Int64")
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")


def map_project_table(df: pd.DataFrame, employee_lookup: dict[str, str]) -> pd.DataFrame:
    """-> 02_Project_Details_clean.csv shape. project_start_date/end_date are
    `datetime2(7)` and came through the sample export corrupted (time-only,
    no date) -- parsed defensively; may resolve cleanly against real
    production data even though the sample couldn't confirm it."""
    out = pd.DataFrame({
        "project_key": df.get("project_surrogate_key"),
        "project_code": _clean_id(df["project_code"]),
        "project_name": df.get("project_name"),
        "project_start_date": _parse_date_strict_only(df["project_start_date"]),
        "project_end_date": _parse_date_strict_only(df["project_end_date"]),
        "type_of_project": df.get("type_of_project"),
        "project_status": df.get("project_status"),
        "reporter_employee_id": _resolve(df["reporter_id"].astype(str), employee_lookup),
        "approver_employee_id": _resolve(df["approver_id"].astype(str), employee_lookup),
        "client_id": df.get("client_id"),
        "tech_coe": df.get("tech_coe"),
        "proposition_coe": df.get("proposition_coe"),
        "is_active_version": pd.to_numeric(df.get("is_active_version"), errors="coerce").fillna(1).astype(int),
        "date_source": "given",
        "extended_end_date": _parse_date_strict_only(df["latest_extension_end_date"]) if "latest_extension_end_date" in df else pd.NaT,
        "extended_end_status": None,
        "cluster_number": _parse_cluster_number(df.get("cluster")),
    })
    return out.dropna(subset=["project_code"])


def map_allocation_table(df: pd.DataFrame, employee_lookup: dict[str, str], project_lookup: dict[str, str]) -> pd.DataFrame:
    """-> 03_Project_Allocation_clean.csv shape. `resourcing_status` has no
    direct JDWH source column -- best-effort derived from the `billable` flag
    (BILLABLE/NON_BILLABLE), same vocabulary the local data already uses.
    `is_active_version` has no real source in this table either (project_allocation
    only tracks `is_active`, a different concept -- "currently active
    allocation", not SCD-versioning) -- defaults to 1 rather than reusing
    `is_active`'s value under a name that means something different.

    `is_allocation_active` is NOT a straight passthrough of the source's own
    `is_active` bit -- confirmed against real data that it doesn't mean what
    downstream code assumes: one real employee had 288 raw allocation rows
    spanning 2021-2023, virtually all still flagged is_active=1 even though
    every date range had long since ended (it evidently just means "never
    formally cancelled," not "temporally current"). allocation_report_service.py
    sums allocation_by_percentage across every row with is_allocation_active==1
    to get an employee's total current %, so trusting the source bit directly
    summed years of someone's entire history into a single "6,062.5%
    allocated" figure. Recomputed here as "the source says active AND the
    allocation's own date range genuinely covers today" -- the same kind of
    date-based currency check this app already uses elsewhere (e.g.
    adapter.py's not_yet_departed) rather than trusting an administrative
    flag whose real meaning turned out to be different from what the field
    name suggests.

    Tried extending "currently active" to also cover allocations with a
    matching recent (last 90 days of data) timesheet entry for the same
    employee+project, since project_allocation's own dates are stale for
    real client/internal/managed-services work (its own `updated_at` never
    moves past 2023-01-24, across all 339 employees it covers -- not just
    the dates, the whole employee population in this table froze then).
    That bridge is impossible with this data: cross-referencing confirmed
    ZERO of the 391 employees with real timesheet activity in the most
    recent 90 days of data have any row at all in project_allocation, under
    either UUID column. The two tables track two disjoint employee
    populations by this point -- project_allocation cannot be repaired into
    a "who's on client work right now" view via mapping alone; that requires
    the JDWH source to resume updating it, or a different report built
    directly from timesheet/WSR instead of project_allocation."""
    billable = df.get("billable").astype(str).str.strip().str.lower()
    start = _parse_date(df["start_date"])
    end = _parse_date(df["end_date"])
    source_is_active = pd.to_numeric(df.get("is_active"), errors="coerce").fillna(0).astype(int) == 1
    today = pd.Timestamp.now().normalize()
    temporally_current = (start.isna() | (start <= today)) & (end.isna() | (end >= today))
    out = pd.DataFrame({
        "project_rolebased_user_id": df.get("project_rolebased_user_id"),
        "project_id": _resolve(df["project_id"].astype(str), project_lookup),
        "employee_id": _resolve(df["jin_employee_id"].astype(str), employee_lookup),
        "resourcing_status": billable.map({"true": "BILLABLE", "1": "BILLABLE", "false": "NON_BILLABLE", "0": "NON_BILLABLE"}),
        "allocated_start_date": start,
        "allocated_end_date": end,
        "is_allocation_active": (source_is_active & temporally_current).astype(int),
        "allocation_by_percentage": pd.to_numeric(df.get("allocation_by_percentage"), errors="coerce"),
        "is_active_version": 1,
        "extended_end_date": pd.NaT,
        "extended_status": None,
        "extended_start_date": pd.NaT,
        "shift_type": None,
        "reviewer_employee_id": None,
    })
    out = out.dropna(subset=["employee_id", "project_id"])
    # core.project_allocation's own valid_to is essentially never closed out
    # (99.7% of real rows are still "9999-12-31", regardless of is_active) --
    # confirmed the source creates a brand new project_rolebased_user_id (a
    # distinct "role assignment" record) for what is otherwise the exact same
    # allocation (same employee, project, date range, %, billable flag)
    # multiple times -- one real employee had the identical BAU allocation
    # (same project, same 2019-02-01..2030-01-01 range, same 100%) recorded 5
    # separate times. allocation_report_service.py sums
    # allocation_by_percentage per employee across every row here, so each
    # re-created record silently multiplied that one real allocation into the
    # total (1700% instead of the real 1300% for that project mix). See
    # _dedupe_allocations for why resourcing_status isn't part of the key.
    return _dedupe_allocations(out)


# Same (employee, project, date range, %) can legitimately show a different
# resourcing_status across sources/re-created records -- confirmed real:
# project_allocation records JMG105's BAU_001 (2022-09-05..2030-12-31, 100%)
# as NON_BILLABLE, while project_rolebased_user records the exact same
# assignment as BILLABLE. Including resourcing_status in the dedup key (an
# earlier version of this fix) let both survive as if they were two separate
# commitments, silently doubling that employee's real total (2900% instead
# of ~1450%) -- worse than the duplication problem this dedup exists to fix.
# The 4-field identity (who/what/when/how much) is what makes it "the same
# allocation"; status is just an attribute of it, and keep="last" already
# prefers whichever source is concatenated later (the newer one, when both
# project_allocation and project_rolebased_user are combined in map_all_tables).
_ALLOCATION_DEDUP_KEY = ["employee_id", "project_id", "allocated_start_date", "allocated_end_date", "allocation_by_percentage"]


def _dedupe_allocations(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    return df.drop_duplicates(subset=_ALLOCATION_DEDUP_KEY, keep=keep)


def map_rolebased_user_table(df: pd.DataFrame, employee_lookup: dict[str, str], project_lookup: dict[str, str]) -> pd.DataFrame:
    """-> 03_Project_Allocation_clean.csv shape, same as map_allocation_table,
    but for a NEW, separately-provisioned real table (`project_rolebased_user`)
    that appears to be the live/current counterpart to the frozen
    `project_allocation` sheet: `data_loaded_at` on this table is a single
    fresh timestamp (loaded the same day as employee/project), vs
    project_allocation's own `updated_at` that never moves past 2023-01-24.
    Confirmed its `is_allocation_active` bit is actually trustworthy here --
    unlike project_allocation's, 100% of is_allocation_active==1 rows in this
    table ALSO have a date range genuinely covering today, and the currently-
    active slice spans real Client Project (808)/Internal Project (285)/
    Managed Services (91) work, not just BAU (13,919) -- the real diversity
    project_allocation could never produce. Still recomputed the same
    date-based way as map_allocation_table rather than trusted blindly, in
    case a future export of this table regresses the same way
    project_allocation did.

    Unlike project_allocation, `resourcing_status` is already a direct real
    column here in this app's own vocabulary (BILLABLE/SHADOW/UNBILLED/
    PROPOSED/PENDING) -- no `billable`-boolean derivation needed, just
    passthrough (upper-cased/stripped)."""
    start = _parse_date(df["allocated_start_date"])
    end = _parse_date(df["allocated_end_date"])
    source_is_active = pd.to_numeric(df.get("is_allocation_active"), errors="coerce").fillna(0).astype(int) == 1
    today = pd.Timestamp.now().normalize()
    temporally_current = (start.isna() | (start <= today)) & (end.isna() | (end >= today))
    out = pd.DataFrame({
        "project_rolebased_user_id": df.get("project_rolebased_user_id"),
        "project_id": _resolve(df["project_id"].astype(str), project_lookup),
        "employee_id": _resolve(df["jin_employee_id"].astype(str), employee_lookup),
        "resourcing_status": df.get("resourcing_status").astype(str).str.strip().str.upper(),
        "allocated_start_date": start,
        "allocated_end_date": end,
        "is_allocation_active": (source_is_active & temporally_current).astype(int),
        "allocation_by_percentage": pd.to_numeric(df.get("allocation_by_percentage"), errors="coerce"),
        "is_active_version": 1,
        "extended_end_date": pd.NaT,
        "extended_status": None,
        "extended_start_date": pd.NaT,
        "shift_type": None,
        "reviewer_employee_id": None,
    })
    out = out.dropna(subset=["employee_id", "project_id"])
    # This table has the identical un-closed-SCD duplication pattern as
    # project_allocation -- confirmed real: JMG105's BAU_001 assignment
    # (2022-09-05..2030-12-31, 100%, BILLABLE) appears twice here under two
    # different project_rolebased_user_id values, same as every other BAU
    # project this employee has. Same content-based dedup, same reason.
    return _dedupe_allocations(out)


def map_timesheet_table(
    df: pd.DataFrame, employee_lookup: dict[str, str], project_lookup: dict[str, str], mapped_employees: pd.DataFrame,
) -> pd.DataFrame:
    """-> 04_Timesheet_Details_clean.csv shape. `department_name` has no
    source column in core.timesheet at all -- looked up from the
    already-mapped employee table by the resolved employee_id, since that's
    the only place this app's data has it."""
    employee_id = _resolve(df["jin_employee_id"].astype(str), employee_lookup)
    # map_employee_table already deduplicates to one row per employee_id, but
    # a non-unique index here would crash .map() outright (confirmed:
    # `InvalidIndexError: Reindexing only valid with uniquely valued Index
    # objects`) -- deduplicating again is cheap insurance against being
    # called with a not-yet-deduplicated employee table.
    dept_by_employee = mapped_employees.drop_duplicates(subset="employee_id", keep="first").set_index("employee_id")["department_name"]
    is_billable = pd.to_numeric(df.get("is_billable"), errors="coerce")
    out = pd.DataFrame({
        "timesheet_surrogate_key": df.get("timesheet_surrogate_key"),
        "employee_id": employee_id,
        "timesheet_id": df.get("timesheet_id"),
        "manager_id": _resolve(df["manager_id"].astype(str), employee_lookup),
        "project_id": _resolve(df["project_id"].astype(str), project_lookup),
        "project_task_id": df.get("project_task_id"),
        "date": _parse_date(df["date"]),
        "time": pd.to_numeric(df.get("time"), errors="coerce"),
        "status": df.get("status"),
        "created_at": _parse_date(df["created_at"]),
        "updated_at": _parse_date(df["updated_at"]),
        "job_name": df["job_name"].replace(_JOB_NAME_TYPO_FIXES),
        "department_name": employee_id.map(dept_by_employee),
        "billing_status": is_billable.map({1: "BILLABLE", 0: "NON_BILLABLE"}),
    })
    return out.dropna(subset=["employee_id"])


def map_wsr_table(df: pd.DataFrame, project_lookup: dict[str, str]) -> pd.DataFrame:
    """-> 08_WSR_Report_clean.csv shape. `comment` has no single source
    column -- core.weekly_status_report splits it across project_highlights/
    support_in_project/opportunities/about_project; coalesced in that
    priority order (first non-null wins) since only one free-text field is
    expected downstream. `risk` is kept as its own separate `risk_note`
    column rather than folded into that same coalesce -- it's real
    risk-specific narrative (confirmed: 243 of 32,538 real rows populated,
    vs. 0 for all 4 comment-source columns above), and the Cluster Governance
    view's Risks section needs to tell "this project has a real logged risk"
    apart from "this project just has a general status comment"."""
    comment = df.get("project_highlights")
    for col in ("support_in_project", "opportunities", "about_project"):
        if col in df:
            comment = comment.fillna(df[col])
    out = pd.DataFrame({
        "wsr_key": df.get("surrogate_key"),
        "wsr_id": df.get("wsr_id"),
        "project_id_masked": _resolve(df["project_id"].astype(str), project_lookup),
        "scope_status": df.get("scope_status"),
        "schedule_status": df.get("schedule_status"),
        "quality_status": df.get("quality_status"),
        "csat_status": df.get("csat_status"),
        "team_status": df.get("team_status"),
        "week_start_date": _parse_date(df["week_start_date"]),
        "week_end_date": _parse_date(df["week_end_date"]),
        "comment": comment,
        "risk_note": df.get("risk"),
        # Real PM-hygiene checkboxes -- sparse (~1% of real rows: 326/281/277
        # of 32,538) but genuinely populated where present, unlike the 4
        # comment-source columns above. NaN here means "not confirmed
        # either way", never "confirmed not done" -- never coerce to a
        # boolean default, or a missing flag would silently read as non-
        # compliance that was never actually reported.
        "jin_allocations_updated": df.get("jin_allocations_updated"),
        "team_timesheets_submitted": df.get("team_timesheets_submitted"),
        "devops_updated": df.get("devops_updated"),
    })
    return out.dropna(subset=["wsr_id"])


def map_all_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """raw: {"employee": df, "project": df, "project_allocation": df,
    "timesheet": df, "weekly_status_report": df} straight off
    INFORMATION_SCHEMA-confirmed real SELECT * pulls (see
    jdwh_connection_service.py) -- never called by this assistant, only by
    the Resource Manager's own "Load Tables" action. designation_history is
    deliberately not mapped here -- it's a temporal audit trail with no
    equivalent slot in this app's current data model.

    `project_rolebased_user` (optional key) is a newer, separately-
    provisioned table that appears to be the live/current counterpart to the
    frozen project_allocation table (see map_rolebased_user_table's
    docstring for the evidence). When present, its mapped rows are combined
    with project_allocation's own and content-deduplicated together --
    project_allocation still contributes real historical rows this table
    doesn't have, while project_rolebased_user contributes the real current
    staffing project_allocation stopped tracking. Optional so a workbook/
    connection that doesn't provision this table yet still loads fine."""
    employee_lookup = build_employee_code_lookup(raw["employee"])
    project_lookup = build_project_code_lookup(raw["project"])

    employees = map_employee_table(raw["employee"])
    projects = map_project_table(raw["project"], employee_lookup)
    allocations = map_allocation_table(raw["project_allocation"], employee_lookup, project_lookup)
    if "project_rolebased_user" in raw:
        rolebased_allocations = map_rolebased_user_table(raw["project_rolebased_user"], employee_lookup, project_lookup)
        allocations = _dedupe_allocations(pd.concat([allocations, rolebased_allocations], ignore_index=True), keep="last")
    timesheets = map_timesheet_table(raw["timesheet"], employee_lookup, project_lookup, employees)
    wsr_reports = map_wsr_table(raw["weekly_status_report"], project_lookup)

    return {
        "employees": employees,
        "projects": projects,
        "allocations": allocations,
        "timesheets": timesheets,
        "wsr_reports": wsr_reports,
    }
