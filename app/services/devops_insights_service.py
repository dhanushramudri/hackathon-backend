"""
app/services/devops_insights_service.py

Azure DevOps board insights — project extension-risk detection.

Problem this solves
-------------------
WSR statuses are all GREEN, no burnout, no escalations — but at the last moment
the project manager requests an allocation extension because work isn't done.
This service detects that pattern early by looking at the DevOps board:

  • Tickets still In Progress / Blocked when the project end date is approaching
  • Tickets whose planned completion date extends past the project end date
  • Remaining effort hours vs time left on the project

All open work items are fetched in a SINGLE round-trip (one WIQL query + batched
detail calls) and then grouped by project code extracted from the AreaPath field.
This keeps the health report fast regardless of how many active projects exist.

Integration
-----------
Called from get_health_report() in the health service:

    from app.services.devops_insights_service import (
        fetch_open_devops_tickets,
        group_tickets_by_project_code,
        compute_devops_extension_risk,
        no_devops_config_risk,
    )

    # Before the per-project loop — single fetch for all projects
    _devops_enabled = bool(os.getenv("AZURE_DEVOPS_PAT"))
    if _devops_enabled:
        _all_open_tickets = fetch_open_devops_tickets()
        _tickets_by_project = group_tickets_by_project_code(_all_open_tickets)
    else:
        _tickets_by_project = {}

    # Inside the per-project loop
    devops_tickets = _tickets_by_project.get(project_code, [])
    devops_risk = (
        compute_devops_extension_risk(devops_tickets, row["project_end_date"])
        if _devops_enabled
        else no_devops_config_risk()
    )
    is_devops_extension_risk = devops_risk["has_devops_extension_risk"]
    if is_devops_extension_risk:
        root_causes.append("devops_extension_risk")

Environment variables required (same as board-automater):
    AZURE_DEVOPS_ORG      e.g. "jmangroupltd"
    AZURE_DEVOPS_PROJECT  e.g. "managed-services"
    AZURE_DEVOPS_PAT      Personal Access Token (Read work items scope)
"""

from __future__ import annotations

import base64
import os
import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from app.core.adapter import get_adapter


# ---------------------------------------------------------------------------
# File logging — writes to backend/logs/devops_insights.log as well as console
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "devops_insights.log"

logger = logging.getLogger("devops_insights")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_sh)
    logger.propagate = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many days before project end date we start flagging DevOps extension risk.
# E.g. 45 = flag if end date is within 45 days AND there are open/blocked tickets.
EXTENSION_RISK_WINDOW_DAYS = 14

STANDARD_WORKDAY_HOURS = 8

# Ticket states considered "still in flight"
_ACTIVE_STATES: frozenset[str] = frozenset({
    "in progress", "doing", "active",
    "in development", "in dev",
    "in review", "review",
    "in qa", "qa", "testing", "in testing",
    "qa requested", "qa in progress",
})

# Tickets explicitly stuck
_BLOCKED_STATES: frozenset[str] = frozenset({
    "blocked", "on hold", "pending", "pending dev", "impediment",
})

_ALL_OPEN_STATES: frozenset[str] = _ACTIVE_STATES | _BLOCKED_STATES

# States to exclude from the WIQL query (work already done)
_DONE_STATES_SQL = (
    "'Done', 'Closed', 'Resolved', 'Completed', 'Finished', "
    "'Cancelled', 'Removed', 'Inactive', 'Cut'"
)

# Python set for secondary client-side filter (handles case variations)
_DONE_STATES: frozenset[str] = frozenset({
    "done", "closed", "resolved", "completed", "finished",
    "cancelled", "removed", "inactive", "cut",
})

# Fields we need from each work item (comma-separated for the ?fields= param)
_DETAIL_FIELDS: str = ",".join([
    "System.Id",
    "System.Title",
    "System.State",
    "System.WorkItemType",
    "System.AreaPath",
    "System.IterationPath",
    "System.AssignedTo",
    "Microsoft.VSTS.Scheduling.StartDate",
    "Microsoft.VSTS.Scheduling.DueDate",
    "Microsoft.VSTS.Scheduling.OriginalEstimate",
    "Microsoft.VSTS.Scheduling.RemainingWork",
    "Microsoft.VSTS.Scheduling.CompletedWork",
])

# Project-code pattern in AreaPath: 2-6 uppercase letters, underscore, 2+ digits
# Matches: WOW_003, DNS_011, JMG_242, CLN_017, PTA_004, CON_003 …
_PROJECT_CODE_RE: re.Pattern = re.compile(r'\b([A-Z]{2,6}_\d{2,})\b')


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _azure_auth_headers() -> Optional[dict]:
    """Build Basic-auth headers from AZURE_DEVOPS_PAT env var. Returns None if not set."""
    pat = os.getenv("AZURE_DEVOPS_PAT")
    if not pat:
        return None
    token = base64.b64encode(f":{pat}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Area-path → project code
# ---------------------------------------------------------------------------

def extract_project_code_from_area_path(area_path: str) -> Optional[str]:
    """
    Extract the resource-system project code from an Azure DevOps AreaPath.

    Examples
    --------
    "managed-services\\WOW_003 Worldline Operations"  →  "WOW_003"
    "managed-services\\DNS_011 - Client Name"          →  "DNS_011"
    "managed-services\\JMG_242 Watch-Tower"            →  "JMG_242"
    "managed-services\\CON_003 Anything"               →  "CON_003"
    Returns None if no recognisable project code is found.
    """
    match = _PROJECT_CODE_RE.search(area_path or "")
    return match.group(1) if match else None



import time
import threading

_ticket_cache: dict = {"tickets": None, "fetched_at": 0.0}
_TICKET_CACHE_TTL_SECONDS = 600
_ticket_cache_lock = threading.Lock()

def fetch_open_devops_tickets_cached() -> list[dict]:
    now = time.monotonic()
    if _ticket_cache["tickets"] is not None and (now - _ticket_cache["fetched_at"]) < _TICKET_CACHE_TTL_SECONDS:
        return _ticket_cache["tickets"]
    with _ticket_cache_lock:
        # Re-check after acquiring the lock -- another thread may have
        # already refreshed the cache while we were waiting on it.
        now = time.monotonic()
        if _ticket_cache["tickets"] is not None and (now - _ticket_cache["fetched_at"]) < _TICKET_CACHE_TTL_SECONDS:
            return _ticket_cache["tickets"]
        tickets = fetch_open_devops_tickets()
        _ticket_cache["tickets"] = tickets
        _ticket_cache["fetched_at"] = time.monotonic()
        return tickets


_iteration_dates_cache: dict = {"dates": None, "fetched_at": 0.0}
_ITERATION_CACHE_TTL_SECONDS = 3600

def fetch_iteration_dates() -> dict[str, dict]:
    """Fetch the org's iteration (sprint) classification tree ONCE and flatten
    it into {full_iteration_path: {"start_date", "finish_date"}}. Ticket-level
    fields never carry a sprint's real start/end date -- only this separate
    Azure DevOps API does, so this must be a dedicated call, not something
    derivable from ticket DueDate fields."""
    now = time.monotonic()
    if _iteration_dates_cache["dates"] is not None and (now - _iteration_dates_cache["fetched_at"]) < _ITERATION_CACHE_TTL_SECONDS:
        return _iteration_dates_cache["dates"]

    azure_org = os.getenv("AZURE_DEVOPS_ORG")
    azure_proj = os.getenv("AZURE_DEVOPS_PROJECT")
    hdrs = _azure_auth_headers()
    if not all([azure_org, azure_proj, hdrs]):
        return {}

    url = (
        f"https://dev.azure.com/{azure_org}/{azure_proj}"
        f"/_apis/wit/classificationnodes/iterations?$depth=15&api-version=7.1"
    )
    try:
        resp = requests.get(url, headers=hdrs, timeout=25)
    except Exception as exc:
        logger.info(f"[DEVOPS INSIGHTS] Iteration dates fetch error: {exc}")
        return {}
    if resp.status_code != 200:
        logger.info(f"[DEVOPS INSIGHTS] Iteration dates fetch HTTP {resp.status_code}: {resp.text[:300]}")
        return {}

    dates_by_path: dict[str, dict] = {}
    dates_by_leaf: dict[str, list[dict]] = {}

    def _walk(node: dict, path_prefix: str):
        name = node.get("name", "")
        full_path = f"{path_prefix}\\{name}" if path_prefix else name
        attrs = node.get("attributes") or {}
        start = attrs.get("startDate")
        finish = attrs.get("finishDate")
        if start or finish:
            entry = {"start_date": start[:10] if start else None, "finish_date": finish[:10] if finish else None}
            dates_by_path[full_path] = entry
            dates_by_leaf.setdefault(name, []).append(entry)
        for child in node.get("children") or []:
            _walk(child, full_path)

    # _walk(resp.json(), azure_proj)
    _walk(resp.json(), "")

    # logger.info(f"[DEVOPS INSIGHTS][DEBUG] Fetched {len(dates_by_path)} iteration date entries.")
    # for _p, _d in list(dates_by_path.items())[:15]:
        # logger.info(f"[DEVOPS INSIGHTS][DEBUG]   {_p!r} -> start={_d['start_date']} finish={_d['finish_date']}")

    # Fallback map: leaf sprint name -> dates, only when that name is unambiguous
    # across the whole tree (e.g. "Sprint 28" appearing under exactly one node).
    leaf_fallback = {name: entries[0] for name, entries in dates_by_leaf.items() if len(entries) == 1}

    result = {"by_path": dates_by_path, "by_unambiguous_leaf": leaf_fallback}
    _iteration_dates_cache["dates"] = result
    _iteration_dates_cache["fetched_at"] = now
    return result
    
# ---------------------------------------------------------------------------
# DevOps network fetch — single call for ALL projects
# ---------------------------------------------------------------------------

def fetch_open_devops_tickets() -> list[dict]:
    """
    Execute ONE WIQL query to collect every non-Done work item in the Azure
    DevOps project, then batch-fetch their scheduling / effort fields.

    Returns a flat list of field-dictionaries — one dict per work item.
    Returns an empty list if credentials are missing or the call fails.
    """
    azure_org  = os.getenv("AZURE_DEVOPS_ORG")
    azure_proj = os.getenv("AZURE_DEVOPS_PROJECT")
    hdrs       = _azure_auth_headers()

    if not all([azure_org, azure_proj, hdrs]):
        print(
            "[DEVOPS INSIGHTS] Azure DevOps credentials not configured "
            "(AZURE_DEVOPS_ORG / AZURE_DEVOPS_PROJECT / AZURE_DEVOPS_PAT). "
            "DevOps extension-risk signal will be skipped."
        )
        return []

    wiql_url  = (
        f"https://dev.azure.com/{azure_org}/{azure_proj}"
        f"/_apis/wit/wiql?api-version=7.1"
    )
    wiql_body = {
        "query": (
            "SELECT [System.Id] FROM WorkItems "
            f"WHERE [System.TeamProject] = '{azure_proj}' "
            f"AND [System.State] NOT IN ({_DONE_STATES_SQL}) "
            "ORDER BY [System.AreaPath] ASC"
        )
    }

    # ── WIQL call ──────────────────────────────────────────────────────────
    try:
        resp = requests.post(wiql_url, headers=hdrs, json=wiql_body, timeout=25)
    except Exception as exc:
        print(f"[DEVOPS INSIGHTS] WIQL network error: {exc}")
        return []

    if resp.status_code != 200:
        print(
            f"[DEVOPS INSIGHTS] WIQL returned HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )
        return []

    refs = resp.json().get("workItems", [])
    if not refs:
        print("[DEVOPS INSIGHTS] No open work items found in DevOps.")
        return []

    ids = [r["id"] for r in refs]
    # print(f"[DEVOPS INSIGHTS] {len(ids)} open work items found — fetching details …")

    # ── Batched detail fetch (max 200 IDs per call) ─────────────────────────
    all_fields: list[dict] = []
    for i in range(0, len(ids), 200):
        batch   = ids[i : i + 200]
        ids_str = ",".join(str(x) for x in batch)
        url     = (
            f"https://dev.azure.com/{azure_org}/{azure_proj}/_apis/wit/workitems"
            f"?ids={ids_str}&fields={_DETAIL_FIELDS}&api-version=7.1"
        )
        try:
            det = requests.get(url, headers=hdrs, timeout=20)
            if det.status_code == 200:
                for item in det.json().get("value", []):
                    f = item.get("fields", {})
                    if f:
                        all_fields.append(f)

            elif det.status_code == 503:
                logger.info(f"[DEVOPS INSIGHTS] Detail batch {i // 200 + 1} got 503, retrying once after backoff")
                time.sleep(3)
                retry = requests.get(url, headers=hdrs, timeout=20)
                if retry.status_code == 200:
                    for item in retry.json().get("value", []):
                        f = item.get("fields", {})
                        if f:
                            all_fields.append(f)
                else:
                    logger.info(f"[DEVOPS INSIGHTS] Detail batch {i // 200 + 1} retry also failed: HTTP {retry.status_code}")
            else:
                print(
                    f"[DEVOPS INSIGHTS] Detail batch {i // 200 + 1} "
                    f"HTTP {det.status_code}"
                )
        except Exception as exc:
            print(f"[DEVOPS INSIGHTS] Detail batch {i // 200 + 1} error: {exc}")

    print(f"[DEVOPS INSIGHTS] Fetched {len(all_fields)} work-item field sets.")

    _area_paths_seen = sorted({f.get("System.AreaPath", "<none>") for f in all_fields})
    # logger.info(f"[DEVOPS INSIGHTS][DEBUG] {len(_area_paths_seen)} distinct AreaPaths returned:")
    # for _ap in _area_paths_seen:
    #     logger.info(f"[DEVOPS INSIGHTS][DEBUG]   {_ap!r}")

    return all_fields


# ---------------------------------------------------------------------------
# Group by project code
# ---------------------------------------------------------------------------

def group_tickets_by_project_code(
    all_tickets: list[dict],
) -> dict[str, list[dict]]:
    """
    Group the flat list returned by fetch_open_devops_tickets() by project code.

    Returns
    -------
    dict[project_code → [field_dict, …]]
    Tickets whose AreaPath doesn't match the project-code pattern are dropped.
    """
    grouped: dict[str, list[dict]] = {}
    unmatched = 0
    for fields in all_tickets:
        area_path = fields.get("System.AreaPath", "")
        pc = extract_project_code_from_area_path(area_path)
        if pc:
            grouped.setdefault(pc, []).append(fields)
        else:
            unmatched += 1

    # if unmatched:
    #     print(
    #         f"[DEVOPS INSIGHTS] {unmatched} work items had no recognisable "
    #         f"project code in AreaPath — they won't contribute to any risk signal."
    #     )
    # print(
    #     f"[DEVOPS INSIGHTS] Tickets grouped into {len(grouped)} project codes: "
    #     f"{sorted(grouped.keys())}"
    # )

    _counts = sorted(((code, len(tix)) for code, tix in grouped.items()), key=lambda x: -x[1])
    # logger.info("[DEVOPS INSIGHTS][DEBUG] Open ticket count by project code (busiest first):")
    # for _code, _count in _counts:
    #     logger.info(f"[DEVOPS INSIGHTS][DEBUG]   {_code}: {_count} open ticket(s)")

    return grouped



def _iteration_leaf_name(path: str) -> str:
    """Last segment of an IterationPath, e.g. '...\\JMG_108\\Sprint 14' -> 'Sprint 14'."""
    return path.split("\\")[-1] if path else "No sprint assigned"


def group_tickets_by_iteration(ticket_fields: list[dict]) -> dict[str, list[dict]]:
    """Group a project's OPEN (non-Done) tickets by IterationPath (sprint)."""
    grouped: dict[str, list[dict]] = {}
    for f in ticket_fields:
        state = (f.get("System.State") or "").strip().lower()
        if state in _DONE_STATES:
            continue
        iteration = f.get("System.IterationPath") or "No sprint assigned"
        grouped.setdefault(iteration, []).append(f)
    return grouped


def compute_sprint_breakdown(
    ticket_fields: list[dict],
    project_end_date: "pd.Timestamp",
) -> list[dict]:
    """Per-sprint ticket breakdown -- blocked / in-progress / to-do counts and
    remaining hours for each sprint that still has open tickets. Sorted with the
    sprint closest to "now" (latest due date seen in it) first, so a PM can see
    at a glance whether the sprint nearest the project end still has open
    work -- which is the signal that a new sprint (i.e. an extension) may be
    needed."""


    iteration_dates = fetch_iteration_dates()
    by_path = iteration_dates.get("by_path", {})
    by_leaf = iteration_dates.get("by_unambiguous_leaf", {})

    by_iteration = group_tickets_by_iteration(ticket_fields)
    rows = []
    for iteration_path, tickets in by_iteration.items():
        blocked = sum(1 for t in tickets if (t.get("System.State") or "").strip().lower() in _BLOCKED_STATES)
        in_progress = sum(1 for t in tickets if (t.get("System.State") or "").strip().lower() in _ACTIVE_STATES)
        to_do = len(tickets) - blocked - in_progress
        remaining = 0.0
        no_effort_data = 0
        for t in tickets:
            r = float(t.get("Microsoft.VSTS.Scheduling.RemainingWork") or 0)
            c = float(t.get("Microsoft.VSTS.Scheduling.CompletedWork") or 0)
            e = float(t.get("Microsoft.VSTS.Scheduling.OriginalEstimate") or 0)
            if r <= 0 and e > c:
                r = e - c
            elif r <= 0 and e <= 0:
                no_effort_data += 1
            remaining += r

        due_dates = []
        for t in tickets:
            due_str = t.get("Microsoft.VSTS.Scheduling.DueDate")
            if due_str:
                try:
                    ts = pd.Timestamp(due_str)
                    if ts.tzinfo is not None:
                        ts = ts.tz_localize(None)
                    due_dates.append(ts.normalize())
                except Exception:
                    pass
        latest_due = max(due_dates) if due_dates else None

        sprint_name = _iteration_leaf_name(iteration_path)
        dates = by_path.get(iteration_path) or by_leaf.get(sprint_name)
        sprint_start_date = dates["start_date"] if dates else None
        sprint_end_date = dates["finish_date"] if dates else None

        if not dates:
            logger.info(f"[DEVOPS INSIGHTS][DEBUG] No date match for iteration_path={iteration_path!r} (leaf={sprint_name!r})")


        rows.append({
            "iteration_path": iteration_path,
            "sprint_name": sprint_name,
            "ticket_count": len(tickets),
            "blocked_count": blocked,
            "in_progress_count": in_progress,
            "to_do_count": to_do,
            "remaining_hours": round(remaining, 1),
            "tickets_with_no_effort_data": no_effort_data,
             "sprint_start_date": sprint_start_date,
            "sprint_end_date": sprint_end_date,
            "latest_due_date": latest_due.strftime("%Y-%m-%d") if latest_due is not None else None,
            "has_open_work": (blocked + in_progress + to_do) > 0,
        })

    rows.sort(key=lambda r: (r["sprint_end_date"] or r["latest_due_date"] or "", r["ticket_count"]), reverse=True)

    # for r in rows:
    #     logger.info(
    #         f"[DEVOPS INSIGHTS][SPRINT] {r['sprint_name']}: {r['ticket_count']} tickets "
    #         f"({r['blocked_count']} blocked, {r['in_progress_count']} in-progress, {r['to_do_count']} to-do), "
    #         f"{r['remaining_hours']}h remaining, {r['tickets_with_no_effort_data']} with no effort data, "
    #         f"dates {r['sprint_start_date']}→{r['sprint_end_date']}"
    #     )
    return rows


def _team_daily_capacity_hours(project_code: str) -> float:
    """Team's steady-state capacity in hours/weekday, independent of the
    EXTENSION_RISK_WINDOW_DAYS window. Unlike _team_capacity_hours() (which
    is only computed when a project is within its risk window, and returns
    0.0 for overdue projects since working_days_in_window is 0 there), this
    is always available -- it's the daily rate needed to project how many
    MORE days an overdue project's remaining work will take, since an
    overdue project has no "window" left to measure against."""
    adapter = get_adapter()
    allocations = adapter.get_allocations()
    active_allocs = allocations[
        (allocations["project_id"] == project_code) & (allocations["is_allocation_active"] == 1)
    ]
    if active_allocs.empty:
        return 0.0
    total = sum(
        STANDARD_WORKDAY_HOURS * (float(a["allocation_by_percentage"]) / 100.0)
        for _, a in active_allocs.iterrows()
    )
    return round(total, 1)


def _count_working_days(start: "pd.Timestamp", end: "pd.Timestamp") -> int:
    """Count Mon–Fri days in [start, end] inclusive. Returns 0 if end < start or NaT."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    days = pd.date_range(start=start.normalize(), end=end.normalize(), freq="D")
    return int((days.dayofweek < 5).sum())


def _team_capacity_hours(
    project_code: str,
    window_start: "pd.Timestamp",
    window_end: "pd.Timestamp",
    working_days_in_window: int,
) -> tuple[float, float]:
    """
    Returns (capacity_before_leave, capacity_after_leave) hours for the project's
    actively allocated employees over the window.

    capacity_before_leave = sum over active employees of
        working_days_in_window * STANDARD_WORKDAY_HOURS * allocation_pct/100

    capacity_after_leave additionally subtracts, per employee, their approved leave
    days that overlap the window (weekdays only), at that employee's allocation %.
    """
    if working_days_in_window <= 0:
        return 0.0, 0.0

    adapter = get_adapter()
    allocations = adapter.get_allocations()
    leaves = adapter.get_leaves()

    active_allocs = allocations[
        (allocations["project_id"] == project_code) & (allocations["is_allocation_active"] == 1)
    ]

    capacity_before = 0.0
    capacity_after = 0.0
    for _, alloc in active_allocs.iterrows():
        pct = float(alloc["allocation_by_percentage"]) / 100.0
        emp_hours = working_days_in_window * STANDARD_WORKDAY_HOURS * pct
        capacity_before += emp_hours

        emp_leaves = leaves[leaves["employee_id"] == alloc["employee_id"]]
        leave_working_days = 0
        for _, lv in emp_leaves.iterrows():
            overlap_start = max(window_start, lv["leave_start_date"])
            overlap_end = min(window_end, lv["leave_end_date"])
            leave_working_days += _count_working_days(overlap_start, overlap_end)
        # Cap at the window's own working days -- overlapping leave rows shouldn't
        # deduct more than the employee was ever contributing in this window.
        leave_working_days = min(leave_working_days, working_days_in_window)
        capacity_after += emp_hours - (leave_working_days * STANDARD_WORKDAY_HOURS * pct)

    return round(capacity_before, 1), round(max(capacity_after, 0.0), 1)

def _team_daily_capacity_hours(project_code: str) -> float:
    """Team's steady-state capacity in hours/weekday, independent of the
    EXTENSION_RISK_WINDOW_DAYS window. Unlike _team_capacity_hours() (which is
    only computed when a project is within its risk window, and returns 0.0
    for overdue projects since working_days_in_window is 0 there), this is
    always available -- it's the daily rate needed to project how many MORE
    days an overdue project's remaining work will take, since an overdue
    project has no "window" left to measure against."""
    adapter = get_adapter()
    allocations = adapter.get_allocations()
    active_allocs = allocations[
        (allocations["project_id"] == project_code) & (allocations["is_allocation_active"] == 1)
    ]
    if active_allocs.empty:
        return 0.0
    total = sum(
        STANDARD_WORKDAY_HOURS * (float(a["allocation_by_percentage"]) / 100.0)
        for _, a in active_allocs.iterrows()
    )
    return round(total, 1)

def _is_effort_inconsistent(fields: dict, today: "pd.Timestamp") -> bool:
    """Flag an in-progress ticket with no completed work logged despite having
    started several calendar days ago."""
    start_str = fields.get("Microsoft.VSTS.Scheduling.StartDate")
    if not start_str:
        return False
    try:
        start_dt = pd.Timestamp(start_str)
        if start_dt.tzinfo is not None:
            start_dt = start_dt.tz_localize(None)
        start_dt = start_dt.normalize()
    except Exception:
        return False
    elapsed_days = (today - start_dt).days
    completed = float(fields.get("Microsoft.VSTS.Scheduling.CompletedWork") or 0)
    estimate = float(fields.get("Microsoft.VSTS.Scheduling.OriginalEstimate") or 0)
    remaining = float(fields.get("Microsoft.VSTS.Scheduling.RemainingWork") or 0)
    return elapsed_days >= 5 and completed == 0 and (remaining > 0 or estimate > 0)


def compute_devops_ticket_stats(ticket_fields: list[dict]) -> dict:
    """Project-independent ticket aggregates -- computed ONCE and reused across
    every project that shares the same ticket set (e.g. the
    DEMO_STATIC_DEVOPS_PROJECT_CODE override), instead of re-scanning the same
    hundreds of tickets once per project on every health-report request. None
    of these numbers depend on any individual project's end date."""
    if not ticket_fields:
        return {
            "open_count": 0, "blocked_count": 0, "in_progress_count": 0, "to_do_count": 0,
            "remaining_hours": 0.0, "completed_hours": 0.0, "orig_est_hours": 0.0,
            "effort_completion_pct": None,
            "missing_remaining_estimate": 0, "no_effort_data_count": 0,
            "sprint_breakdown": [],
        }

    open_count = blocked_count = in_progress_count = to_do_count = 0
    remaining_hours = completed_hours = orig_est_hours = 0.0
    missing_remaining_estimate = no_effort_data_count = 0

    for f in ticket_fields:
        state = (f.get("System.State") or "").strip().lower()
        if state in _DONE_STATES:
            continue
        open_count += 1
        if state in _BLOCKED_STATES:
            blocked_count += 1
        elif state in _ACTIVE_STATES:
            in_progress_count += 1
        else:
            to_do_count += 1

        remaining = float(f.get("Microsoft.VSTS.Scheduling.RemainingWork") or 0)
        completed = float(f.get("Microsoft.VSTS.Scheduling.CompletedWork") or 0)
        estimate = float(f.get("Microsoft.VSTS.Scheduling.OriginalEstimate") or 0)

        if remaining <= 0 and estimate > completed:
            remaining = estimate - completed
            missing_remaining_estimate += 1
        elif remaining <= 0 and estimate <= 0:
            no_effort_data_count += 1

        remaining_hours += remaining
        completed_hours += completed
        orig_est_hours += estimate

    total_work = completed_hours + remaining_hours
    effort_completion_pct = round(100.0 * completed_hours / total_work, 1) if total_work > 0 else None

    return {
        "open_count": open_count,
        "blocked_count": blocked_count,
        "in_progress_count": in_progress_count,
        "to_do_count": to_do_count,
        "remaining_hours": round(remaining_hours, 1),
        "completed_hours": round(completed_hours, 1),
        "orig_est_hours": round(orig_est_hours, 1),
        "effort_completion_pct": effort_completion_pct,
        "missing_remaining_estimate": missing_remaining_estimate,
        "no_effort_data_count": no_effort_data_count,
        "sprint_breakdown": compute_sprint_breakdown(ticket_fields, project_end_date=None),
    }


# ---------------------------------------------------------------------------
# Per-project risk computation
# ---------------------------------------------------------------------------
def compute_devops_extension_risk(
    ticket_fields: list[dict],
    project_end_date: "pd.Timestamp",
    project_code: str,
    ticket_stats: dict | None = None,
) -> dict:
    """
    Compute capacity-based extension-risk metrics for a single project.

    ticket_stats: pass the result of compute_devops_ticket_stats(ticket_fields)
    when the same ticket_fields list backs multiple projects, so the expensive
    ticket-level scan runs once instead of once per project. If omitted, it's
    computed here (backward compatible, but slow when called in a loop).
    """
    today = pd.Timestamp.now().normalize()
    end_valid = pd.notna(project_end_date)
    days_to_end = int((project_end_date - today).days) if end_valid else None
    within_risk_window = days_to_end is not None and 0 <= days_to_end <= EXTENSION_RISK_WINDOW_DAYS
    is_overdue = days_to_end is not None and days_to_end < 0

    working_days_in_window = _count_working_days(today, project_end_date) if within_risk_window else 0
    capacity_before_leave, capacity_after_leave = (
        _team_capacity_hours(project_code, today, project_end_date, working_days_in_window)
        if within_risk_window
        else (0.0, 0.0)
    )

    if not ticket_fields:
        return {
            "devops_data_available": True,
            "has_devops_extension_risk": False,
            "open_ticket_count": 0,
            "blocked_ticket_count": 0,
            "in_progress_ticket_count": 0,
            "to_do_ticket_count": 0,
            "tickets_due_past_project_end": 0,
            "remaining_effort_hours": 0.0,
            "completed_work_hours": 0.0,
            "original_estimate_hours": 0.0,
            "effort_completion_pct": None,
            "within_risk_window": within_risk_window,
            "is_overdue": is_overdue,
            "working_days_in_window": working_days_in_window,
            "team_capacity_hours": capacity_before_leave,
            "team_capacity_hours_after_leave": capacity_after_leave,
            "team_daily_capacity_hours": _team_daily_capacity_hours(project_code),
            "capacity_surplus_hours": None,
            "tickets_missing_remaining_estimate": 0,
            "tickets_with_no_effort_data": 0,
            "sprint_breakdown": [],
            "likely_needs_new_sprint": False,
        }

    stats = ticket_stats if ticket_stats is not None else compute_devops_ticket_stats(ticket_fields)

    # Only pass -- cheap single-field scan, genuinely depends on THIS project's
    # end date so it can't be precomputed/shared like the rest of `stats`.
    tickets_past_due = 0
    if end_valid:
        for f in ticket_fields:
            state = (f.get("System.State") or "").strip().lower()
            if state in _DONE_STATES:
                continue
            due_str = f.get("Microsoft.VSTS.Scheduling.DueDate")
            if not due_str:
                continue
            try:
                due_dt = pd.Timestamp(due_str).normalize()
                if due_dt > project_end_date:
                    tickets_past_due += 1
            except Exception:
                pass

    sprint_breakdown = stats["sprint_breakdown"]
    likely_needs_new_sprint = bool(
        sprint_breakdown and sprint_breakdown[0]["has_open_work"] and (is_overdue or within_risk_window)
    )

    capacity_surplus_hours = (
        round(capacity_after_leave - stats["remaining_hours"], 1) if (within_risk_window or is_overdue) else None
    )

    has_risk = bool(
        stats["blocked_count"] > 0
        or tickets_past_due > 0
        or (within_risk_window and stats["remaining_hours"] > capacity_after_leave)
        or (is_overdue and stats["open_count"] > 0)
    )

    return {
        "devops_data_available":        True,
        "has_devops_extension_risk":    has_risk,
        "open_ticket_count":            stats["open_count"],
        "blocked_ticket_count":         stats["blocked_count"],
        "in_progress_ticket_count":     stats["in_progress_count"],
        "to_do_ticket_count":           stats["to_do_count"],
        "tickets_due_past_project_end": tickets_past_due,
        "remaining_effort_hours":       stats["remaining_hours"],
        "completed_work_hours":         stats["completed_hours"],
        "original_estimate_hours":      stats["orig_est_hours"],
        "effort_completion_pct":        stats["effort_completion_pct"],
        "within_risk_window":           within_risk_window,
        "is_overdue":                   is_overdue,
        "working_days_in_window":       working_days_in_window,
        "team_capacity_hours":          capacity_before_leave,
        "team_capacity_hours_after_leave": capacity_after_leave,
        "team_daily_capacity_hours":     _team_daily_capacity_hours(project_code),
        "capacity_surplus_hours":       capacity_surplus_hours,
        "tickets_missing_remaining_estimate": stats["missing_remaining_estimate"],
        "tickets_with_no_effort_data": stats["no_effort_data_count"],
        "sprint_breakdown": sprint_breakdown,
        "likely_needs_new_sprint": likely_needs_new_sprint,
    }


def no_devops_config_risk() -> dict:
    """
    Sentinel value returned when Azure DevOps credentials are not configured.
    Keeps the health-report record schema consistent.
    """
    return {
        "devops_data_available":        False,
        "has_devops_extension_risk":    False,
        "open_ticket_count":            0,
        "blocked_ticket_count":         0,
        "in_progress_ticket_count":     0,
        "to_do_ticket_count":           0,
        "tickets_due_past_project_end": 0,
        "remaining_effort_hours":       0.0,
        "completed_work_hours":         0.0,
        "original_estimate_hours":      0.0,
        "effort_completion_pct":        None,
        "within_risk_window":           False,
        "is_overdue":                   False,
        "working_days_in_window":       0,
        "team_capacity_hours":          0.0,
        "team_capacity_hours_after_leave": 0.0,
        "team_daily_capacity_hours":     0.0,
         "team_daily_capacity_hours":    0.0,
        "capacity_surplus_hours":       None,
        "tickets_missing_remaining_estimate": 0,
        "tickets_with_no_effort_data": 0,
        "sprint_breakdown": [],
        "likely_needs_new_sprint": False,

    }

def list_devops_tickets_for_display(
    ticket_fields: list[dict],
    project_end_date: "pd.Timestamp",
) -> list[dict]:
    """Per-ticket rows for the DevOps tab in the project detail view.
    Blocked tickets first, then tickets due past the project end, then
    in-progress tickets — same priority order as the risk decision itself."""
    end_valid = pd.notna(project_end_date)
    _today = pd.Timestamp.now().normalize()
    rows = []
    for f in ticket_fields:
        state = (f.get("System.State") or "").strip()
        state_lower = state.lower()
        if state_lower in _DONE_STATES:
            continue

        due_str = f.get("Microsoft.VSTS.Scheduling.DueDate")
        is_past_due = False
        if due_str and end_valid:
            try:
                is_past_due = pd.Timestamp(due_str).normalize() > project_end_date
            except Exception:
                pass

        assigned = f.get("System.AssignedTo")
        assigned_name = assigned.get("displayName") if isinstance(assigned, dict) else assigned


        r = float(f.get("Microsoft.VSTS.Scheduling.RemainingWork") or 0)
        c = float(f.get("Microsoft.VSTS.Scheduling.CompletedWork") or 0)
        e = float(f.get("Microsoft.VSTS.Scheduling.OriginalEstimate") or 0)
        has_effort_data = True
        if r <= 0 and e > c:
            r = e - c
        elif r <= 0 and e <= 0:
            has_effort_data = False

        rows.append({
            "id": f.get("System.Id"),
            "title": f.get("System.Title"),
            "work_item_type": f.get("System.WorkItemType"),
            "state": state,
            "is_blocked": state_lower in _BLOCKED_STATES,
            "is_in_progress": state_lower in _ACTIVE_STATES,
            "assigned_to": assigned_name,
            "start_date": f.get("Microsoft.VSTS.Scheduling.StartDate"),
            "due_date": due_str,
            "is_past_project_end": is_past_due,
            "original_estimate_hours": f.get("Microsoft.VSTS.Scheduling.OriginalEstimate"),
            "remaining_hours": f.get("Microsoft.VSTS.Scheduling.RemainingWork"),
            "completed_hours": f.get("Microsoft.VSTS.Scheduling.CompletedWork"),
            "is_effort_inconsistent": _is_effort_inconsistent(f, _today),
            "sprint_name": _iteration_leaf_name(f.get("System.IterationPath") or ""),
            "effective_remaining_hours": round(r, 1) if has_effort_data else None,
        })

    rows.sort(key=lambda r: (not r["is_blocked"], not r["is_past_project_end"], not r["is_in_progress"]))
    return rows



def log_devops_project_candidates(all_tickets: list[dict]) -> None:
    """Log per-project ticket stats so a real, currently-busy project can be
    picked for DEMO_STATIC_DEVOPS_PROJECT_CODE instead of a static guess.
    Run this file directly and check logs/devops_insights.log afterward:

        python -m app.services.devops_insights_service
    """
    grouped = group_tickets_by_project_code(all_tickets)
    today = pd.Timestamp.now().normalize()
    recent_cutoff = today - pd.Timedelta(days=60)
    month_start = today.replace(day=1) 

    stats = []
    for project_code, tickets in grouped.items():
        open_tickets = [t for t in tickets if (t.get("System.State") or "").strip().lower() not in _DONE_STATES]
        blocked = sum(1 for t in open_tickets if (t.get("System.State") or "").strip().lower() in _BLOCKED_STATES)
        in_progress = sum(1 for t in open_tickets if (t.get("System.State") or "").strip().lower() in _ACTIVE_STATES)
        iterations = {t.get("System.IterationPath") for t in open_tickets if t.get("System.IterationPath")}

        recent_count = 0
        this_month_count = 0
        assignees_this_month: set[str] = set()
        for t in open_tickets:
            has_date_this_month = False
            for field in ("Microsoft.VSTS.Scheduling.StartDate", "Microsoft.VSTS.Scheduling.DueDate"):
                val = t.get(field)
                if val:
                    try:
                        ts = pd.Timestamp(val)
                        if ts.tzinfo is not None:
                            ts = ts.tz_localize(None)
                        ts_norm = ts.normalize()
                        if ts_norm >= recent_cutoff:
                            recent_count += 1
                            break
                    except Exception:
                        pass
            for field in ("Microsoft.VSTS.Scheduling.StartDate", "Microsoft.VSTS.Scheduling.DueDate"):
                val = t.get(field)
                if val:
                    try:
                        ts = pd.Timestamp(val)
                        if ts.tzinfo is not None:
                            ts = ts.tz_localize(None)
                        ts_norm = ts.normalize()
                        if ts_norm >= month_start:
                            has_date_this_month = True
                    except Exception:
                        pass
            if has_date_this_month:
                this_month_count += 1
                assigned = t.get("System.AssignedTo")
                name = assigned.get("displayName") if isinstance(assigned, dict) else assigned
                if name:
                    assignees_this_month.add(name)

        stats.append({
            "project_code": project_code,
            "open_ticket_count": len(open_tickets),
            "blocked": blocked,
            "in_progress": in_progress,
            "distinct_sprints": len(iterations),
            "tickets_with_recent_dates": recent_count,
            "tickets_this_month": this_month_count,
            "distinct_employees_this_month": len(assignees_this_month),
        })

    # Primary sort: tickets actually active THIS month (the real "currently
    # being worked on" signal) -> distinct employees touching it this month
    # (breadth of real activity, not just one person's backlog dump) ->
    # in-progress count -> total open volume as a final tiebreaker.
    stats.sort(
        key=lambda s: (
            s["tickets_this_month"],
            s["distinct_employees_this_month"],
            s["in_progress"],
            s["open_ticket_count"],
        ),
        reverse=True,
    )

    logger.info("=== DevOps project candidates (sorted by recent activity, then volume) ===")
    logger.info(f"(today = {today.date()}, current month starts {month_start.date()})")
    for s in stats[:20]:
        logger.info(
            f"{s['project_code']}: {s['open_ticket_count']} open, {s['blocked']} blocked, "
            f"{s['in_progress']} in progress, {s['distinct_sprints']} distinct sprint(s), "
             f"{s['tickets_with_recent_dates']} ticket(s) with a start/due date in the last 60 days, "
            f"{s['tickets_this_month']} ticket(s) with a start/due date THIS MONTH, "
            f"{s['distinct_employees_this_month']} distinct employee(s) active this month"
        )

    if stats:
        best = stats[0]
        logger.info("")
        logger.info(
            f"[DEVOPS INSIGHTS][RECOMMENDED] Best candidate for DEMO_STATIC_DEVOPS_PROJECT_CODE: "
            f"'{best['project_code']}' — {best['tickets_this_month']} ticket(s) active this month, "
            f"{best['distinct_employees_this_month']} distinct employee(s) touching it, "
            f"{best['in_progress']} in progress overall, {best['open_ticket_count']} open total."
        )
        # Runner-up context, in case the top pick has some other issue (e.g. too
        # few employees actually mapped in the resourcing system for that code).
        for s in stats[1:4]:
            logger.info(
                f"[DEVOPS INSIGHTS][RECOMMENDED] Runner-up: '{s['project_code']}' — "
                f"{s['tickets_this_month']} this month, {s['distinct_employees_this_month']} employees, "
                f"{s['in_progress']} in progress, {s['open_ticket_count']} open total."
            )
    else:
        logger.info("[DEVOPS INSIGHTS][RECOMMENDED] No project candidates found at all.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("=== Logging DevOps project candidates for demo project selection ===")
    _tickets = fetch_open_devops_tickets()
    log_devops_project_candidates(_tickets)