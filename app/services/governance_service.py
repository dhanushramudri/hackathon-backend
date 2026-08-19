"""Assembles the Cluster Governance dashboard payload -- the live replacement
for one cluster's slides in the weekly JQA deck. Combines real project/WSR
data (adapter, health_monitor_service) with the app-authored manual entries
(cluster_assignment_service, governance_risk_service, governance_spotlight_
service, governance_kickoff_tracking_service). One function call per cluster
keeps the frontend to a single fetch, same shape as health_detail_service's
get_project_health_detail for the existing project drill-down."""
import pandas as pd

from app.core.adapter import get_adapter
from app.services.cluster_assignment_service import (
    CLUSTER_NAMES, cluster_counts, list_projects_by_cluster, list_unassigned_projects, set_cluster,
)
from app.services.governance_common import current_week_start, week_end as _week_end
from app.services.governance_kickoff_tracking_service import list_kickoff_tracking, save_kickoff_tracking
from app.services.governance_risk_service import add_risk, list_risks, resolve_risk
from app.services.governance_spotlight_service import (
    DEFAULT_SPOTLIGHT_SIZE, add_or_update_spotlight, list_manual_overrides, remove_from_spotlight,
)
from app.services.governance_synthetic_service import summarize_cluster, summarize_delivery_comment, synthesize_risk
from app.services.health_monitor_service import _RAG_COLUMNS, get_health_report, worst_wsr_signal_vectorized

DEVOPS_FIELDS = [
    "devops_data_available", "devops_extension_risk", "devops_open_tickets", "devops_blocked_tickets",
    "devops_in_progress_tickets", "devops_tickets_past_project_end", "devops_remaining_effort_hours",
    "devops_capacity_surplus_hours", "devops_is_overdue",
]

# Real PM-hygiene checkboxes from WSR (see map_wsr_table) -- sparse (~1% of
# real rows) but genuine where present. None means "not confirmed either
# way", never "confirmed not done" -- _real_bool preserves that distinction
# instead of defaulting a missing flag to False.
HYGIENE_FLAGS = ["jin_allocations_updated", "team_timesheets_submitted", "devops_updated"]

def _real_bool(value) -> bool | None:
    return bool(value) if pd.notna(value) else None

def _date_str(value) -> str | None:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else None

# How far ahead to look for real allocation ramp-down activity -- a tighter
# "immediate" window than health_monitor_service's own RAMP_DOWN_WINDOW_DAYS
# (30 days), since this is a weekly governance view, not a monthly one.
_RAMP_WINDOW_DAYS = 21

def devops_summary_text(p: dict) -> str:
    """Real one-line DevOps summary from the same raw fields the main Health
    table's DevopsBoardCell already renders client-side -- built here too so
    the Spotlight table's Comments AI summary (and a server-rendered DevOps
    Visibility column) can use the identical real signal without duplicating
    ticket-fetching logic."""
    if not p.get("devops_data_available"):
        return "No DevOps board data available for this project."
    if p.get("devops_open_tickets") == 0:
        # Same real fact (0 open tickets) as before, phrased in the deck's
        # own tone ("Daily usage is happening", "On track") instead of the
        # more clinical "All DevOps tickets closed."
        return "On track -- DevOps board is clear, no open tickets."
    parts = []
    if p.get("devops_blocked_tickets", 0):
        parts.append(f"{p['devops_blocked_tickets']} blocked")
    if p.get("devops_in_progress_tickets", 0):
        parts.append(f"{p['devops_in_progress_tickets']} in progress")
    surplus = p.get("devops_capacity_surplus_hours")
    if surplus is not None and surplus < 0:
        parts.append(f"{abs(surplus):.0f}h capacity shortfall")
    remaining = p.get("devops_remaining_effort_hours") or 0
    if remaining:
        parts.append(f"{remaining:.0f}h remaining")
    return (", ".join(parts) if parts else f"{p.get('devops_open_tickets')} open ticket(s)") + "."

def compute_delivery_signals(project_code: str, wsr_trend: str | None) -> dict:
    """Real "Milestone Visibility" facts for the Spotlight table -- there is
    no source column anywhere for literal milestones, so this is built from
    the real signals that ARE available: timesheet activity (is work
    actually being logged), allocation ramp changes (is the team about to
    shrink or just grew), and the real WSR trend already computed elsewhere
    in this app (health_monitor_service.trend_from_severity_series).

    Also returns activity_score -- real hours logged + a weighted count of
    real allocation ramp changes -- used to fill spotlight slots with the
    cluster's most active engagements once every genuinely at-risk project
    already has a seat (see get_cluster_dashboard's spotlight selection)."""
    adapter = get_adapter()
    today = pd.Timestamp.now().normalize()
    week_start_ts = pd.Timestamp(current_week_start())
    last_week_start, last_week_end = week_start_ts - pd.Timedelta(days=7), week_start_ts - pd.Timedelta(days=1)

    facts = []

    timesheets = adapter.get_timesheets()
    proj_ts = timesheets[
        (timesheets["project_id"] == project_code)
        & (timesheets["status"] != "REJECTED")
        & (timesheets["date"] >= last_week_start)
        & (timesheets["date"] <= last_week_end)
    ]
    hours = float(proj_ts["time"].astype(float).sum())
    contributors = proj_ts["employee_id"].nunique()
    if hours > 0:
        facts.append(f"{hours:.0f}h logged last week across {contributors} team member{'s' if contributors != 1 else ''}.")
    else:
        facts.append("No timesheet hours logged last week.")

    allocations = adapter.get_allocations()
    proj_alloc = allocations[(allocations["project_id"] == project_code) & (allocations["is_allocation_active"] == 1)]
    ending_soon = proj_alloc[
        proj_alloc["allocated_end_date"].between(today, today + pd.Timedelta(days=_RAMP_WINDOW_DAYS))
    ]
    ending_count = ending_soon["employee_id"].nunique()
    if ending_count:
        facts.append(f"{ending_count} team member{'s' if ending_count != 1 else ''} ending allocation within {_RAMP_WINDOW_DAYS} days.")
    starting_recently = proj_alloc[
        proj_alloc["allocated_start_date"].between(today - pd.Timedelta(days=14), today)
    ]
    starting_count = starting_recently["employee_id"].nunique()
    if starting_count:
        facts.append(f"{starting_count} team member{'s' if starting_count != 1 else ''} started in the last 14 days.")

    if wsr_trend:
        facts.append(f"WSR trend: {wsr_trend}.")

    return {"facts": facts, "activity_score": hours + 5 * (ending_count + starting_count)}

def _active_projects() -> pd.DataFrame:
    projects = get_adapter().get_projects()
    return projects[projects["project_status"] == "ACTIVE"].copy()

def list_clusters() -> list[dict]:
    active = _active_projects()
    counts = cluster_counts(active)
    return [{"number": n, "name": name, "project_count": counts.get(n, 0)} for n, name in CLUSTER_NAMES.items()]

def get_unassigned_projects() -> list[dict]:
    active = _active_projects()
    codes = list_unassigned_projects(active)
    rows = active[active["project_code"].isin(codes)]
    return [
        {"project_code": r["project_code"], "project_name": r.get("project_name"), "client_id": r.get("client_id")}
        for _, r in rows.iterrows()
    ]

def assign_cluster(project_code: str, cluster_number: int) -> dict:
    return set_cluster(project_code, cluster_number)

def get_cluster_dashboard(cluster_number: int) -> dict:
    if cluster_number not in CLUSTER_NAMES:
        raise ValueError(f"cluster_number must be one of {sorted(CLUSTER_NAMES)}")

    active = _active_projects()
    cluster_codes = list_projects_by_cluster(cluster_number, active)
    cluster_df = active[active["project_code"].isin(cluster_codes)].copy()

    health_by_code = {r["project_code"]: r for r in get_health_report()}

    projects: list[dict] = []
    for _, row in cluster_df.iterrows():
        code = row["project_code"]
        health = health_by_code.get(code, {})
        projects.append(
            {
                "project_code": code,
                "project_name": row.get("project_name"),
                "client_id": row.get("client_id"),
                "type_of_project": row.get("type_of_project"),
                "project_start_date": _date_str(row.get("project_start_date")),
                "project_end_date": _date_str(row.get("project_end_date")),
                "extended_end_date": _date_str(row.get("extended_end_date")),
                "wsr_latest_signal": health.get("wsr_latest_signal"),
                "wsr_worst_signal": health.get("wsr_worst_signal"),
                "wsr_trend": health.get("wsr_trend"),
                "risk_score": health.get("risk_score", 0),
                "risk_band": health.get("risk_band"),
                **{f: health.get(f) for f in DEVOPS_FIELDS},
            }
        )

    week_start = current_week_start()
    week_end_date = _week_end(week_start)
    week_start_ts, week_end_ts = pd.Timestamp(week_start), pd.Timestamp(week_end_date)

    project_by_code = {p["project_code"]: p for p in projects}

    kickoff_this_week = []
    ending_this_week = []
    open_risks = list_risks(cluster_codes, include_resolved=False)
    for r in open_risks:
        base = project_by_code.get(r["project_code"], {})
        r["project_name"] = base.get("project_name")
        r["wsr_signal"] = base.get("wsr_latest_signal")
    risks_by_project: dict[str, list[dict]] = {}
    for r in open_risks:
        risks_by_project.setdefault(r["project_code"], []).append(r)

    for _, row in cluster_df.iterrows():
        code = row["project_code"]
        start = row.get("project_start_date")
        end = row.get("project_end_date")
        extended = row.get("extended_end_date")
        if pd.notna(start) and week_start_ts <= start <= week_end_ts:
            kickoff_this_week.append(
                {"project_code": code, "project_name": row.get("project_name"), "project_start_date": _date_str(start)}
            )
        effective_end = extended if pd.notna(extended) else end
        if pd.notna(effective_end) and week_start_ts <= effective_end <= week_end_ts:
            ending_this_week.append(
                {
                    "project_code": code,
                    "project_name": row.get("project_name"),
                    "scheduled_end_date": _date_str(end),
                    "actual_end_date": _date_str(extended) or _date_str(end),
                    "open_risks": risks_by_project.get(code, []),
                }
            )

    kickoff_tracking = list_kickoff_tracking(cluster_codes, week_start)
    for entry in kickoff_this_week:
        tracking = kickoff_tracking.get(entry["project_code"])
        entry["kickoff_completed"] = tracking["kickoff_completed"] if tracking else "pending"
        entry["scope_approved"] = tracking["scope_approved"] if tracking else "pending"
        entry["devops_setup"] = tracking["devops_setup"] if tracking else "pending"
        entry["comment"] = tracking["comment"] if tracking else None

    # Default spotlight, matching how the real deck's picks actually read
    # (WRS_002/DJL_001/BMF_005 all have a real blocker or delivery risk;
    # HCE_001/NZO_003/PEC_003 are mostly fine but still active/notable) --
    # not "worst N" and not "all N", but "the genuinely at-risk projects,
    # plus whichever others are most active this week if there's room":
    #   Tier 1 -- any project with a real RED/AMBER WSR signal or a
    #             high/medium risk_band, ranked by risk_score descending.
    #             Never excluded from filling a slot by activity level --
    #             a real risk always gets a seat first.
    #   Tier 2 -- remaining projects ranked by real activity_score
    #             (compute_delivery_signals -- timesheet hours + allocation
    #             ramp changes this week), filling any slots left over.
    # Overridable per project per week either way (see
    # governance_spotlight_service) -- excluding one backfills from the
    # next candidate in its own tier's order so the count stays at
    # DEFAULT_SPOTLIGHT_SIZE whenever the cluster has enough projects.
    overrides = list_manual_overrides(cluster_codes, week_start)
    not_excluded = [c for c in cluster_codes if not overrides.get(c, {}).get("excluded")]

    def is_at_risk(code: str) -> bool:
        p = project_by_code.get(code, {})
        return p.get("wsr_latest_signal") in ("RED", "AMBER") or p.get("risk_band") in ("high", "medium")

    risk_tier = sorted(
        (c for c in not_excluded if is_at_risk(c)),
        key=lambda c: project_by_code.get(c, {}).get("risk_score", 0), reverse=True,
    )
    signals_by_code = {code: compute_delivery_signals(code, project_by_code.get(code, {}).get("wsr_trend")) for code in cluster_codes}
    activity_tier = sorted(
        (c for c in not_excluded if not is_at_risk(c)),
        key=lambda c: signals_by_code.get(c, {}).get("activity_score", 0), reverse=True,
    )
    auto_picks = (risk_tier + activity_tier)[:DEFAULT_SPOTLIGHT_SIZE]
    spotlight_codes = list(auto_picks)
    for code, ov in overrides.items():
        if not ov.get("excluded") and code not in spotlight_codes:
            spotlight_codes.append(code)

    spotlight = []
    for code in spotlight_codes:
        base = project_by_code.get(code)
        if base is None:
            continue
        # DevOps Visibility and Milestone Visibility are 100% real, computed
        # fresh every load -- never stored, never manually typed. Comments is
        # an AI sentence grounded strictly in those same two real inputs.
        # Score Card / Action Plan are the only fields still manually entered
        # (see governance_spotlight_service's docstring for why).
        devops_visibility = devops_summary_text(base)
        milestone_facts = signals_by_code.get(code, compute_delivery_signals(code, base.get("wsr_trend")))["facts"]
        note = overrides.get(code, {})
        spotlight.append(
            {
                **base,
                "devops_visibility": devops_visibility,
                "milestone_facts": milestone_facts,
                "comments": summarize_delivery_comment(code, devops_visibility, milestone_facts),
                "action_plan": note.get("action_plan"),
                "is_auto_picked": code in auto_picks,
            }
        )
    spotlight.sort(key=lambda s: s.get("risk_score", 0), reverse=True)

    wsr = get_adapter().get_wsr_reports()
    cluster_wsr = wsr[wsr["project_id_masked"].isin(cluster_codes)].copy()
    wsr_status: dict[str, list[dict]] = {"RED": [], "AMBER": [], "GREEN": [], "no_report": []}

    # Same "most recent non-null across all history" treatment for hygiene
    # flags only -- they only ever land on old (2024/early-2025) WSR rows in
    # the real data, and the UI already labels them "(on record)" with a
    # tooltip disclaiming they aren't necessarily from this week, so showing
    # a stale-but-real confirmation here is honest, not misleading.
    latest_hygiene_by_code: dict[str, dict[str, bool]] = {}
    for flag in HYGIENE_FLAGS:
        if flag not in cluster_wsr.columns:
            continue
        with_flag = cluster_wsr[cluster_wsr[flag].notna()].sort_values("week_start_date")
        for code, value in with_flag.groupby("project_id_masked")[flag].last().items():
            latest_hygiene_by_code.setdefault(code, {})[flag] = _real_bool(value)

    if not cluster_wsr.empty:
        cluster_wsr["worst_signal"] = worst_wsr_signal_vectorized(cluster_wsr)
        cluster_wsr = cluster_wsr[cluster_wsr["worst_signal"] != "NO_COLOR"]
        cluster_wsr = cluster_wsr.sort_values("week_start_date")
        latest_by_project = cluster_wsr.groupby("project_id_masked").tail(1)
        latest_by_code = {r["project_id_masked"]: r for _, r in latest_by_project.iterrows()}
    else:
        latest_by_code = {}

    # risk_note is read ONLY off each project's own current-latest real row
    # (the exact same row the WSR badge/RAG columns below come from) -- NOT
    # searched across older history. A real risk_note from months ago,
    # displayed next to THIS week's WSR badge, would silently pair two
    # different weeks' data as if one snapshot: confirmed live, e.g. a
    # project with a real May risk note ("no secondary resource to cover
    # leave...") had gone fully GREEN by August, but the old note kept
    # showing up next to the new GREEN badge as if still current -- exactly
    # the kind of mismatch this app's "verify against real data" rule exists
    # to prevent, even when the mismatched data is itself real.
    latest_risk_note_by_code: dict[str, str] = {}
    for code, row in latest_by_code.items():
        note = row.get("risk_note")
        if pd.notna(note):
            latest_risk_note_by_code[code] = note
    for code in cluster_codes:
        base = next((p for p in projects if p["project_code"] == code), None)
        name = base["project_name"] if base else None
        latest = latest_by_code.get(code)
        if latest is None:
            wsr_status["no_report"].append({"project_code": code, "project_name": name})
            continue
        hygiene = latest_hygiene_by_code.get(code, {})
        entry = {
            "project_code": code,
            "project_name": name,
            "comment": latest.get("comment") if pd.notna(latest.get("comment")) else None,
            "risk_note": latest_risk_note_by_code.get(code),
            **{col: latest.get(col) for col in _RAG_COLUMNS},
            **{f: hygiene.get(f) for f in HYGIENE_FLAGS},
        }
        bucket = latest["worst_signal"] if latest["worst_signal"] in ("RED", "AMBER") else "GREEN"
        wsr_status[bucket].append(entry)

    # Synthetic (or, where a real WSR `risk` narrative exists, real-but-
    # app-assembled) risk suggestions -- only for projects with no REAL
    # logged risk on file yet. Disappears the moment someone logs a real one
    # (or promotes this suggestion via "Log as real risk" in the UI).
    synthetic_risks = []
    for code in cluster_codes:
        if code in risks_by_project:
            continue
        latest = latest_by_code.get(code)
        if latest is None:
            continue
        base = project_by_code.get(code, {})
        suggestion = synthesize_risk(code, base.get("project_name"), latest, latest_risk_note_by_code.get(code))
        if suggestion:
            suggestion["wsr_signal"] = base.get("wsr_latest_signal")
            synthetic_risks.append(suggestion)

    # Cluster-level AI summary -- 3-4 opening bullet points for the call,
    # built ONLY from the same real aggregates already assembled above for
    # the Risks/Spotlight/Kick-off/Ending/WSR sections. No new data is
    # computed here; this just hands the AI a compact, factual digest of
    # what's already on the page and asks for the bullet-point version.
    # Project NAMES, not codes, are passed through -- codes are an internal
    # shorthand meaningless when read aloud on a call.
    def _label(entry: dict) -> str:
        return entry.get("project_name") or entry["project_code"]

    cluster_summary_points = summarize_cluster(
        cluster_number,
        CLUSTER_NAMES[cluster_number],
        {
            "project_count": len(cluster_codes),
            "red": len(wsr_status["RED"]), "amber": len(wsr_status["AMBER"]),
            "green": len(wsr_status["GREEN"]), "no_report": len(wsr_status["no_report"]),
            "red_names": ", ".join(_label(e) for e in wsr_status["RED"]),
            "amber_names": ", ".join(_label(e) for e in wsr_status["AMBER"]),
            "logged_risks": "; ".join(f"{_label(r)}: {r['risk_description']}" for r in open_risks),
            "flagged_risks": "; ".join(f"{_label(r)}: {r['risk_description']}" for r in synthetic_risks),
            "spotlight_count": len(spotlight),
            "spotlight_names": ", ".join(_label(s) for s in spotlight),
            "kickoff_names": ", ".join(_label(e) for e in kickoff_this_week),
            "ending_names": ", ".join(_label(e) for e in ending_this_week),
        },
    )

    return {
        "number": cluster_number,
        "name": CLUSTER_NAMES[cluster_number],
        "week_start_date": week_start,
        "week_end_date": week_end_date,
        "projects": projects,
        "open_risks": open_risks,
        "synthetic_risks": synthetic_risks,
        "spotlight": spotlight,
        "kickoff_this_week": kickoff_this_week,
        "ending_this_week": ending_this_week,
        "wsr_status": wsr_status,
        "cluster_summary_points": cluster_summary_points,
    }

# Thin re-exports so the router only needs one import line for the mutating
# endpoints (add_risk/resolve_risk/spotlight/kickoff-tracking already have
# their real implementations in their own service modules).
__all__ = [
    "list_clusters", "get_unassigned_projects", "assign_cluster", "get_cluster_dashboard",
    "add_risk", "resolve_risk", "add_or_update_spotlight", "remove_from_spotlight", "save_kickoff_tracking",
    "current_week_start",
]
