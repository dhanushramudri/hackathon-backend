"""Flags employees whose current allocation sits on a project the Health monitor
already predicts will extend past its official end date (is_extension_risk /
devops_extension_risk). A person can look 100% free on paper while actually being
tied up on a project that's about to run long -- this surfaces that doubt wherever
employee availability is shown (Recommendations candidates, employee profile, free
pool, etc.) instead of only on the Health page itself.
"""
import threading
import time

from app.core.adapter import get_adapter

# No-arg, pure function of current health-report + allocation data -- called
# from ~7 unrelated places (recommendations, free pool, employee profile,
# leave, and once per redeploy-candidate lookup inside a new-project
# forecast's future-date search, which alone can call it 50-100+ times in a
# single request). Same short-TTL pattern as get_health_report -- avoid
# rebuilding the same dict from scratch on every one of those calls.
_HOLD_FLAGS_CACHE_TTL_SECONDS = 120
_hold_flags_cache: dict = {"flags": None, "computed_at": 0.0}
_hold_flags_cache_lock = threading.Lock()


def get_employee_hold_flags() -> dict[str, dict]:
    """employee_id -> {"on_hold": True, "projects": [{project_code, reason, ...}]}
    for anyone currently actively allocated (is_allocation_active == 1) to a
    project flagged as an extension risk. Employees with no such allocation are
    simply absent from the returned dict (treat missing key as "not on hold")."""
    now = time.monotonic()
    if _hold_flags_cache["flags"] is not None and (now - _hold_flags_cache["computed_at"]) < _HOLD_FLAGS_CACHE_TTL_SECONDS:
        return _hold_flags_cache["flags"]
    with _hold_flags_cache_lock:
        now = time.monotonic()
        if _hold_flags_cache["flags"] is not None and (now - _hold_flags_cache["computed_at"]) < _HOLD_FLAGS_CACHE_TTL_SECONDS:
            return _hold_flags_cache["flags"]
        flags = _compute_employee_hold_flags()
        _hold_flags_cache["flags"] = flags
        _hold_flags_cache["computed_at"] = time.monotonic()
        return flags


def _compute_employee_hold_flags() -> dict[str, dict]:
    from app.services.health_monitor_service import get_health_report

    report = get_health_report()
    flagged_projects = {
        r["project_code"]: r
        for r in report
        if r.get("is_extension_risk") or r.get("devops_extension_risk")
    }
    if not flagged_projects:
        return {}

    adapter = get_adapter()
    allocations = adapter.get_allocations()
    active = allocations[
        (allocations["is_allocation_active"] == 1)
        & (allocations["project_id"].isin(flagged_projects.keys()))
    ]

    holds: dict[str, dict] = {}
    for _, row in active.iterrows():
        emp_id = row["employee_id"]
        project_code = row["project_id"]
        info = flagged_projects[project_code]
        entry = holds.setdefault(emp_id, {"on_hold": True, "projects": []})
        entry["projects"].append({
            "project_code": project_code,
            "is_extension_risk": bool(info.get("is_extension_risk")),
            "devops_extension_risk": bool(info.get("devops_extension_risk")),
            "projected_extension_duration_label": info.get("projected_extension_duration_label"),
            "projected_extension_confidence": info.get("projected_extension_confidence"),
        })
    return holds
