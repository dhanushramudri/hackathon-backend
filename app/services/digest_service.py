from app.services.health_monitor_service import get_health_report
from app.services.leave_service import get_leave_impact

TOP_N_RISK_PROJECTS = 5

def build_digest() -> dict:
    """Gathers the handful of things a Resource Manager actually needs to act on --
    not a full data dump, just leave coverage gaps and fired project risk -- for the
    Friday EOD / Monday AM email digest. Same content works for either send; the
    router decides the subject line framing."""
    leave_rows = get_leave_impact()
    no_backfill = [r for r in leave_rows if not r["backfill_available"]]

    health_rows = get_health_report()
    high_risk = sorted(
        [r for r in health_rows if r["risk_band"] == "high"],
        key=lambda r: -r["risk_score"],
    )[:TOP_N_RISK_PROJECTS]

    return {
        "no_backfill_leaves": no_backfill,
        "no_backfill_count": len(no_backfill),
        "high_risk_projects": high_risk,
        "high_risk_total_count": sum(1 for r in health_rows if r["risk_band"] == "high"),
    }
