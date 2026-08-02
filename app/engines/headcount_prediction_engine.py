"""
Headcount Prediction engine -- rebuilt on real data.

Headcount, hires, resignations, and CoE mix are computed live from real
employee records (app/core/adapter.py's date_of_join/date_of_resignation/
account_status) -- not a synthetic dataset. Revenue and Adj. EBITDA margin
keep the real JMAN FY26 Townhall P&L anchors (REAL_LTM_REVENUE_GBP_000 /
REAL_EBITDA_MARGIN_PCT, originally sourced in
app/scripts/generate_headcount_prediction_data.py), fitted/extrapolated
against these same real calendar months -- but "revenue per head" now divides
by the REAL headcount for that month, not a synthetic series. Utilization
(free pool / over- / under-allocated) reuses the same real functions the
Allocations/Free Pool pages already use.

Real employee departure records are extremely sparse (14 rows total across
1050 employees, every one of them from 2026-05 onward) -- nowhere near enough
to train or validate a statistical forecasting model. The forecast here is
therefore a simple, explicitly low-confidence trailing-average net-change
extrapolation (same honesty pattern as demand_forecast_service's
get_financial_summary: a `low_confidence` flag + `sample_months`/`sample_size`
reported alongside the number), not a machine-learned model with a fabricated
accuracy claim.
"""

import numpy as np
import pandas as pd

from app.core.adapter import get_adapter
from app.engines.employee_coe import get_employee_primary_coe_map
from app.services.allocation_report_service import get_allocation_report
from app.services.free_pool_service import get_free_pool

HISTORY_MONTHS_BACK = 24
MAX_FORECAST_MONTHS = 12
FORECAST_TRAILING_MONTHS = 3
# A month's hires only get flagged as a likely data-migration artifact when
# BOTH: (1) the month's total is well above this org's own typical monthly
# volume (> 3x the real median across the window, floored at 20 so a thin
# window with a near-zero median doesn't make everything look anomalous),
# and (2) a single day accounts for most of that month's joins. Requiring
# both avoids flagging ordinary batch-onboarding months (e.g. a new-grad
# cohort starting the same day is normal and not itself anomalous).
ANOMALY_SHARE_THRESHOLD = 0.5
ANOMALY_MEDIAN_MULTIPLIER = 3.0
ANOMALY_MIN_JOINS = 20

NOT_DETERMINED_LABEL = "Not determined"

_LOCATION_LABEL = {"Chennai": "Chennai", "London": "UK", "New York": "USA"}

# Real trailing-12-month (LTM) revenue and Adj. EBITDA margin, GBP '000,
# reported for 2025-05 through 2026-05 (JMAN FY26 Global Townhall P&L deck) --
# see generate_headcount_prediction_data.py's "REAL DATA GROUNDING" docstring
# for the original sourcing. Months outside this real window are extrapolated
# from a quadratic fit through these anchors, not fabricated independently.
REAL_LTM_REVENUE_GBP_000 = {
    "2025-05-01": 27290, "2025-06-01": 27895, "2025-07-01": 28960, "2025-08-01": 30099,
    "2025-09-01": 31319, "2025-10-01": 32401, "2025-11-01": 33629, "2025-12-01": 34853,
    "2026-01-01": 36351, "2026-02-01": 38101, "2026-03-01": 39531, "2026-04-01": 40962,
    "2026-05-01": 42493,
}
REAL_EBITDA_MARGIN_PCT = {
    "2025-05-01": 27.9, "2025-06-01": 27.1, "2025-07-01": 27.7, "2025-08-01": 29.2,
    "2025-09-01": 30.8, "2025-10-01": 29.2, "2025-11-01": 28.8, "2025-12-01": 28.6,
    "2026-01-01": 29.1, "2026-02-01": 29.5, "2026-03-01": 29.9, "2026-04-01": 29.9,
    "2026-05-01": 29.8,
}


def _month_range(months_back: int, end: pd.Timestamp) -> list[pd.Timestamp]:
    last_month_start = end.to_period("M").to_timestamp()
    return list(pd.date_range(end=last_month_start, periods=months_back, freq="MS"))


def _fit_and_extrapolate(anchors: dict[str, float], months: list[pd.Timestamp]) -> np.ndarray:
    """Fits a quadratic trend through the real anchor points (keyed by
    month string) against `months`' index, evaluated at every month in the
    window -- extrapolating smoothly for months outside the anchors' real
    date range rather than leaving them undefined. Falls back to a flat
    mean if there aren't enough real anchors in range to fit a curve."""
    month_to_idx = {m.strftime("%Y-%m-01"): i for i, m in enumerate(months)}
    xs = np.array([month_to_idx[k] for k in anchors if k in month_to_idx], dtype=float)
    ys = np.array([v for k, v in anchors.items() if k in month_to_idx], dtype=float)
    if len(xs) < 3:
        return np.full(len(months), float(np.mean(ys)) if len(ys) else 0.0)
    coeffs = np.polyfit(xs, ys, deg=2)
    return np.polyval(coeffs, np.arange(len(months), dtype=float))


def _real_monthly_history(months_back: int = HISTORY_MONTHS_BACK) -> list[dict]:
    employees = get_adapter().get_employees()
    today = pd.Timestamp.now().normalize()
    months = _month_range(months_back, today)

    joins = pd.to_datetime(employees["date_of_join"], errors="coerce")
    resigns = pd.to_datetime(employees["date_of_resignation"], errors="coerce")
    locations = employees["location"]
    first_resignation = resigns.min()
    max_real_join_period = joins.max().to_period("M") if joins.notna().any() else None

    # First pass: real per-month hire counts, needed before we can tell which
    # months are real annual bulk/intern-batch intakes (a single big cohort
    # starting the same day -- real and expected, not a data error) vs the
    # steady month-to-month lateral (experienced-hire) pace.
    hires_masks = [joins.dt.to_period("M") == m.to_period("M") for m in months]
    new_hires_by_month = [int(mask.sum()) for mask in hires_masks]
    median_hires = float(np.median(new_hires_by_month)) if new_hires_by_month else 0.0
    bulk_threshold = max(ANOMALY_MEDIAN_MULTIPLIER * median_hires, ANOMALY_MIN_JOINS)

    bulk_periods = set()
    for i, m in enumerate(months):
        new_hires = new_hires_by_month[i]
        if new_hires > bulk_threshold:
            day_counts = joins[hires_masks[i]].value_counts()
            if len(day_counts) and day_counts.iloc[0] / new_hires >= ANOMALY_SHARE_THRESHOLD:
                bulk_periods.add(m.to_period("M"))

    # Real hire-date coverage genuinely ends at the last real join date -- every
    # month after that is structurally guaranteed to have zero real records (the
    # data just doesn't extend that far yet), not a coincidental quiet month, so
    # it's filled with a real-pattern-based estimate instead of a literal,
    # misleading zero. The estimate is the median STEADY lateral-hiring month
    # (bulk/intern-batch months excluded, since a batch intake day isn't
    # representative of month-to-month lateral hiring pace) over the real
    # trailing year before the gap started.
    gap_start_period = (max_real_join_period + 1) if max_real_join_period is not None else None
    steady_window = [
        new_hires_by_month[i] for i, m in enumerate(months)
        if max_real_join_period is not None
        and m.to_period("M") <= max_real_join_period
        and m.to_period("M") > max_real_join_period - 12
        and m.to_period("M") not in bulk_periods
    ]
    estimated_monthly_hires = round(float(np.median(steady_window))) if steady_window else 0

    location_totals = {label: 0 for label in _LOCATION_LABEL.values()}
    window_total = 0
    for i, m in enumerate(months):
        if (max_real_join_period is not None and m.to_period("M") <= max_real_join_period
                and m.to_period("M") > max_real_join_period - 12
                and m.to_period("M") not in bulk_periods):
            for real_loc, label in _LOCATION_LABEL.items():
                c = int(((locations == real_loc) & hires_masks[i]).sum())
                location_totals[label] += c
                window_total += c
    location_share = {label: (c / window_total) for label, c in location_totals.items() if window_total and c}

    history = []
    running_headcount = None
    for i, m in enumerate(months):
        is_current_month = i == len(months) - 1
        month_end = min(m.to_period("M").end_time.normalize(), today) if is_current_month else m.to_period("M").end_time.normalize()
        period = m.to_period("M")
        is_gap_month = gap_start_period is not None and period >= gap_start_period

        resign_mask = resigns.dt.to_period("M") == period
        resignations = int(resign_mask.sum())
        departures_data_reliable = bool(pd.notna(first_resignation) and m >= first_resignation.to_period("M").to_timestamp())

        hires_estimated = False
        note = None
        if is_gap_month:
            new_hires = estimated_monthly_hires
            hires_by_location = {label: round(estimated_monthly_hires * share) for label, share in location_share.items()}
            hires_estimated = True
            note = (
                f"No real join-date records exist past {max_real_join_period} -- {new_hires} is an estimate "
                "(median real steady/lateral hiring month over the prior year, excluding real annual bulk/intern "
                "intakes), not an actual record."
            )
        else:
            hires_mask = hires_masks[i]
            new_hires = new_hires_by_month[i]
            hires_by_location = {}
            if new_hires > 0:
                for real_loc, label in _LOCATION_LABEL.items():
                    count = int(((locations == real_loc) & hires_mask).sum())
                    if count:
                        hires_by_location[label] = hires_by_location.get(label, 0) + count
                # A real employee record can have a blank/unrecognized location
                # (a genuine gap in the source data, not every hire elsewhere on
                # this page has one) -- surfaced as its own honest bucket so the
                # location breakdown always reconciles to the real new_hires
                # count instead of silently summing to less than it.
                unattributed = new_hires - sum(hires_by_location.values())
                if unattributed > 0:
                    hires_by_location["Unknown"] = unattributed
            if period in bulk_periods:
                day_counts = joins[hires_mask].value_counts()
                top_count = int(day_counts.iloc[0])
                note = (
                    f"{top_count} of this month's {new_hires} joins share a single date "
                    f"({day_counts.index[0].strftime('%Y-%m-%d')}) -- a real annual bulk/intern-batch intake, not "
                    "day-by-day lateral hiring. Excluded from the steady-hiring baseline used elsewhere on this page."
                )

        # Headcount: real snapshot filter (exact per-employee join/resignation
        # dates) for real months -- most accurate. Rolled forward from the last
        # real month for gap months, since there's no real join-date roster left
        # to filter against past that point.
        if not is_gap_month:
            active_mask = joins.notna() & (joins <= month_end) & (resigns.isna() | (resigns > month_end))
            total_active = int(active_mask.sum())
            running_headcount = total_active
        else:
            running_headcount = max(0, running_headcount + new_hires - resignations)
            total_active = running_headcount

        history.append({
            "month": m.strftime("%Y-%m"),
            "total_active_headcount": total_active,
            "new_hires": new_hires,
            "resignations": resignations,
            "net": new_hires - resignations,
            "hires_by_location": hires_by_location,
            "hires_estimated": hires_estimated,
            "departures_data_reliable": departures_data_reliable,
            "note": note,
        })
    return history


def _real_coe_breakdown() -> dict:
    employees = get_adapter().get_employees()
    active = employees[employees["account_status"] == 1]
    coe_map = get_employee_primary_coe_map()
    coe_series = active["employee_id"].map(coe_map).fillna(NOT_DETERMINED_LABEL)
    counts = coe_series.value_counts()
    total = int(counts.sum()) or 1
    mix = [
        {"coe": coe, "headcount": int(n), "share_pct": round(n / total * 100, 1)}
        for coe, n in counts.items()
    ]
    mix.sort(key=lambda r: -r["headcount"])
    return {"mix": mix, "total_active_headcount": total}


def _real_utilization_snapshot() -> dict:
    report = get_allocation_report()
    if report:
        by_employee = pd.DataFrame(report).drop_duplicates("employee_id")
        over_allocated = int((by_employee["utilization_band"] == "over_allocated").sum())
        under_allocated = int((by_employee["utilization_band"] == "under_utilized").sum())
    else:
        over_allocated = under_allocated = 0
    free_pool = len(get_free_pool(include_redeploy_summary=False))
    return {
        "free_pool_current": free_pool,
        "over_allocated_current": over_allocated,
        "under_allocated_current": under_allocated,
    }


def _real_trailing_forecast(history: list[dict], horizon_months: int) -> list[dict]:
    reliable = [h for h in history if h["departures_data_reliable"]]
    sample = reliable[-FORECAST_TRAILING_MONTHS:]
    sample_months = len(sample)
    sample_nets = [h["net"] for h in sample]
    avg_net = (sum(sample_nets) / sample_months) if sample_months else 0.0
    # Hires/resignations decomposed separately (not just net) so the Attrition
    # & Retention chart can show a predicted continuation of EACH line, not
    # just the combined headcount effect.
    avg_hires = (sum(h["new_hires"] for h in sample) / sample_months) if sample_months else 0.0
    avg_resignations = (sum(h["resignations"] for h in sample) / sample_months) if sample_months else 0.0
    # Predicted location split for the forecast months -- reuses the SAME
    # trailing sample's real hires_by_location breakdown (already real-location-
    # proportioned for gap months, see _real_monthly_history), not a separately
    # invented distribution, so it's consistent with forecast_new_hires above.
    location_totals: dict[str, int] = {}
    for h in sample:
        for loc, cnt in (h.get("hires_by_location") or {}).items():
            location_totals[loc] = location_totals.get(loc, 0) + cnt
    location_total_sum = sum(location_totals.values())
    forecast_location_share = (
        {loc: cnt / location_total_sum for loc, cnt in location_totals.items()} if location_total_sum else {}
    )
    # Range = real observed spread of the SAME trailing months feeding the point
    # forecast (worst vs. best real net-change month), not a fabricated
    # statistical confidence interval -- with only a handful of real months to
    # go on, a made-up +/-X% band would just be a new invented number. min==max
    # (a zero-width band) when there's only one real month to compare against,
    # which is honest: no real variability has been observed yet.
    min_net = min(sample_nets) if sample_nets else 0.0
    max_net = max(sample_nets) if sample_nets else 0.0

    last_month = pd.Period(history[-1]["month"], freq="M")
    start_headcount = float(history[-1]["total_active_headcount"])
    running = start_headcount
    running_lower = start_headcount
    running_upper = start_headcount

    forecast = []
    for h in range(1, horizon_months + 1):
        running = max(0.0, running + avg_net)
        running_lower = max(0.0, running_lower + min_net)
        running_upper = max(0.0, running_upper + max_net)
        target_month = (last_month + h).to_timestamp()
        forecast.append({
            "month": target_month.strftime("%Y-%m"),
            "forecast": round(running, 1),
            "lower": round(min(running_lower, running_upper), 1),
            "upper": round(max(running_lower, running_upper), 1),
            "sample_months": sample_months,
            "low_confidence": sample_months < FORECAST_TRAILING_MONTHS,
            "forecast_new_hires": round(avg_hires, 1),
            "forecast_resignations": round(avg_resignations, 1),
            "forecast_hires_by_location": {
                loc: round(avg_hires * share) for loc, share in forecast_location_share.items()
            },
        })
    return forecast


def _productivity_series(history: list[dict], forecast: list[dict]) -> list[dict]:
    """Real revenue-per-head/EBITDA-margin for every history month, PLUS a
    predicted continuation for every forecast month -- the same real-anchored
    revenue/margin trend (see REAL_LTM_REVENUE_GBP_000/REAL_EBITDA_MARGIN_PCT)
    extended forward and divided by the FORECASTED headcount for that month,
    instead of stopping at "today" with no forward view at all."""
    all_months = (
        [pd.Period(h["month"], freq="M").to_timestamp() for h in history]
        + [pd.Period(f["month"], freq="M").to_timestamp() for f in forecast]
    )
    revenue_ltm_gbp_000 = _fit_and_extrapolate(REAL_LTM_REVENUE_GBP_000, all_months)
    ebitda_margin_pct = _fit_and_extrapolate(REAL_EBITDA_MARGIN_PCT, all_months)
    # LTM figure -> a monthly run-rate proxy (same real-anchored approach the
    # original grounding used) -- not an exact deconvolution of the reported
    # trailing-12-month sums back into individual months, which is
    # under-determined from a 13-point LTM series alone.
    revenue_monthly_gbp = revenue_ltm_gbp_000 * 1000 / 12

    out = []
    for i, h in enumerate(history):
        headcount = h["total_active_headcount"] or 1
        month_key = all_months[i].strftime("%Y-%m-01")
        out.append({
            "month": h["month"],
            "revenue_per_head_gbp": round(float(revenue_monthly_gbp[i]) / headcount, 0),
            "ebitda_margin_pct": round(float(ebitda_margin_pct[i]), 1),
            "revenue_ltm_gbp_000": round(float(revenue_ltm_gbp_000[i]), 0),
            "headcount": headcount,
            "is_real_revenue_anchor": month_key in REAL_LTM_REVENUE_GBP_000,
            "is_forecast": False,
        })
    for j, f in enumerate(forecast):
        i = len(history) + j
        headcount = f["forecast"] or 1
        month_key = all_months[i].strftime("%Y-%m-01")
        out.append({
            "month": f["month"],
            "revenue_per_head_gbp": round(float(revenue_monthly_gbp[i]) / headcount, 0),
            "ebitda_margin_pct": round(float(ebitda_margin_pct[i]), 1),
            "revenue_ltm_gbp_000": round(float(revenue_ltm_gbp_000[i]), 0),
            "headcount": headcount,
            "is_real_revenue_anchor": month_key in REAL_LTM_REVENUE_GBP_000,
            "is_forecast": True,
        })
    return out


def _compute_insights(history: list[dict], forecast: list[dict], coe: dict, utilization: dict) -> dict:
    latest = history[-1]
    lookback_idx = max(0, len(history) - 4)  # ~3 months back
    prior = history[lookback_idx]

    headcount_change_pct = (
        round((latest["total_active_headcount"] - prior["total_active_headcount"]) / prior["total_active_headcount"] * 100, 1)
        if prior["total_active_headcount"] else None
    )

    # End of the REQUESTED horizon (forecast has exactly horizon_months entries),
    # not a hardcoded 3rd month -- so switching the page's 3M/6M/12M selector
    # actually changes what these cards/panels show, instead of always freezing
    # on a fixed 3-month-ahead point regardless of the selected horizon.
    forecast_end = forecast[-1] if forecast else None
    forecast_change_pct = (
        round((forecast_end["forecast"] - latest["total_active_headcount"]) / latest["total_active_headcount"] * 100, 1)
        if forecast_end and latest["total_active_headcount"] else None
    )

    productivity_series = _productivity_series(history, forecast)
    productivity_history = productivity_series[:len(history)]
    productivity_forecast = productivity_series[len(history):]
    current_productivity = productivity_history[-1]
    prior_productivity = productivity_history[lookback_idx]
    productivity_end = productivity_forecast[-1] if productivity_forecast else None

    hires_vs_resignations = [
        {"month": h["month"], "new_hires": h["new_hires"], "resignations": h["resignations"], "net": h["net"]}
        for h in history[-6:]
    ]
    # Predicted continuation of the SAME chart -- same trailing-average hires/
    # resignations feeding the headcount forecast above, extended across the
    # full requested horizon (not just history), so Attrition & Retention also
    # responds to the 3M/6M/12M selector instead of only ever showing real
    # history.
    hires_vs_resignations_forecast = [
        {
            "month": f["month"],
            "new_hires": f["forecast_new_hires"],
            "resignations": f["forecast_resignations"],
            "net": round(f["forecast_new_hires"] - f["forecast_resignations"], 1),
        }
        for f in forecast
    ]

    risk_flags = []
    if latest["net"] < 0:
        risk_flags.append({
            "severity": "warning",
            "message": f"Net hiring flow is negative this month ({latest['new_hires']} hires vs {latest['resignations']} resignations) -- attrition is currently outpacing hiring.",
        })
    if forecast_change_pct is not None and forecast_change_pct < -2:
        risk_flags.append({
            "severity": "warning",
            "message": f"Headcount is forecast to decline {abs(forecast_change_pct)}% over the forecast horizon.",
        })
    if forecast and forecast[0]["low_confidence"]:
        risk_flags.append({
            "severity": "info",
            "message": (
                f"This forecast is a trailing-{FORECAST_TRAILING_MONTHS}-month average extrapolation, based on only "
                f"{forecast[0]['sample_months']} real month(s) of departure data (real resignation records only start "
                "2026-05) -- treat it as a rough scenario, not a validated prediction."
            ),
        })
    # Scan the FULL displayed history, not just the most recent months -- an
    # anomaly like the Feb-2026 bulk-migration spike sits well before "today"
    # but still visibly distorts every chart on this page (headcount, revenue
    # per head) for months after it, so it needs to stay flagged for as long
    # as it's still on screen, not just while it's recent.
    for h in history:
        if h["note"]:
            risk_flags.append({"severity": "info", "message": f"{h['month']}: {h['note']}"})

    executive_summary = []
    if headcount_change_pct is not None:
        direction = "grown" if headcount_change_pct > 0 else "declined" if headcount_change_pct < 0 else "held steady"
        executive_summary.append(
            f"Headcount has {direction} {abs(headcount_change_pct)}% over the last 3 months "
            f"({prior['total_active_headcount']} → {latest['total_active_headcount']})."
        )
    if forecast_end:
        forecast_word = "grow" if (forecast_change_pct or 0) > 0 else "decline" if (forecast_change_pct or 0) < 0 else "hold steady"
        executive_summary.append(
            f"Based on the last {forecast_end['sample_months']} real month(s) of net hiring, headcount is projected to "
            f"{forecast_word} to ~{round(forecast_end['forecast'])} by {forecast_end['month']} -- "
            f"{'a low-confidence extrapolation' if forecast_end['low_confidence'] else 'a trailing-average extrapolation'}, not a validated model."
        )
    rev_trend = "up" if current_productivity["revenue_per_head_gbp"] > prior_productivity["revenue_per_head_gbp"] else "down" if current_productivity["revenue_per_head_gbp"] < prior_productivity["revenue_per_head_gbp"] else "flat"
    executive_summary.append(
        f"Revenue per active headcount is currently ~£{current_productivity['revenue_per_head_gbp']:,.0f}/month, "
        f"{rev_trend} from ~£{prior_productivity['revenue_per_head_gbp']:,.0f}/month three months ago."
    )
    margin_trend = "up" if current_productivity["ebitda_margin_pct"] > prior_productivity["ebitda_margin_pct"] else "down" if current_productivity["ebitda_margin_pct"] < prior_productivity["ebitda_margin_pct"] else "flat"
    executive_summary.append(
        f"Adj. EBITDA margin is currently {current_productivity['ebitda_margin_pct']:.1f}%, "
        f"{margin_trend} from {prior_productivity['ebitda_margin_pct']:.1f}% three months ago."
    )
    executive_summary.append(
        f"Net hiring this month: {'+' if latest['net'] >= 0 else ''}{latest['net']} "
        f"({latest['new_hires']}{' (estimated)' if latest['hires_estimated'] else ''} hires vs {latest['resignations']} resignations)."
    )
    executive_summary.append(
        f"{utilization['over_allocated_current']} employee(s) are over-allocated, {utilization['under_allocated_current']} "
        f"under-allocated, and {utilization['free_pool_current']} are in the free pool right now."
    )

    return {
        "executive_summary": executive_summary,
        "risk_flags": risk_flags,
        "headcount_change_pct_3mo": headcount_change_pct,
        "forecast_change_pct": forecast_change_pct,
        "productivity": {
            "current_revenue_per_head_gbp": current_productivity["revenue_per_head_gbp"],
            "predicted_revenue_per_head_gbp_forecast": productivity_end["revenue_per_head_gbp"] if productivity_end else None,
            "history": [
                {
                    "month": p["month"], "value": p["revenue_per_head_gbp"],
                    "revenue_ltm_gbp_000": p["revenue_ltm_gbp_000"], "headcount": p["headcount"],
                    "is_real_revenue_anchor": p["is_real_revenue_anchor"],
                }
                for p in productivity_history
            ],
            "forecast": [
                {
                    "month": p["month"], "value": p["revenue_per_head_gbp"],
                    "revenue_ltm_gbp_000": p["revenue_ltm_gbp_000"], "headcount": p["headcount"],
                    "is_real_revenue_anchor": p["is_real_revenue_anchor"],
                }
                for p in productivity_forecast
            ],
            "current_ebitda_margin_pct": current_productivity["ebitda_margin_pct"],
            "predicted_ebitda_margin_pct_forecast": productivity_end["ebitda_margin_pct"] if productivity_end else None,
            "ebitda_margin_history": [{"month": p["month"], "value": p["ebitda_margin_pct"]} for p in productivity_history],
            "ebitda_margin_forecast": [{"month": p["month"], "value": p["ebitda_margin_pct"]} for p in productivity_forecast],
        },
        "coe_breakdown": {
            "latest_month": latest["month"],
            "mix": coe["mix"],
        },
        "attrition": {
            "hires_vs_resignations": hires_vs_resignations,
            "hires_vs_resignations_forecast": hires_vs_resignations_forecast,
        },
        "utilization": {
            "free_pool_current": utilization["free_pool_current"],
            "over_allocated_current": utilization["over_allocated_current"],
            "under_allocated_current": utilization["under_allocated_current"],
            "history": [{"month": h["month"], "free_pool": None, "over_allocated": None, "under_allocated": None} for h in history[-6:]],
        },
    }


def get_headcount_prediction(horizon_months: int = 12) -> dict:
    horizon_months = max(1, min(horizon_months, MAX_FORECAST_MONTHS))

    history = _real_monthly_history()
    coe = _real_coe_breakdown()
    utilization = _real_utilization_snapshot()
    forecast = _real_trailing_forecast(history, horizon_months)
    insights = _compute_insights(history, forecast, coe, utilization)

    sample_months = forecast[0]["sample_months"] if forecast else 0

    return {
        "history": [
            {
                "month": h["month"],
                "total_active_headcount": h["total_active_headcount"],
                "new_hires": h["new_hires"],
                "resignations": h["resignations"],
                "hires_by_location": h["hires_by_location"],
                "hires_estimated": h["hires_estimated"],
                "note": h["note"],
            }
            for h in history
        ],
        "training_period": f"{history[0]['month']} → {history[-1]['month']}",
        "horizon_months": horizon_months,
        "forecast": forecast,
        "insights": insights,
        "model_info": {
            "type": f"Trailing {FORECAST_TRAILING_MONTHS}-month average net-change extrapolation",
            "sample_months": sample_months,
            "low_confidence": sample_months < FORECAST_TRAILING_MONTHS,
            "trained_on": "Real employee join/resignation dates (app/core/adapter.py) -- no synthetic data.",
            "note": (
                "Real resignation records only exist from 2026-05 onward (14 rows total across 1050 employees) -- "
                "far too little history to fit or validate a statistical model. This forecast is a plain trailing-"
                f"average of real net hiring over the last {sample_months} month(s) with real departure data, held "
                "constant forward. Revenue and EBITDA margin are grounded in real JMAN FY26 P&L figures (fitted/"
                "extrapolated), divided by the real headcount above -- not a synthetic denominator."
            ),
        },
    }


def simulate_headcount_prediction(edited_history: list[dict], horizon_months: int = 12) -> dict:
    """Re-runs the forecast/insights on resource-manager-edited monthly-history
    rows instead of the live real data -- a what-if scenario tool. CoE mix and
    utilization stay live/real (editing those isn't in scope here); only the
    hires/resignations/headcount driving the forecast are user-supplied."""
    horizon_months = max(1, min(horizon_months, MAX_FORECAST_MONTHS))

    history = []
    for row in edited_history:
        new_hires = int(row.get("new_hires") or 0)
        resignations = int(row.get("resignations") or 0)
        history.append({
            "month": row["month"],
            "total_active_headcount": int(row.get("total_active_headcount") or 0),
            "new_hires": new_hires,
            "resignations": resignations,
            "net": new_hires - resignations,
            "hires_by_location": row.get("hires_by_location") or {},
            "hires_estimated": bool(row.get("hires_estimated", False)),
            "departures_data_reliable": bool(row.get("departures_data_reliable", True)),
            "note": row.get("note"),
        })
    history.sort(key=lambda r: r["month"])
    if not history:
        raise ValueError("No history rows supplied to simulate.")

    coe = _real_coe_breakdown()
    utilization = _real_utilization_snapshot()
    forecast = _real_trailing_forecast(history, horizon_months)
    insights = _compute_insights(history, forecast, coe, utilization)
    sample_months = forecast[0]["sample_months"] if forecast else 0

    return {
        "history": [
            {
                "month": h["month"],
                "total_active_headcount": h["total_active_headcount"],
                "new_hires": h["new_hires"],
                "resignations": h["resignations"],
                "hires_by_location": h["hires_by_location"],
                "hires_estimated": h["hires_estimated"],
                "note": h["note"],
            }
            for h in history
        ],
        "training_period": f"{history[0]['month']} → {history[-1]['month']}",
        "horizon_months": horizon_months,
        "forecast": forecast,
        "insights": insights,
        "model_info": {
            "type": f"Trailing {FORECAST_TRAILING_MONTHS}-month average net-change extrapolation (simulated scenario)",
            "sample_months": sample_months,
            "low_confidence": sample_months < FORECAST_TRAILING_MONTHS,
            "trained_on": "Resource-manager-edited monthly history -- a what-if scenario, not the live real dataset.",
            "note": (
                "This result reflects manually edited hires/resignations/headcount values, not live real data. "
                "Reset to see the forecast computed from the real dataset."
            ),
        },
    }


def list_raw_tables() -> list[dict]:
    return [
        {"table": "monthly_history", "label": "Real Monthly History", "description": "Real headcount, hires, resignations, and location breakdown per month, computed from employee join/resignation dates."},
        {"table": "coe_breakdown", "label": "Real CoE Breakdown", "description": "Current active headcount tallied by real CoE assignment (from observed skills data), including a real 'Not determined' bucket."},
        {"table": "forecast", "label": "Forecast", "description": "Trailing-average net-change extrapolation of headcount, hires, and resignations, with real sample size and low-confidence flag."},
        {"table": "productivity", "label": "Real Revenue & Margin", "description": "Revenue-per-head and Adj. EBITDA margin per month -- is_real_revenue_anchor marks months matched to the real JMAN FY26 Townhall P&L; all other months (including every forecast month) are extrapolated from that real trend, not independently fabricated."},
    ]


def get_raw_table(table: str) -> dict:
    history = _real_monthly_history()
    if table == "monthly_history":
        rows = history
    elif table == "coe_breakdown":
        rows = _real_coe_breakdown()["mix"]
    elif table == "forecast":
        rows = _real_trailing_forecast(history, MAX_FORECAST_MONTHS)
    elif table == "productivity":
        forecast = _real_trailing_forecast(history, MAX_FORECAST_MONTHS)
        rows = _productivity_series(history, forecast)
    else:
        raise ValueError(f"Unknown table {table!r}. Valid options: monthly_history, coe_breakdown, forecast, productivity")
    spec = next(t for t in list_raw_tables() if t["table"] == table)
    return {**spec, "columns": list(rows[0].keys()) if rows else [], "rows": rows, "row_count": len(rows)}
