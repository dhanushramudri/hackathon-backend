"""
Headcount Prediction — Phase 1: synthetic data generation.

Produces three CSVs under data/HeadcountPrediction/:
  - monthly_snapshot.csv          (24 rows, one per month)
  - notice_period_cohort.csv      (one row per employee per month while under notice)
  - weekly_pulse_monthly_agg.csv  (24 rows, one per month)

This is 100% synthetic — no real employee/project/financial data is used or
referenced. It exists to prototype the Headcount Prediction feature's data
pipeline before real historical data is available.

ASSUMPTIONS RESOLVED (see plan's "Open questions" — proceeding per the user's
instruction to fill any gap with dummy data rather than block on these):
  1. Training window: a ROLLING 24 months ending at the app's current
     fictional "today" (see TRAINING_WINDOW_END below), not the fixed
     Jan 2023-Dec 2024 window originally proposed. That fixed window meant
     the model's forecast projected into 2025 while the app's "today" was
     mid-2026 -- already 19 months in the past by the time anyone looked at
     it, and the chart's "Today" reference line was mislabeling a date over
     a year stale. Rolling the window to end at "today" makes the forecast
     actually project forward from now, which is what a Resource Manager
     opening this page needs.
  2. "Cluster" = proposition_coe groupings (Core Reporting, Exit Support,
     Managed Services, Value Creation, Data Advisory, Due Diligence) — the
     only cluster-like real dimension in the existing schema.
  3. Pulse overall_score = mean(q1..q5) on the real 1-4 scale found in
     weekly_pulse_dummy_question_legend.csv / _answer_scale_legend.csv
     (that field doesn't exist in the source data, so it's derived here).
  4. Client location mix per project is a PROXY built from the real
     employee-location skew (Chennai/UK/USA), since no real per-project
     client-geography field exists.
  5. Since this script has no clock access at import time that the rest of
     the app can share (and re-running the generator monthly to keep the
     window rolling isn't wired up yet), TRAINING_WINDOW_END is a literal
     constant that must be bumped forward periodically by whoever maintains
     this feature -- it is NOT re-derived from today's real date automatically.

Run:  python -m app.scripts.generate_headcount_prediction_data
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = BACKEND_ROOT / "data" / "HeadcountPrediction"

# Last month of REAL synthetic history -- the forecast (produced by the
# Phase 5 engine) starts the month after this. Keep this at (or just before)
# the app's current fictional date so "Headcount Prediction" actually means
# "from today forward," not a stale historical backtest.
TRAINING_WINDOW_END = "2026-07-01"
MONTHS = pd.date_range(end=TRAINING_WINDOW_END, periods=24, freq="MS")
N = len(MONTHS)  # 24

# ── Reference distributions, pulled from the real dataset's actual skew ────
# (see plan grounding: 673 active employees, Chennai/London/New York split,
# canonical COE map and proposition_coe shares from role_mix_engine.py /
# 02_Project_Details_clean.csv)
COES = ["data_engineering", "bi_reporting", "ai_ml", "full_stack", "techops"]
COE_SHARE = {"data_engineering": 0.32, "bi_reporting": 0.28, "ai_ml": 0.13, "full_stack": 0.12, "techops": 0.10}
# Remaining ~5% of company totals is "other/unmapped" (real data has ~60% of
# projects with no tech_coe at all) and is not broken into its own column.

CLUSTERS = ["core_reporting", "exit_support", "managed_services", "value_creation", "data_advisory", "due_diligence"]
CLUSTER_SHARE = {
    "core_reporting": 0.45, "exit_support": 0.15, "managed_services": 0.15,
    "value_creation": 0.14, "data_advisory": 0.06, "due_diligence": 0.03,
}
# Remaining ~2% is "other" clusters (Pricing, PE Services, etc. in the real
# data) and is absorbed into the total rather than broken out.

ROLES = [
    "Trainee Software Engineer", "Senior Software Engineer", "Software Engineer",
    "Solutions Enabler", "Associate Consultant", "Solutions Consultant",
    "Senior Associate Consultant", "Consultant", "Senior Consultant", "Manager",
]
ROLE_WEIGHTS = np.array([0.24, 0.17, 0.11, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03])
ROLE_WEIGHTS = ROLE_WEIGHTS / ROLE_WEIGHTS.sum()

LOCATIONS = ["chennai", "uk", "usa"]
LOCATION_SHARE = {"chennai": 0.72, "uk": 0.23, "usa": 0.05}  # matches real 520/164/39 split
CLIENT_LOCATION_SHARE = {"chennai": 0.55, "uk": 0.25, "usa": 0.10, "other": 0.10}  # proxy, see assumption #4

# Real UK bank-holiday counts per calendar month (same every year, approx.)
UK_HOLIDAYS_BY_MONTH = {1: 1, 2: 0, 3: 0, 4: 2, 5: 2, 6: 0, 7: 0, 8: 1, 9: 0, 10: 0, 11: 0, 12: 2}

RATE_PER_FTE_MONTH_USD = {
    # Blended $/FTE/month per COE, derived from rate_card_service.py bands
    # (AI/ML and Full Stack skew to higher-band roles; BI/Reporting and
    # TechOps skew to mid/lower bands) at ~160h/month.
    "data_engineering": 10_400, "bi_reporting": 9_600, "ai_ml": 12_800,
    "full_stack": 11_200, "techops": 8_800,
}

STANDARD_MONTHLY_HOURS = 160


# ── Seasonal shape ──────────────────────────────────────────────────────────
# UK fiscal-year pattern (high Apr-Jun, low Dec-Jan) + explicit Feb-Mar new-
# project intake peak + explicit December dip. NOT a linear trend -- this is
# the multiplicative seasonal component in a trend x seasonal x noise model.
SEASONAL_FACTOR = {
    1: 0.80, 2: 1.16, 3: 1.18, 4: 1.14, 5: 1.16, 6: 1.10,
    7: 0.96, 8: 0.92, 9: 1.00, 10: 1.02, 11: 1.00, 12: 0.70,
}

# HR-calendar pattern is DIFFERENT from the revenue/ops seasonal curve above --
# promotions and resignations cluster around bi-annual appraisal cycles
# (Apr, Oct), not the fiscal revenue curve.
HR_CYCLE_FACTOR = {m: (1.8 if m in (4, 10) else 0.9) for m in range(1, 13)}

# FTE/headcount are STOCK variables (actual people currently staffed) and
# can't swing with the same ~±20% amplitude as flow variables like revenue
# or new-project counts -- real staffing levels move gradually. Dampened to
# ~20% of the flow-variable amplitude so seasonality shows up as a gentle
# wave in FTE, not a near-doubling month to month.
FTE_SEASONAL_FACTOR = {m: 1 + (f - 1) * 0.2 for m, f in SEASONAL_FACTOR.items()}

# One synthetic "shock" event: a large new client lands a few months into the
# window, causing a temporary bump in new projects / hires / revenue for a
# few months, with plenty of runway afterward for the series to settle back
# down before the training window ends.
SHOCK_MONTHS = pd.date_range("2024-11-01", "2025-01-01", freq="MS")
SHOCK_MULTIPLIER = {0: 1.9, 1: 1.6, 2: 1.25}  # tapering off over 3 months


def is_shock_month(month: pd.Timestamp) -> float:
    for i, sm in enumerate(SHOCK_MONTHS):
        if sm == month:
            return SHOCK_MULTIPLIER[i]
    return 1.0


def add_noise(values: np.ndarray, pct_std: float | None = None) -> np.ndarray:
    """Gaussian noise with std = 8-15% of the (per-call) mean, per the brief."""
    if pct_std is None:
        pct_std = rng.uniform(0.08, 0.15)
    mean_abs = np.mean(np.abs(values)) or 1.0
    noise = rng.normal(0, mean_abs * pct_std, size=len(values))
    return values + noise


def random_walk(n: int, start: float, step_std: float, drift: float = 0.0) -> np.ndarray:
    steps = rng.normal(drift, step_std, size=n)
    return start + np.cumsum(steps)


def month_key(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-01")


# ── Bottom-up HR flow (hires/resignations -> headcount), not an independent
# random headcount series -- keeps total_active_headcount internally
# consistent with the hires/resignations columns instead of two unrelated
# random draws that happen to be shown side by side. ──────────────────────
def build_hr_flow() -> dict[str, np.ndarray]:
    base_hires = 13.0
    base_resign = 8.4  # ~15% annualized attrition on a ~670-head base

    hires = np.array([
        base_hires * HR_CYCLE_FACTOR.get(9 if m.month == 9 else m.month, 1.0) * is_shock_month(m)
        for m in MONTHS
    ], dtype=float)
    # Small extra bump every September (grad-hire intake), independent of the
    # bi-annual promotion/resignation cycle.
    hires = np.array([h * (1.3 if m.month == 9 else 1.0) for h, m in zip(hires, MONTHS)])
    hires = np.clip(add_noise(hires), 3, None)

    resign = np.array([base_resign * HR_CYCLE_FACTOR[m.month] for m in MONTHS], dtype=float)
    resign = np.clip(add_noise(resign), 2, None)

    promotions = np.array([
        (11.0 if m.month in (4, 10) else 5.0) for m in MONTHS
    ], dtype=float)
    promotions = np.clip(add_noise(promotions), 1, None)

    headcount = np.zeros(N)
    headcount[0] = 555.0  # plausible Jan-2023 starting headcount for a ~670-head-by-"now" org
    for i in range(1, N):
        headcount[i] = headcount[i - 1] + hires[i] - resign[i]

    return {
        "new_hires_total": np.round(hires).astype(int),
        "resignations_total": np.round(resign).astype(int),
        "promotions_total": np.round(promotions).astype(int),
        "total_active_headcount": np.round(headcount).astype(int),
    }


def split_by_weights(total: np.ndarray, weights: dict[str, float]) -> dict[str, np.ndarray]:
    """Split a total series into weighted, independently-noised sub-series that
    still sum (approximately) back to the total -- noise is applied to the
    weight shares per month, not to each sub-series independently, so parts
    don't drift arbitrarily far from the whole."""
    keys = list(weights.keys())
    base = np.array([weights[k] for k in keys])
    out: dict[str, np.ndarray] = {}
    for i in range(len(total)):
        shares = np.clip(rng.normal(base, base * 0.06), 0.01, None)
        shares = shares / shares.sum()
        for k, s in zip(keys, shares):
            out.setdefault(k, np.zeros(len(total)))[i] = total[i] * s
    return out


def build_monthly_snapshot() -> pd.DataFrame:
    hr = build_hr_flow()
    headcount = hr["total_active_headcount"].astype(float)

    seasonal = np.array([SEASONAL_FACTOR[m.month] for m in MONTHS])
    seasonal_fte = np.array([FTE_SEASONAL_FACTOR[m.month] for m in MONTHS])
    seasonal_fte_mean = np.mean(list(FTE_SEASONAL_FACTOR.values()))
    shock = np.array([is_shock_month(m) for m in MONTHS])
    # New-client shock ramps revenue in gradually (staffing/billing lags the
    # deal by weeks) rather than an instant multiplier -- dampened to 40% of
    # the effect it has on new_projects_total.
    revenue_shock = 1.0 + (shock - 1.0) * 0.4

    # Billable FTE ~63% of active headcount, following the DAMPENED seasonal
    # curve (see FTE_SEASONAL_FACTOR) with a slow random-walk drift on top --
    # NOT a linear trend, and not the full-amplitude seasonal curve (that's
    # for flow variables like revenue, applied once below, not here too).
    billable_ratio = np.clip(0.63 + random_walk(N, 0, 0.004), 0.58, 0.70)
    billable_fte_total = add_noise(headcount * billable_ratio * seasonal_fte / seasonal_fte_mean, pct_std=0.06)
    unbillable_fte_total = np.clip(add_noise(headcount * 0.09 * (2 - seasonal_fte)), 20, None)

    total_logged_hours = add_noise((billable_fte_total + unbillable_fte_total) * STANDARD_MONTHLY_HOURS * rng.uniform(0.8, 0.92, N))
    utilization = np.clip(rng.normal(0.85, 0.05, N) * (seasonal_fte / seasonal_fte_mean), 0.65, 0.98)
    billable_hours = np.clip(billable_fte_total * STANDARD_MONTHLY_HOURS * utilization, 0, total_logged_hours)

    # Revenue: random-walk component (per the brief) layered under the FULL
    # seasonal shape (applied once, here -- not also inside billable_fte_total
    # above, which would double-count it) and the COE blended rate card, plus
    # the dampened shock effect.
    revenue_walk = random_walk(N, 0, 25_000)
    avg_rate = np.mean(list(RATE_PER_FTE_MONTH_USD.values()))
    revenue_usd_total = np.clip(
        billable_fte_total * avg_rate * (seasonal / seasonal.mean()) * revenue_shock + revenue_walk,
        1_500_000, None,
    )
    revenue_usd_total = add_noise(revenue_usd_total, pct_std=0.10)
    billable_share = np.clip(rng.normal(0.85, 0.03, N) - (1 - seasonal) * 0.1, 0.65, 0.93)
    revenue_usd_billable = revenue_usd_total * billable_share
    revenue_usd_unbillable = revenue_usd_total - revenue_usd_billable

    revenue_by_coe = split_by_weights(revenue_usd_total, COE_SHARE)
    revenue_by_cluster = split_by_weights(revenue_usd_total, CLUSTER_SHARE)
    fte_by_coe = split_by_weights(billable_fte_total, COE_SHARE)

    new_projects_total = np.clip(add_noise(np.array([9.0 for _ in MONTHS]) * seasonal * shock), 3, None)
    new_projects_by_coe = {k: np.round(v).astype(int) for k, v in split_by_weights(new_projects_total, COE_SHARE).items()}
    new_projects_by_cluster = {k: np.round(v).astype(int) for k, v in split_by_weights(new_projects_total, CLUSTER_SHARE).items()}

    extensions_billable = np.round(np.clip(add_noise(np.full(N, 4.2) * seasonal), 1, None)).astype(int)
    extensions_unbillable = np.round(np.clip(add_noise(np.full(N, 1.8)), 0, None)).astype(int)

    completions_total = np.round(np.clip(add_noise(np.full(N, 7.0)), 2, None)).astype(int)
    completions_on_time = np.round(completions_total * np.clip(rng.normal(0.72, 0.06, N), 0.5, 0.9)).astype(int)

    new_phases_started = np.round(np.clip(add_noise(np.full(N, 6.0) * seasonal), 2, None)).astype(int)

    over_allocated_count = np.round(np.clip(add_noise(headcount * 0.055 * seasonal), 10, None)).astype(int)
    under_allocated_count = np.round(np.clip(add_noise(headcount * 0.09 * (2 - seasonal)), 20, None)).astype(int)
    escalations_total = np.round(np.clip(add_noise(2.5 + over_allocated_count * 0.06), 1, None)).astype(int)

    free_pool_headcount = np.round(np.clip(add_noise(headcount * 0.035 * (2 - seasonal)), 8, None)).astype(int)
    free_pool_avg_days = np.clip(add_noise(np.full(N, 20.0) * (2 - seasonal)), 8, 45)

    shadow_unbilled_count = np.round(np.clip(add_noise(headcount * 0.075), 15, None)).astype(int)
    shadow_unbilled_hours = np.clip(add_noise(shadow_unbilled_count * 22.0), 0, None)

    notice_period_headcount = np.round(np.clip(add_noise(hr["resignations_total"].astype(float) * 3.0, pct_std=0.10), 5, None)).astype(int)

    hires_by_role = [
        {r: int(round(h * w)) for r, w in zip(ROLES, rng.dirichlet(ROLE_WEIGHTS * 20))}
        for h in hr["new_hires_total"]
    ]
    resign_by_role = [
        {r: int(round(rs * w)) for r, w in zip(ROLES, rng.dirichlet(ROLE_WEIGHTS * 20))}
        for rs in hr["resignations_total"]
    ]

    rows = []
    for i, m in enumerate(MONTHS):
        row = {
            "month": month_key(m),
            "revenue_usd_total": round(float(revenue_usd_total[i]), 2),
            **{f"revenue_usd_{k}": round(float(v[i]), 2) for k, v in revenue_by_coe.items()},
            **{f"revenue_usd_{k}": round(float(v[i]), 2) for k, v in revenue_by_cluster.items()},
            "revenue_usd_billable": round(float(revenue_usd_billable[i]), 2),
            "revenue_usd_unbillable": round(float(revenue_usd_unbillable[i]), 2),
            "new_projects_total": int(round(new_projects_total[i])),
            **{f"new_projects_{k}": int(v[i]) for k, v in new_projects_by_coe.items()},
            **{f"new_projects_{k}": int(v[i]) for k, v in new_projects_by_cluster.items()},
            "extensions_billable": int(extensions_billable[i]),
            "extensions_unbillable": int(extensions_unbillable[i]),
            "completions_total": int(completions_total[i]),
            "completions_on_time": int(completions_on_time[i]),
            "new_phases_started": int(new_phases_started[i]),
            "escalations_total": int(escalations_total[i]),
            "client_location_chennai_pct": CLIENT_LOCATION_SHARE["chennai"] * 100,
            "client_location_uk_pct": CLIENT_LOCATION_SHARE["uk"] * 100,
            "client_location_usa_pct": CLIENT_LOCATION_SHARE["usa"] * 100,
            "client_location_other_pct": CLIENT_LOCATION_SHARE["other"] * 100,
            "new_hires_total": int(hr["new_hires_total"][i]),
            "new_hires_by_role_json": json.dumps(hires_by_role[i]),
            "resignations_total": int(hr["resignations_total"][i]),
            "resignations_by_role_json": json.dumps(resign_by_role[i]),
            "notice_period_headcount": int(notice_period_headcount[i]),
            "promotions_total": int(hr["promotions_total"][i]),
            "free_pool_headcount": int(free_pool_headcount[i]),
            "free_pool_avg_days": round(float(free_pool_avg_days[i]), 1),
            "over_allocated_count": int(over_allocated_count[i]),
            "under_allocated_count": int(under_allocated_count[i]),
            "shadow_unbilled_count": int(shadow_unbilled_count[i]),
            "shadow_unbilled_hours": round(float(shadow_unbilled_hours[i]), 1),
            "employee_location_chennai_pct": round(LOCATION_SHARE["chennai"] * 100 + rng.normal(0, 0.5), 1),
            "employee_location_uk_pct": round(LOCATION_SHARE["uk"] * 100 + rng.normal(0, 0.5), 1),
            "employee_location_usa_pct": round(LOCATION_SHARE["usa"] * 100 + rng.normal(0, 0.3), 1),
            "billable_fte_total": round(float(billable_fte_total[i]), 1),
            **{f"billable_fte_{k}": round(float(v[i]), 1) for k, v in fte_by_coe.items()},
            "unbillable_fte_total": round(float(unbillable_fte_total[i]), 1),
            "total_logged_hours": round(float(total_logged_hours[i]), 0),
            "billable_hours": round(float(billable_hours[i]), 0),
            "uk_public_holidays": UK_HOLIDAYS_BY_MONTH[m.month],
            "is_low_season": int(m.month in (12, 1)),
            "is_peak_season": int(m.month in (2, 3)),
            "is_month_end": 1,
            "is_quarter_end": int(m.month in (3, 6, 9, 12)),
            "total_active_headcount": int(hr["total_active_headcount"][i]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_notice_period_cohort(snapshot: pd.DataFrame, pulse: pd.DataFrame) -> pd.DataFrame:
    """One row per (synthetic employee, month) while under the 3-month notice
    window. Synthetic employee IDs are prefixed SYN_ to make it unambiguous
    this is not real employee data and can't collide with real EMP ids."""
    rows = []
    emp_counter = 0
    pulse_by_month = pulse.set_index("month")

    for i, m in enumerate(MONTHS):
        n_resigning = snapshot.iloc[i]["resignations_total"]
        for _ in range(int(n_resigning)):
            emp_counter += 1
            employee_id = f"SYN_EMP_{emp_counter:04d}"
            resignation_date = m + pd.Timedelta(days=int(rng.integers(0, 28)))
            notice_start = resignation_date - pd.DateOffset(months=3)
            role = rng.choice(ROLES, p=ROLE_WEIGHTS)
            coe = rng.choice(COES, p=[COE_SHARE[c] / sum(COE_SHARE.values()) for c in COES])
            location = rng.choice(LOCATIONS, p=[LOCATION_SHARE[l] for l in LOCATIONS])

            before_pulse = float(np.clip(rng.normal(3.05, 0.25), 1.5, 4.0))
            delta = float(np.clip(rng.normal(-0.55, 0.15), -1.2, -0.05))

            for months_in in (1, 2, 3):
                notice_month = notice_start + pd.DateOffset(months=months_in - 1)
                notice_month = notice_month.replace(day=1)
                if notice_month < MONTHS[0] or notice_month > MONTHS[-1]:
                    continue
                during_pulse = round(before_pulse + delta * (months_in / 3.0), 2)
                rows.append({
                    "employee_id": employee_id,
                    "resignation_date": resignation_date.strftime("%Y-%m-%d"),
                    "notice_start_date": notice_start.strftime("%Y-%m-%d"),
                    "month": month_key(notice_month),
                    "months_into_notice": months_in,
                    "role": role,
                    "coe": coe,
                    "location": location,
                    "pulse_score_current_month": during_pulse,
                    "pulse_score_3mo_before_notice": round(before_pulse, 2),
                })
    return pd.DataFrame(rows)


def build_weekly_pulse_monthly_agg(snapshot: pd.DataFrame, notice_cohort: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregate of the (synthetic) weekly pulse survey, using the
    REAL question schema found in weekly_pulse_dummy_question_legend.csv
    (q1_inspired_motivated .. q5_workload_sustainable, 1-4 scale)."""
    over_alloc = snapshot["over_allocated_count"].to_numpy(dtype=float)
    seasonal = np.array([SEASONAL_FACTOR[pd.Timestamp(m).month] for m in snapshot["month"]])
    headcount = snapshot["total_active_headcount"].to_numpy(dtype=float)

    strain = (over_alloc - over_alloc.mean()) / (over_alloc.std() or 1.0)

    avg_q1 = np.clip(3.05 - 0.10 * strain + rng.normal(0, 0.06, N) - (1 - seasonal) * 0.08, 2.4, 3.6)
    avg_q5 = np.clip(2.85 - 0.18 * strain + rng.normal(0, 0.07, N) - (1 - seasonal) * 0.10, 2.1, 3.4)

    overall = (avg_q1 + avg_q5) / 2.0  # rough proxy combining the two tracked axes
    pct_le_2 = np.clip(0.14 + 0.05 * strain + rng.normal(0, 0.015, N), 0.05, 0.30)

    n_respondents = np.round(headcount * rng.uniform(0.55, 0.75, N)).astype(int)
    response_rate = np.round(n_respondents / headcount * 100, 1)

    rows = []
    for i, m in enumerate(MONTHS):
        mk = month_key(m)
        cohort_month = notice_cohort[notice_cohort["month"] == mk]
        before = cohort_month["pulse_score_3mo_before_notice"].mean() if not cohort_month.empty else None
        during = cohort_month["pulse_score_current_month"].mean() if not cohort_month.empty else None
        delta = (during - before) if (before is not None and during is not None) else None

        rows.append({
            "month": mk,
            "avg_q1_motivation": round(float(avg_q1[i]), 2),
            "avg_q5_workload": round(float(avg_q5[i]), 2),
            "pct_employees_overall_le_2": round(float(pct_le_2[i]) * 100, 1),
            "n_respondents": int(n_respondents[i]),
            "response_rate_pct": float(response_rate[i]),
            "notice_cohort_avg_pulse_before": round(before, 2) if before is not None else None,
            "notice_cohort_avg_pulse_during": round(during, 2) if during is not None else None,
            "notice_cohort_pulse_delta": round(delta, 2) if delta is not None else None,
        })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = build_monthly_snapshot()

    # weekly_pulse_monthly_agg needs the notice cohort's pulse deltas, and the
    # notice cohort doesn't depend on pulse -- build cohort first with a
    # placeholder pulse frame, then the real pulse agg, in that order.
    notice_cohort = build_notice_period_cohort(snapshot, pulse=pd.DataFrame({"month": snapshot["month"]}))
    pulse_agg = build_weekly_pulse_monthly_agg(snapshot, notice_cohort)

    snapshot.to_csv(OUT_DIR / "monthly_snapshot.csv", index=False)
    notice_cohort.to_csv(OUT_DIR / "notice_period_cohort.csv", index=False)
    pulse_agg.to_csv(OUT_DIR / "weekly_pulse_monthly_agg.csv", index=False)

    print(f"Wrote {len(snapshot)} rows to monthly_snapshot.csv")
    print(f"Wrote {len(notice_cohort)} rows to notice_period_cohort.csv")
    print(f"Wrote {len(pulse_agg)} rows to weekly_pulse_monthly_agg.csv")
    print(f"Output dir: {OUT_DIR}")

    # Quick sanity summary
    print("\n--- Sanity checks ---")
    print("Headcount range:", snapshot["total_active_headcount"].min(), "->", snapshot["total_active_headcount"].max())
    print("Revenue range ($M):", round(snapshot["revenue_usd_total"].min() / 1e6, 2), "->", round(snapshot["revenue_usd_total"].max() / 1e6, 2))
    dec_rows = snapshot[pd.to_datetime(snapshot["month"]).dt.month == 12]
    print("December revenue (should dip):", dec_rows["revenue_usd_total"].round(0).tolist())
    febmar_rows = snapshot[pd.to_datetime(snapshot["month"]).dt.month.isin([2, 3])]
    print("Feb/Mar revenue (should peak):", febmar_rows["revenue_usd_total"].round(0).tolist())
    shock_month_keys = [m.strftime("%Y-%m-01") for m in SHOCK_MONTHS]
    print(f"{shock_month_keys[0]}..{shock_month_keys[-1]} new_projects (shock, should be elevated):",
          snapshot[snapshot["month"].isin(shock_month_keys)]["new_projects_total"].tolist())


if __name__ == "__main__":
    main()
