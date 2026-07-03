"""
Simple demand forecasting engine using synthetic historical data.
Model: Linear Trend + Seasonal Decomposition (pure numpy, no ML libraries).
Formula: Y(t) = a + b*t + seasonal(month)
Confidence interval: 90% (±1.64 * std of residuals).
"""

import numpy as np
import pandas as pd

# Seasonal factors derived from consulting industry patterns:
# Q4 (Oct-Dec) is high due to year-end budget pushes,
# Jan-Feb slow (new budgets not yet approved), Jul dip (summer).
SEASONAL_FACTORS = {
    1:  0.78,
    2:  0.82,
    3:  0.96,
    4:  1.05,
    5:  1.10,
    6:  1.00,
    7:  0.88,
    8:  0.95,
    9:  1.08,
    10: 1.18,
    11: 1.22,
    12: 1.08,
}

HISTORY_START = "2023-01"
HISTORY_END   = "2026-06"
MONTHLY_RATE_PER_HEAD = 14_500   # illustrative $/head/month


def _generate_synthetic_history() -> list[dict]:
    """
    42 months of synthetic monthly pipeline history (Jan 2023 – Jun 2026).
    Built from: base growth trend + seasonal factor + gaussian noise.
    Deterministic (seed=42) so the chart never jumps between page loads.
    """
    np.random.seed(42)
    months = pd.date_range(HISTORY_START, periods=42, freq="MS")
    n = len(months)

    records = []
    for i, month in enumerate(months):
        # Base demand: linear growth from 8 → 19 over 42 months
        base = 8.0 + (i / (n - 1)) * 11.0
        seasonal = SEASONAL_FACTORS[month.month]
        noise = np.random.normal(0, 0.9)
        headcount = max(1, round(base * seasonal + noise))

        # Win rate: stable 68-73% with small upward drift
        win_rate = 0.68 + (i / (n - 1)) * 0.05 + np.random.normal(0, 0.025)
        win_rate = float(np.clip(win_rate, 0.50, 0.85))

        confirmed = max(0, round(headcount * win_rate))
        revenue = confirmed * MONTHLY_RATE_PER_HEAD + np.random.normal(0, 9_000)
        revenue = max(0.0, revenue)

        records.append({
            "month": month.strftime("%Y-%m"),
            "total_headcount_demand": int(headcount),
            "confirmed_headcount": int(confirmed),
            "win_rate_pct": round(win_rate * 100, 1),
            "revenue_usd": round(revenue, 0),
        })
    return records


def _fit_and_forecast(history: list[dict], metric: str, horizon_months: int) -> tuple[list[dict], float, dict]:
    """
    Fit Y(t) = a + b*t + seasonal(month) using numpy polyfit.
    Returns (forecast_rows, monthly_growth_slope, seasonal_index_by_month).
    """
    values = np.array([r[metric] for r in history], dtype=float)
    t = np.arange(len(values), dtype=float)

    # Step 1: linear trend via least squares
    coeffs = np.polyfit(t, values, 1)          # [slope, intercept]
    trend_fitted = np.polyval(coeffs, t)

    # Step 2: residuals → seasonal index per calendar month
    residuals = values - trend_fitted
    cal_months = [int(r["month"].split("-")[1]) for r in history]

    seasonal: dict[int, float] = {}
    for m in range(1, 13):
        m_resids = [residuals[i] for i, mo in enumerate(cal_months) if mo == m]
        seasonal[m] = float(np.mean(m_resids)) if m_resids else 0.0

    # Step 3: 90% CI from residual standard deviation
    ci_90 = 1.64 * float(np.std(residuals))

    # Step 4: project forward
    last_t = float(len(values) - 1)
    last_month = pd.Timestamp(history[-1]["month"] + "-01")

    forecast_rows = []
    for i in range(1, horizon_months + 1):
        ft = last_t + i
        future = last_month + pd.DateOffset(months=i)
        point = float(np.polyval(coeffs, ft)) + seasonal[future.month]
        forecast_rows.append({
            "month": future.strftime("%Y-%m"),
            "forecast": round(max(0.0, point), 1),
            "lower":    round(max(0.0, point - ci_90), 1),
            "upper":    round(point + ci_90, 1),
        })

    return forecast_rows, round(float(coeffs[0]), 3), seasonal


def get_prediction_forecast(horizon_months: int = 24) -> dict:
    horizon_months = max(6, min(horizon_months, 36))
    history = _generate_synthetic_history()

    demand_fc,  demand_slope,  _ = _fit_and_forecast(history, "total_headcount_demand", horizon_months)
    revenue_fc, revenue_slope, _ = _fit_and_forecast(history, "revenue_usd",            horizon_months)
    winrate_fc, _,             _ = _fit_and_forecast(history, "win_rate_pct",            horizon_months)

    # Clip win-rate forecast to sensible bounds
    for r in winrate_fc:
        r["forecast"] = round(min(95.0, max(40.0, r["forecast"])), 1)
        r["lower"]    = round(min(95.0, max(20.0, r["lower"])),    1)
        r["upper"]    = round(min(99.0, max(40.0, r["upper"])),     1)

    # Peak demand month in forecast window
    peak = max(demand_fc, key=lambda r: r["forecast"])

    return {
        "history":          history,
        "horizon_months":   horizon_months,
        "demand_forecast":  demand_fc,
        "revenue_forecast": revenue_fc,
        "winrate_forecast": winrate_fc,
        "summary": {
            "peak_demand_month":      peak["month"],
            "peak_demand_headcount":  peak["forecast"],
            "total_forecast_headcount": round(sum(r["forecast"] for r in demand_fc), 0),
            "total_forecast_revenue_usd": round(sum(r["forecast"] for r in revenue_fc), 0),
            "demand_monthly_growth_rate": demand_slope,
        },
        "model_info": {
            "type":    "Linear Trend + Seasonal Decomposition",
            "formula": "Y(t) = a + b×t + seasonal(month)",
            "training_months": len(history),
            "training_period": f"{history[0]['month']} → {history[-1]['month']}",
            "confidence_interval": "90%",
            "note": "Trained on synthetic historical data. Real forecasting requires 3+ years of JMAN pipeline outcome records.",
        },
    }
