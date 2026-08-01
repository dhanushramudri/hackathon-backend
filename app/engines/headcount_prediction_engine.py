"""
Headcount Prediction engine — Phase 5.

Serves live forecasts using the synthetic data + engineered features +
selected feature set + validated model choice produced by the Phase 1-4
scripts under app/scripts/ (generate_headcount_prediction_data.py,
build_headcount_features.py, select_headcount_features.py,
train_headcount_models.py). Those scripts are the source of truth for HOW
the data/features/model choice were built -- see
data/HeadcountPrediction/README.md for the full writeup, including the
assumptions made (training window, "cluster" definition, etc.) and the
walk-forward validation results (data/HeadcountPrediction/model_results.json).

This engine does NOT regenerate synthetic data or re-run feature selection
on every request -- it loads the already-computed engineered_features.csv +
selected_features.json (checked-in artifacts) and re-fits the single
recommended model (Trend + Seasonal + Ridge hybrid) at request time, which
is fast (24 rows) and guarantees the served forecast always reflects the
latest artifacts on disk without needing a separate "retrain" step.

HONESTY NOTE ON HORIZON: the model was trained and walk-forward validated
for EXACTLY a 3-month-ahead forecast (see model_results.json). Months 1-3 of
any forecast returned here use real historical feature data and are within
that validated horizon. Months 4+ are produced by recursively feeding the
model's own prior predictions back in as the headcount feature and holding
every other selected feature at its last known real value -- a standard but
much rougher extrapolation, clearly flagged as such in the response
(`is_validated_horizon: false`) rather than presented with the same
confidence as the first 3 months.
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from app.core.config import HEADCOUNT_PREDICTION_DIR

MAX_FORECAST_MONTHS = 12
RIDGE_ALPHA = 5.0

# Every table this feature actually used, exposed as-is for full transparency
# -- a Resource Manager should be able to see the literal rows/columns that
# fed the forecast, not just a model_info summary. Order here is the order
# tabs are offered in the UI.
RAW_TABLES: dict[str, dict[str, str]] = {
    "monthly_snapshot": {
        "file": "monthly_snapshot.csv",
        "label": "Monthly Snapshot",
        "description": "The primary 24-month table -- revenue, EBITDA margin, projects, headcount, FTE, and calendar signals (Categories A-E of the plan).",
    },
    "notice_period_cohort": {
        "file": "notice_period_cohort.csv",
        "label": "Notice Period Cohort",
        "description": "One row per synthetic employee per month while under their 3-month notice window, with before/during pulse scores.",
    },
    "weekly_pulse_monthly_agg": {
        "file": "weekly_pulse_monthly_agg.csv",
        "label": "Weekly Pulse (Monthly Agg)",
        "description": "Monthly aggregates of the synthetic weekly pulse survey (motivation, workload, notice-cohort deltas).",
    },
    "engineered_features": {
        "file": "engineered_features.csv",
        "label": "Engineered Features",
        "description": "The full Phase 2 output: base columns + lags, rolling stats, ratios, interactions, and calendar encoding (285 columns).",
    },
    "feature_correlation_report": {
        "file": "feature_correlation_report.csv",
        "label": "Correlation Report",
        "description": "Every candidate feature's Spearman/Pearson correlation with the 3-month-ahead target, and whether it survived redundancy pruning.",
    },
    "feature_importance_report": {
        "file": "feature_importance_report.csv",
        "label": "Feature Importance",
        "description": "Permutation importance ranking (on the mutual-info-screened set) used to pick the final 12 features.",
    },
}


def get_raw_table(table: str) -> dict:
    if table not in RAW_TABLES:
        raise ValueError(f"Unknown table {table!r}. Valid options: {list(RAW_TABLES)}")
    spec = RAW_TABLES[table]
    df = pd.read_csv(HEADCOUNT_PREDICTION_DIR / spec["file"])
    # NaN -> null, and force everything to plain Python types so this
    # serializes cleanly regardless of pandas/numpy dtype quirks.
    df = df.astype(object).where(pd.notna(df), None)
    rows = df.to_dict(orient="records")
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                row[k] = bool(v)
    return {
        "table": table,
        "label": spec["label"],
        "description": spec["description"],
        "columns": list(df.columns),
        "rows": rows,
        "row_count": len(rows),
    }


def list_raw_tables() -> list[dict]:
    return [{"table": key, "label": spec["label"], "description": spec["description"]} for key, spec in RAW_TABLES.items()]


def _load_artifacts() -> tuple[pd.DataFrame, dict, dict]:
    df = pd.read_csv(HEADCOUNT_PREDICTION_DIR / "engineered_features.csv", parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)
    with open(HEADCOUNT_PREDICTION_DIR / "selected_features.json") as f:
        selected = json.load(f)
    with open(HEADCOUNT_PREDICTION_DIR / "model_results.json") as f:
        results = json.load(f)
    return df, selected, results


def _fit_ridge(X: np.ndarray, y: np.ndarray) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


@lru_cache(maxsize=1)
def _production_model():
    """Fit the recommended model on ALL available (feature, 3mo-ahead-target)
    pairs -- not the train/holdout split used for validation in Phase 4
    (that split's only job was to produce an honest MAPE; the model actually
    served uses every row once that's done, same as retraining a validated
    model on the full dataset before shipping it)."""
    df, selected, results = _load_artifacts()
    features = selected["final_features"]
    horizon = selected["forecast_horizon_months"]

    target = df["total_active_headcount"].shift(-horizon)
    aligned = pd.concat([df, target.rename("_target")], axis=1).dropna(subset=features + ["_target"])

    time_index = aligned.index.to_numpy(dtype=float)  # original df row index (0..23), not a 0..n re-index
    X = np.column_stack([time_index, aligned[features].to_numpy(dtype=float)])
    y = aligned["_target"].to_numpy(dtype=float)

    model, scaler = _fit_ridge(X, y)

    residuals = y - model.predict(scaler.transform(X))
    residual_std = float(np.std(residuals))

    return {
        "df": df,
        "features": features,
        "horizon": horizon,
        "model": model,
        "scaler": scaler,
        "residual_std": residual_std,
        "last_row_index": int(df.index[-1]),
        "results": results,
    }


def _predict_one(state: dict, time_idx: float, feature_row: dict) -> float:
    x = np.array([[time_idx] + [feature_row[f] for f in state["features"]]], dtype=float)
    x_scaled = state["scaler"].transform(x)
    return float(state["model"].predict(x_scaled)[0])


# Canonical COE keys (matching generate_headcount_prediction_data.py's COES
# list) -> display labels, used for the COE breakdown and notice-period-by-COE
# panels below.
_COE_LABELS = {
    "data_engineering": "Data Engineering",
    "bi_reporting": "BI & Reporting",
    "ai_ml": "AI & ML",
    "full_stack": "Full Stack Engineering",
    "techops": "TechOps & Automation",
}


def _compute_insights(df: pd.DataFrame, forecast: list[dict]) -> dict:
    """Executive-facing insights built entirely from data this feature already
    generates (monthly_snapshot / notice_period_cohort / weekly_pulse) but
    that was previously only visible in the raw data tables. Nothing here
    retrains or re-forecasts anything -- it's read-and-summarize over the
    same `df` the model already uses, plus one extra read of
    notice_period_cohort.csv for the current month's COE breakdown."""
    latest = df.iloc[-1]
    lookback_idx = max(0, len(df) - 4)  # ~3 months back
    prior = df.iloc[lookback_idx]

    headcount_change_pct = round(
        (latest["total_active_headcount"] - prior["total_active_headcount"]) / prior["total_active_headcount"] * 100, 1
    )
    net_hire_flow = int(latest["new_hires_total"] - latest["resignations_total"])

    validated_forecast = [f for f in forecast if f["is_validated_horizon"]]
    forecast_3mo = validated_forecast[-1] if validated_forecast else (forecast[-1] if forecast else None)
    forecast_change_pct = (
        round((forecast_3mo["forecast"] - latest["total_active_headcount"]) / latest["total_active_headcount"] * 100, 1)
        if forecast_3mo else None
    )

    # --- Workforce productivity: revenue per active headcount ---
    # Currency is GBP (£), matching the real reported financials this data is
    # grounded in (see generate_headcount_prediction_data.py's "REAL DATA
    # GROUNDING" section) -- the "revenue_usd_*" column names are a naming
    # artifact predating that grounding and don't reflect actual currency.
    revenue_per_head_history = [
        {"month": row["month"].strftime("%Y-%m"), "value": round(row["revenue_usd_total"] / row["total_active_headcount"], 0)}
        for _, row in df.iterrows()
    ]
    current_revenue_per_head = revenue_per_head_history[-1]["value"]
    prior_revenue_per_head = revenue_per_head_history[lookback_idx]["value"]

    ebitda_margin_history = [
        {"month": row["month"].strftime("%Y-%m"), "value": round(float(row["ebitda_margin_pct"]), 1)}
        for _, row in df.iterrows()
    ]
    current_ebitda_margin_pct = ebitda_margin_history[-1]["value"]
    prior_ebitda_margin_pct = ebitda_margin_history[lookback_idx]["value"]

    # --- Headcount by COE: current mix (from billable FTE shares) applied
    # proportionally to the total forecast -- an approximation (FTE share
    # used as a headcount-share proxy), not an independently modeled
    # per-COE forecast. Labeled as such in the API response's own field
    # names ("estimated", "mix") rather than presented as equally precise.
    coe_fte = {key: float(latest[f"billable_fte_{key}"]) for key in _COE_LABELS}
    total_coe_fte = sum(coe_fte.values()) or 1.0
    coe_mix = [
        {"coe": _COE_LABELS[key], "fte": round(fte, 1), "share_pct": round(fte / total_coe_fte * 100, 1)}
        for key, fte in coe_fte.items()
    ]
    coe_forecast = [
        {
            "month": f["month"],
            "by_coe": {_COE_LABELS[key]: round(f["forecast"] * (fte / total_coe_fte), 1) for key, fte in coe_fte.items()},
        }
        for f in forecast
    ]

    # --- Attrition & retention ---
    # notice_period_headcount (used above/in monthly_snapshot) is a smoothed
    # statistical estimate; notice_period_cohort.csv is built from actual
    # per-resignation notice windows, which structurally CANNOT have rows for
    # the most recent month(s) in the window -- that would require someone's
    # resignation to already be known further in the future than the
    # synthetic data extends (see the Phase-1 README's documented edge
    # effect). Falling back to the most recent month that has real cohort
    # rows, and saying so explicitly, rather than silently showing an empty
    # breakdown next to a nonzero headline count.
    notice_df = pd.read_csv(HEADCOUNT_PREDICTION_DIR / "notice_period_cohort.csv")
    months_with_data = sorted(notice_df["month"].unique()) if not notice_df.empty else []
    notice_by_coe_month = months_with_data[-1] if months_with_data else None
    current_notice = notice_df[notice_df["month"] == notice_by_coe_month] if notice_by_coe_month else notice_df.iloc[0:0]
    notice_by_coe = (
        [{"coe": _COE_LABELS.get(k, k), "count": int(v)} for k, v in current_notice["coe"].value_counts().items()]
        if not current_notice.empty else []
    )

    flight_risk_note = None
    pulse_delta = latest.get("notice_cohort_pulse_delta")
    if pd.notna(pulse_delta) and pulse_delta < -0.2:
        flight_risk_note = (
            f"Employees currently in their notice period showed a pulse-score drop of "
            f"{pulse_delta:.2f} (on the 1-4 scale) in the months before resigning -- "
            f"consistent with a disengagement-before-exit pattern in this data."
        )

    hires_vs_resignations = [
        {
            "month": row["month"].strftime("%Y-%m"),
            "new_hires": int(row["new_hires_total"]),
            "resignations": int(row["resignations_total"]),
            "net": int(row["new_hires_total"] - row["resignations_total"]),
        }
        for _, row in df.tail(6).iterrows()
    ]

    # --- Utilization / bench health ---
    utilization_history = [
        {
            "month": row["month"].strftime("%Y-%m"),
            "free_pool": int(row["free_pool_headcount"]),
            "over_allocated": int(row["over_allocated_count"]),
            "under_allocated": int(row["under_allocated_count"]),
        }
        for _, row in df.tail(6).iterrows()
    ]

    # --- Risk flags ---
    risk_flags = []
    if net_hire_flow < 0:
        risk_flags.append({
            "severity": "warning",
            "message": f"Net hiring flow is negative this month ({int(latest['new_hires_total'])} hires vs "
                       f"{int(latest['resignations_total'])} resignations) -- attrition is currently outpacing hiring.",
        })
    notice_count = int(latest["notice_period_headcount"])
    if notice_count > 0:
        pct_of_headcount = round(notice_count / latest["total_active_headcount"] * 100, 1)
        risk_flags.append({
            "severity": "warning" if pct_of_headcount > 3 else "info",
            "message": f"{notice_count} employee(s) ({pct_of_headcount}% of headcount) are currently in their notice period.",
        })
    if forecast_change_pct is not None and forecast_change_pct < -2:
        risk_flags.append({
            "severity": "warning",
            "message": f"Headcount is forecast to decline {abs(forecast_change_pct)}% over the validated horizon.",
        })
    if flight_risk_note:
        risk_flags.append({"severity": "warning", "message": flight_risk_note})
    risk_flags.append({
        "severity": "info",
        "message": f"This forecast is statistically validated {len(validated_forecast)} month(s) ahead; "
                   f"treat anything beyond that as a directional scenario, not a validated number.",
    })

    # --- Executive summary ---
    direction_word = "grown" if headcount_change_pct > 0 else "declined" if headcount_change_pct < 0 else "held steady"
    executive_summary = []
    if forecast_3mo:
        forecast_word = "grow" if (forecast_change_pct or 0) > 0 else "decline" if (forecast_change_pct or 0) < 0 else "hold steady"
        executive_summary.append(
            f"Headcount has {direction_word} {abs(headcount_change_pct)}% over the last 3 months "
            f"({int(prior['total_active_headcount'])} → {int(latest['total_active_headcount'])}), and is forecast to "
            f"{forecast_word} to ~{round(forecast_3mo['forecast'])} by {forecast_3mo['month']}."
        )
    rev_trend = "up" if current_revenue_per_head > prior_revenue_per_head else "down" if current_revenue_per_head < prior_revenue_per_head else "flat"
    executive_summary.append(
        f"Revenue per active headcount is currently ~£{current_revenue_per_head:,.0f}/month, "
        f"{rev_trend} from ~£{prior_revenue_per_head:,.0f}/month three months ago."
    )
    margin_trend = "up" if current_ebitda_margin_pct > prior_ebitda_margin_pct else "down" if current_ebitda_margin_pct < prior_ebitda_margin_pct else "flat"
    executive_summary.append(
        f"Adj. EBITDA margin is currently {current_ebitda_margin_pct:.1f}%, "
        f"{margin_trend} from {prior_ebitda_margin_pct:.1f}% three months ago."
    )
    executive_summary.append(
        f"Net hiring this month: {'+' if net_hire_flow >= 0 else ''}{net_hire_flow} "
        f"({int(latest['new_hires_total'])} hires vs {int(latest['resignations_total'])} resignations)."
    )
    executive_summary.append(
        f"{int(latest['over_allocated_count'])} employee(s) are over-allocated, {int(latest['under_allocated_count'])} "
        f"under-allocated, and {int(latest['free_pool_headcount'])} are in the free pool right now."
    )

    return {
        "executive_summary": executive_summary,
        "risk_flags": risk_flags,
        "headcount_change_pct_3mo": headcount_change_pct,
        "forecast_change_pct": forecast_change_pct,
        "productivity": {
            "current_revenue_per_head_usd": current_revenue_per_head,
            "history": revenue_per_head_history,
            "current_ebitda_margin_pct": current_ebitda_margin_pct,
            "ebitda_margin_history": ebitda_margin_history,
        },
        "coe_breakdown": {
            "latest_month": latest["month"].strftime("%Y-%m"),
            "mix": coe_mix,
            "forecast": coe_forecast,
        },
        "attrition": {
            "notice_period_current": notice_count,
            "notice_period_by_coe": notice_by_coe,
            "notice_period_by_coe_as_of_month": (
                pd.Timestamp(notice_by_coe_month).strftime("%Y-%m") if notice_by_coe_month else None
            ),
            "hires_vs_resignations": hires_vs_resignations,
            "flight_risk_note": flight_risk_note,
        },
        "utilization": {
            "free_pool_current": int(latest["free_pool_headcount"]),
            "over_allocated_current": int(latest["over_allocated_count"]),
            "under_allocated_current": int(latest["under_allocated_count"]),
            "history": utilization_history,
        },
    }


def get_headcount_prediction(horizon_months: int = 12) -> dict:
    horizon_months = max(1, min(horizon_months, MAX_FORECAST_MONTHS))
    state = _production_model()
    df, features, horizon = state["df"], state["features"], state["horizon"]

    history = [
        {
            "month": row["month"].strftime("%Y-%m"),
            "total_active_headcount": int(row["total_active_headcount"]),
            "new_hires_chennai": int(row["new_hires_chennai"]),
            "new_hires_uk": int(row["new_hires_uk"]),
            "new_hires_usa": int(row["new_hires_usa"]),
        }
        for _, row in df.iterrows()
    ]

    last_real_row_idx = state["last_row_index"]  # 23 (Dec 2024)
    last_month = df.loc[last_real_row_idx, "month"]

    # Rows last_real_row_idx - horizon + 1 .. last_real_row_idx already have
    # REAL, fully-known features and simply haven't had a target computed
    # yet (their target would be horizon months past the end of the data) --
    # these give the first `horizon` forecast months for free, no
    # extrapolation needed, and are within the validated horizon.
    forecast = []
    running_headcount_row = dict(df.loc[last_real_row_idx])  # last known real feature values, for the recursive fallback below

    for h in range(1, horizon_months + 1):
        target_month = last_month + pd.DateOffset(months=h)
        source_row_idx = last_real_row_idx - horizon + h  # the feature-row this month's prediction is based on

        if source_row_idx <= last_real_row_idx:
            # Real historical features -- within the validated 3-month horizon.
            feature_row = df.loc[source_row_idx]
            is_validated = True
        else:
            # Recursive extrapolation beyond the validated horizon: every
            # selected feature except total_active_headcount is held at its
            # last known real value; total_active_headcount is fed back from
            # this engine's own prior prediction. See module docstring.
            feature_row = dict(running_headcount_row)
            is_validated = False

        feature_values = {f: float(feature_row[f]) for f in features}
        point = _predict_one(state, float(source_row_idx), feature_values)
        point = max(0.0, point)

        # Confidence band widens the further we are past the validated
        # horizon -- sqrt(h/horizon) is a simple, explicit heuristic for
        # compounding uncertainty under held-constant-feature extrapolation,
        # not a statistically derived interval.
        widen = np.sqrt(h / horizon) if h > horizon else 1.0
        ci_90 = 1.64 * state["residual_std"] * widen

        forecast.append({
            "month": target_month.strftime("%Y-%m"),
            "forecast": round(point, 1),
            "lower": round(max(0.0, point - ci_90), 1),
            "upper": round(point + ci_90, 1),
            "is_validated_horizon": is_validated,
        })

        if not is_validated:
            running_headcount_row["total_active_headcount"] = point

    validated_months = sum(1 for f in forecast if f["is_validated_horizon"])
    insights = _compute_insights(df, forecast)

    return {
        "history": history,
        "training_period": f"{history[0]['month']} → {history[-1]['month']}",
        "horizon_months": horizon_months,
        "validated_horizon_months": validated_months,
        "forecast": forecast,
        "insights": insights,
        "model_info": {
            "type": "Trend + Seasonal + Ridge Regression (hybrid)",
            "formula": "Y(t+3) = Ridge(time_index, selected_features(t)), 12 features selected via Spearman correlation pruning + mutual info + permutation importance",
            "trained_on": "100% synthetic data -- see data/HeadcountPrediction/README.md",
            "training_rows_used": state["results"]["n_rows_used"],
            "features_used": features,
            "validated_forecast_horizon_months": horizon,
            "holdout_mape_pct": state["results"]["candidates"]["ridge_hybrid"]["holdout_mape"],
            "walk_forward_avg_mape_pct": state["results"]["candidates"]["ridge_hybrid"]["walk_forward_avg_mape"],
            "acceptance_threshold_mape_pct": state["results"]["acceptance_threshold_mape"],
            "other_candidates_evaluated": {
                name: {"holdout_mape_pct": r["holdout_mape"], "walk_forward_avg_mape_pct": r["walk_forward_avg_mape"]}
                for name, r in state["results"]["candidates"].items() if name != "ridge_hybrid"
            },
            "recommendation_rationale": state["results"]["recommendation_rationale"],
            "confidence_interval": "90% for the first 3 (validated) months; widens for months beyond that via a simple sqrt(h/3) heuristic, since those months hold most features constant rather than using real data",
            "note": (
                f"Trained on synthetic historical data ({history[0]['month']}-{history[-1]['month']}). Real forecasting "
                "requires multiple years of actual JMAN headcount/revenue/pulse records. "
                "Months beyond the 3-month validated horizon are extrapolated by holding "
                "revenue/FTE/pulse/etc. features at their last known value and feeding this "
                "model's own prior prediction back in as the headcount signal -- treat those "
                "months as a rough scenario, not a forecast with the same footing as the first 3."
            ),
        },
    }
