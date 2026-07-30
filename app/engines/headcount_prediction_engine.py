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
        "description": "The primary 24-month table -- revenue, projects, headcount, FTE, and calendar signals (Categories A-E of the plan).",
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
        "description": "The full Phase 2 output: base columns + lags, rolling stats, ratios, interactions, and calendar encoding (267 columns).",
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


def get_headcount_prediction(horizon_months: int = 12) -> dict:
    horizon_months = max(1, min(horizon_months, MAX_FORECAST_MONTHS))
    state = _production_model()
    df, features, horizon = state["df"], state["features"], state["horizon"]

    history = [
        {"month": row["month"].strftime("%Y-%m"), "total_active_headcount": int(row["total_active_headcount"])}
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

    return {
        "history": history,
        "training_period": f"{history[0]['month']} → {history[-1]['month']}",
        "horizon_months": horizon_months,
        "validated_horizon_months": validated_months,
        "forecast": forecast,
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
                "Trained on synthetic historical data (Jan 2023-Dec 2024). Real forecasting "
                "requires multiple years of actual JMAN headcount/revenue/pulse records. "
                "Months beyond the 3-month validated horizon are extrapolated by holding "
                "revenue/FTE/pulse/etc. features at their last known value and feeding this "
                "model's own prior prediction back in as the headcount signal -- treat those "
                "months as a rough scenario, not a forecast with the same footing as the first 3."
            ),
        },
    }
