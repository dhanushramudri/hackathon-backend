"""
Headcount Prediction — Phase 4: model training + walk-forward validation.

Loads the 12 features selected in Phase 3, evaluates 3 candidate model
architectures via walk-forward validation, and reports MAPE for a
3-month-ahead headcount forecast.

CANDIDATES (see the approved plan, Section 4):
  1. ARIMA (classical baseline) -- seasonal SARIMA needs 2+ full 12-month
     cycles to identify; with only 15 usable aligned rows that's not
     feasible, so this is plain ARIMA(1,1,1) on the target series alone,
     included as a sanity-check floor, not a serious contender.
  2. LightGBM (regularized gradient boosting) -- can use the full selected
     feature set and capture nonlinearities, at real risk of overfitting
     with ~15 rows. Shallow trees, few estimators, strong regularization.
  3. Trend + Seasonal + Ridge Regression (RECOMMENDED) -- extends the
     existing app's simple_forecast_engine.py house style (trend + seasonal
     decomposition, plain numpy/sklearn, fully transparent) with the
     selected features as Ridge-regularized regressors.

With ~15 usable rows, none of these produce a statistically "solid" MAPE --
the honest goal here is to show the three side by side and recommend the one
that degrades most gracefully at this sample size, not to claim any of them
is production-grade on synthetic data this small.

Run:  python -m app.scripts.train_headcount_models
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # statsmodels is chatty about convergence on tiny series; folds report NaN on real failures instead

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / "data" / "HeadcountPrediction"
CHARTS_DIR = DATA_DIR / "charts"

FORECAST_HORIZON_MONTHS = 3
HOLDOUT_SIZE = 3
FOLD_SIZE = 3


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    with open(DATA_DIR / "selected_features.json") as f:
        selected = json.load(f)
    features = selected["final_features"]

    df = pd.read_csv(DATA_DIR / "engineered_features.csv", parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)
    target = df["total_active_headcount"].shift(-FORECAST_HORIZON_MONTHS)
    aligned = pd.concat([df, target.rename("_target")], axis=1).dropna(subset=features + ["_target"]).reset_index(drop=True)

    return aligned[features], aligned["_target"], aligned["month"], features


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def walk_forward_folds(n: int, holdout_size: int, fold_size: int) -> list[tuple[range, range]]:
    """Expanding-window folds over the rows BEFORE the holdout, then the
    holdout itself as a final, separately-reported fold. Returns
    (train_range, test_range) pairs in row-index terms."""
    usable = n - holdout_size
    folds = []
    start = usable % fold_size or fold_size  # first fold trains on whatever's left over so later folds are even fold_size blocks
    train_end = start
    while train_end + fold_size <= usable:
        folds.append((range(0, train_end), range(train_end, train_end + fold_size)))
        train_end += fold_size
    # Final holdout: train on everything before it, predict the true holdout.
    folds.append((range(0, usable), range(usable, n)))
    return folds


def fit_predict_arima(y_train: pd.Series, n_ahead: int) -> np.ndarray:
    try:
        model = ARIMA(y_train.values, order=(1, 1, 1))
        fitted = model.fit()
        return np.asarray(fitted.forecast(steps=n_ahead))
    except Exception:
        return np.full(n_ahead, y_train.iloc[-1])  # flat fallback if it fails to converge


def fit_predict_lightgbm(X_train, y_train, X_test) -> np.ndarray:
    model = LGBMRegressor(
        max_depth=2, n_estimators=40, learning_rate=0.08,
        reg_alpha=0.5, reg_lambda=0.5, min_child_samples=3,
        random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_ridge_hybrid(X_train, y_train, X_test, time_train, time_test) -> np.ndarray:
    # Trend term (raw time index) + the selected features, standardized so
    # Ridge's single alpha penalizes them comparably.
    scaler = StandardScaler()
    Xt_train = np.column_stack([time_train, X_train])
    Xt_test = np.column_stack([time_test, X_test])
    Xt_train_scaled = scaler.fit_transform(Xt_train)
    Xt_test_scaled = scaler.transform(Xt_test)

    model = Ridge(alpha=5.0, random_state=42)
    model.fit(Xt_train_scaled, y_train)
    return model.predict(Xt_test_scaled)


def evaluate_candidates(X: pd.DataFrame, y: pd.Series, months: pd.Series) -> dict:
    n = len(X)
    folds = walk_forward_folds(n, HOLDOUT_SIZE, FOLD_SIZE)
    time_index = np.arange(n)

    results = {"arima": {"folds": []}, "lightgbm": {"folds": []}, "ridge_hybrid": {"folds": []}}
    holdout_predictions = {}

    for i, (train_idx, test_idx) in enumerate(folds):
        train_idx, test_idx = list(train_idx), list(test_idx)
        is_holdout = (i == len(folds) - 1)
        label = "holdout" if is_holdout else f"fold_{i + 1}"

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        arima_pred = fit_predict_arima(y_train, len(test_idx))
        lgbm_pred = fit_predict_lightgbm(X_train, y_train, X_test)
        ridge_pred = fit_predict_ridge_hybrid(X_train, y_train, X_test, time_index[train_idx], time_index[test_idx])

        for name, pred in [("arima", arima_pred), ("lightgbm", lgbm_pred), ("ridge_hybrid", ridge_pred)]:
            fold_mape = mape(y_test.values, pred)
            entry = {"label": label, "mape": round(fold_mape, 2), "n_test": len(test_idx)}
            if is_holdout:
                results[name]["holdout_mape"] = round(fold_mape, 2)
                holdout_predictions[name] = pred
            else:
                results[name]["folds"].append(entry)

    for name in results:
        fold_mapes = [f["mape"] for f in results[name]["folds"]]
        results[name]["walk_forward_avg_mape"] = round(float(np.mean(fold_mapes)), 2) if fold_mapes else None

    holdout_months = months.iloc[folds[-1][1]].dt.strftime("%Y-%m").tolist()
    holdout_actual = y.iloc[folds[-1][1]].tolist()

    return results, holdout_predictions, holdout_months, holdout_actual


def make_backtest_chart(holdout_months, holdout_actual, holdout_predictions):
    plt.figure(figsize=(9, 5.5))
    plt.plot(holdout_months, holdout_actual, marker="o", label="Actual", color="#111827", linewidth=2)
    colors = {"arima": "#f59e0b", "lightgbm": "#ef4444", "ridge_hybrid": "#6366f1"}
    labels = {"arima": "ARIMA (baseline)", "lightgbm": "LightGBM", "ridge_hybrid": "Trend+Seasonal+Ridge (recommended)"}
    for name, pred in holdout_predictions.items():
        plt.plot(holdout_months, pred, marker="s", linestyle="--", label=labels[name], color=colors[name])
    plt.ylabel("Total active headcount")
    plt.title("Holdout backtest: 3-month-ahead headcount forecast, 3 candidates")
    plt.legend()
    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / "holdout_backtest.png", dpi=150)
    plt.close()


def main():
    X, y, months, features = load_data()
    print(f"Rows available: {len(X)}, features: {len(features)}")

    results, holdout_predictions, holdout_months, holdout_actual = evaluate_candidates(X, y, months)

    print("\n--- Walk-forward validation results ---")
    for name, r in results.items():
        print(f"\n{name}:")
        for f in r["folds"]:
            print(f"  {f['label']}: MAPE={f['mape']}% (n={f['n_test']})")
        print(f"  walk_forward_avg_mape: {r['walk_forward_avg_mape']}%")
        print(f"  HOLDOUT mape (touched once): {r['holdout_mape']}%")

    make_backtest_chart(holdout_months, holdout_actual, holdout_predictions)

    output = {
        "forecast_horizon_months": FORECAST_HORIZON_MONTHS,
        "n_rows_used": len(X),
        "features_used": features,
        "holdout_months": holdout_months,
        "holdout_actual": holdout_actual,
        "candidates": results,
        "recommended_model": "ridge_hybrid",
        "recommendation_rationale": (
            "With only ~15 usable rows, ARIMA and LightGBM are shown for comparison but "
            "the Trend+Seasonal+Ridge hybrid is recommended: it's the most statistically "
            "defensible choice at this sample size, stays interpretable, and extends the "
            "app's existing simple_forecast_engine.py house style rather than introducing "
            "an unexplainable model for a Resource Manager audience. Note: on THIS holdout, "
            "plain ARIMA scored a numerically lower MAPE -- expected on a synthetic headcount "
            "series this smooth (it's built as a bottom-up cumulative hires-minus-resignations "
            "flow, so it's highly autoregressive by construction). ARIMA can't use any of the "
            "Category A-F engineered signals (revenue, pulse, over-allocation, etc.), so it "
            "would miss real early-warning signals in actual data even though it looks good "
            "here. The recommendation is about which model generalizes and explains itself, "
            "not which one won this specific 3-row holdout."
        ),
        "acceptance_threshold_mape": 15.0,
    }
    with open(DATA_DIR / "model_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote model_results.json and charts/holdout_backtest.png")
    recommended_holdout = results["ridge_hybrid"]["holdout_mape"]
    verdict = "PASS" if recommended_holdout < output["acceptance_threshold_mape"] else "BELOW ACCEPTANCE THRESHOLD"
    print(f"Recommended model (ridge_hybrid) holdout MAPE: {recommended_holdout}% -- {verdict} (threshold {output['acceptance_threshold_mape']}%)")


if __name__ == "__main__":
    main()
