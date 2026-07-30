"""
Headcount Prediction — Phase 3: feature selection.

Reads engineered_features.csv (Phase 2) and:
  1. Builds the 3-month-ahead target (total_active_headcount shifted back 3
     rows relative to its feature row -- see Phase 2's leakage note).
  2. Prunes redundant features via Spearman correlation (primary) with
     Pearson reported for reference/disagreement-flagging.
  3. Screens survivors via mutual information (cheap), then ranks via
     permutation importance (more expensive, more trustworthy out-of-sample
     signal) on a small RandomForest.
  4. Caps the final feature set at 12 columns -- a hard cap driven by having
     only ~15 usable rows after lag warm-up + target shift, not by whatever
     the importance ranking would otherwise suggest keeping.
  5. Writes selected_features.json + two charts (correlation heatmap,
     permutation importance bar chart).

IMPORTANT CAVEAT, stated plainly rather than dressed up: after dropping the
6-row lag/rolling warm-up period and the 3 rows lost to the forward target
shift, only ~15 rows are available for this analysis. Every ranking here
(correlation, mutual info, permutation importance) is a directional signal
on a very small sample, not a statistically robust result. That's the honest
consequence of 24 months of history, not a bug in the method.

Run:  python -m app.scripts.select_headcount_features
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / "data" / "HeadcountPrediction"
CHARTS_DIR = DATA_DIR / "charts"

FORECAST_HORIZON_MONTHS = 3
CORRELATION_DROP_THRESHOLD = 0.85
MI_SCREEN_TOP_N = 25
FINAL_FEATURE_CAP = 12

NON_FEATURE_COLS = {"month", "new_hires_by_role_json", "resignations_by_role_json"}


def load_aligned_frame() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(DATA_DIR / "engineered_features.csv", parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)

    target = df["total_active_headcount"].shift(-FORECAST_HORIZON_MONTHS)
    df = pd.concat([df, target.rename("_target")], axis=1)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and c != "_target"]

    # Only rows where EVERY candidate feature is populated (past the lag/
    # rolling warm-up) AND the target exists (not in the last 3 months).
    aligned = df.dropna(subset=feature_cols + ["_target"]).reset_index(drop=True)

    # Drop zero-variance columns (e.g. is_month_end, always 1 at monthly grain
    # -- flagged as a known constant back in the Phase-1 README). Correlation
    # is undefined for a constant, and it carries no information either way.
    constant_cols = [c for c in feature_cols if aligned[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant column(s): {constant_cols}")
    feature_cols = [c for c in feature_cols if c not in constant_cols]

    return aligned[feature_cols], aligned["_target"], feature_cols


def prune_by_correlation(X: pd.DataFrame, y: pd.Series) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    spearman_target = X.apply(lambda col: col.corr(y, method="spearman"))
    pearson_target = X.apply(lambda col: col.corr(y, method="pearson"))

    spearman_matrix = X.corr(method="spearman")

    # Greedy: process features in descending |correlation with target|, keep
    # a feature only if it isn't too redundant (|rho| > threshold) with a
    # feature already kept -- since we process in target-correlation order,
    # the one we drop from any redundant pair is always the weaker predictor.
    order = spearman_target.abs().sort_values(ascending=False).index.tolist()
    kept: list[str] = []
    for feat in order:
        if pd.isna(spearman_target[feat]):
            continue
        redundant = any(abs(spearman_matrix.loc[feat, k]) > CORRELATION_DROP_THRESHOLD for k in kept)
        if not redundant:
            kept.append(feat)

    disagreements = []
    for feat in kept:
        s, p = spearman_target[feat], pearson_target[feat]
        if pd.notna(s) and pd.notna(p) and (np.sign(s) != np.sign(p) or abs(s - p) > 0.3):
            disagreements.append({"feature": feat, "spearman": round(float(s), 3), "pearson": round(float(p), 3)})

    corr_report = pd.DataFrame({
        "feature": order,
        "spearman_vs_target": [round(float(spearman_target[f]), 3) if pd.notna(spearman_target[f]) else None for f in order],
        "pearson_vs_target": [round(float(pearson_target[f]), 3) if pd.notna(pearson_target[f]) else None for f in order],
        "kept_after_pruning": [f in kept for f in order],
    })
    return kept, corr_report, pd.DataFrame(disagreements)


def screen_by_mutual_info(X: pd.DataFrame, y: pd.Series, candidates: list[str]) -> list[str]:
    mi = mutual_info_regression(X[candidates], y, random_state=42)
    ranked = pd.Series(mi, index=candidates).sort_values(ascending=False)
    return ranked.head(min(MI_SCREEN_TOP_N, len(ranked))).index.tolist()


def rank_by_permutation_importance(X: pd.DataFrame, y: pd.Series, candidates: list[str]) -> pd.DataFrame:
    # Small, heavily-regularized RandomForest -- with ~15 rows this is a
    # ranking tool, not a model we'd ship. See module docstring.
    model = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42)
    model.fit(X[candidates], y)
    result = permutation_importance(model, X[candidates], y, n_repeats=30, random_state=42, scoring="neg_mean_absolute_error")
    ranked = pd.DataFrame({
        "feature": candidates,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    return ranked


def make_charts(X: pd.DataFrame, y: pd.Series, final_features: list[str], importance_df: pd.DataFrame) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    heat_data = X[final_features].copy()
    heat_data["target_headcount_3mo_ahead"] = y.values
    corr = heat_data.corr(method="spearman")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, cbar_kws={"label": "Spearman ρ"})
    plt.title("Selected features vs. 3-month-ahead headcount target (Spearman correlation)")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    top = importance_df[importance_df["feature"].isin(final_features)].sort_values("perm_importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["perm_importance_mean"], xerr=top["perm_importance_std"], color="#6366f1")
    plt.xlabel("Permutation importance (Δ MAE when shuffled, in-sample ranking only)")
    plt.title("Final selected features — permutation importance")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main():
    X, y, all_feature_cols = load_aligned_frame()
    print(f"Aligned rows for feature selection: {len(X)} (out of 24 total, after 6-row lag warm-up + {FORECAST_HORIZON_MONTHS}-row target shift)")
    print(f"Candidate features before pruning: {len(all_feature_cols)}")

    kept, corr_report, disagreements = prune_by_correlation(X, y)
    print(f"After Spearman correlation pruning (threshold {CORRELATION_DROP_THRESHOLD}): {len(kept)} features")
    if not disagreements.empty:
        print(f"\nSpearman/Pearson disagreements among kept features ({len(disagreements)}):")
        print(disagreements.to_string(index=False))

    mi_screened = screen_by_mutual_info(X, y, kept)
    print(f"\nAfter mutual-info screen (top {MI_SCREEN_TOP_N}): {len(mi_screened)} features")

    importance_df = rank_by_permutation_importance(X, y, mi_screened)
    final_features = importance_df.head(FINAL_FEATURE_CAP)["feature"].tolist()
    print(f"\nFinal feature set (capped at {FINAL_FEATURE_CAP}):")
    print(importance_df.head(FINAL_FEATURE_CAP).to_string(index=False))

    make_charts(X, y, final_features, importance_df)

    corr_report.to_csv(DATA_DIR / "feature_correlation_report.csv", index=False)
    importance_df.to_csv(DATA_DIR / "feature_importance_report.csv", index=False)
    with open(DATA_DIR / "selected_features.json", "w") as f:
        json.dump({
            "forecast_horizon_months": FORECAST_HORIZON_MONTHS,
            "n_rows_used": len(X),
            "final_features": final_features,
            "correlation_drop_threshold": CORRELATION_DROP_THRESHOLD,
        }, f, indent=2)

    print(f"\nWrote: selected_features.json, feature_correlation_report.csv, feature_importance_report.csv")
    print(f"Wrote charts to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
