"""
Headcount Prediction — Phase 2: feature engineering pipeline.

Reads the three Phase-1 synthetic tables (data/HeadcountPrediction/*.csv) and
builds a single wide feature table with lags, rolling stats, ratios,
interaction terms, and cyclical calendar encoding.

This does NOT do correlation/importance-based pruning -- that's Phase 3, on
purpose, so the full engineered set is inspectable before anything gets cut.

Leakage note for whoever builds Phase 4 (model training): all lag/rolling
features here use only data up to and including month t, which is correct
for predicting a target 3 months ahead (t+3) -- the target itself must be
built by shifting total_active_headcount BACKWARD by 3 rows relative to the
feature row (i.e. row t's features pair with month t+3's headcount), not by
using this table's own total_active_headcount(t) as a same-row target. That
shift is deliberately left to Phase 4, not done here, so this table stays a
general-purpose "features as of month t" table rather than being pre-baked
for one specific horizon.

Run:  python -m app.scripts.build_headcount_features
"""

from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / "data" / "HeadcountPrediction"

LAG_PERIODS = [1, 2, 3, 6]
ROLLING_WINDOWS = [3, 6]

# The ~90-column monthly_snapshot has a lot of narrow COE/cluster splits that
# aren't worth lagging individually (that's 90 x 4 lags = 360 columns for
# metrics we don't expect to be independently predictive). Lag/roll only the
# headline metrics that Section 3 of the plan called out; the COE/cluster
# splits stay available in their raw (unlagged) form in the output table for
# anyone who wants them, they just don't get the full derived treatment.
KEY_METRICS = [
    "total_active_headcount",
    "new_hires_total",
    "resignations_total",
    "promotions_total",
    "revenue_usd_total",
    "revenue_usd_billable",
    "billable_fte_total",
    "unbillable_fte_total",
    "billable_hours",
    "total_logged_hours",
    "over_allocated_count",
    "under_allocated_count",
    "free_pool_headcount",
    "shadow_unbilled_count",
    "notice_period_headcount",
    "completions_total",
    "completions_on_time",
    "extensions_billable",
    "extensions_unbillable",
    "new_projects_total",
    "avg_q1_motivation",
    "avg_q5_workload",
]


def load_base_frame() -> pd.DataFrame:
    snapshot = pd.read_csv(DATA_DIR / "monthly_snapshot.csv", parse_dates=["month"])
    pulse = pd.read_csv(DATA_DIR / "weekly_pulse_monthly_agg.csv", parse_dates=["month"])
    notice = pd.read_csv(DATA_DIR / "notice_period_cohort.csv", parse_dates=["month"])

    df = snapshot.merge(pulse, on="month", how="left", validate="one_to_one")

    # Extra monthly aggregates from the notice cohort beyond what
    # weekly_pulse_monthly_agg already carries (notice_cohort_avg_pulse_*,
    # notice_cohort_pulse_delta) -- cohort size as a cross-check against
    # notice_period_headcount, and how far into their notice period the
    # average person in the cohort is that month.
    notice_monthly = (
        notice.groupby("month")
        .agg(notice_cohort_size=("employee_id", "nunique"), notice_cohort_avg_months_in=("months_into_notice", "mean"))
        .reset_index()
    )
    df = df.merge(notice_monthly, on="month", how="left")
    df["notice_cohort_size"] = df["notice_cohort_size"].fillna(0).astype(int)
    df["notice_cohort_avg_months_in"] = df["notice_cohort_avg_months_in"].fillna(0.0)

    df = df.sort_values("month").reset_index(drop=True)
    return df


def _attach(df: pd.DataFrame, new_cols: dict) -> pd.DataFrame:
    """Concat all new columns at once instead of assigning one at a time --
    per-column assignment on a wide frame fragments it (pandas warns loudly
    about this), concat doesn't."""
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {
        f"{col}_lag{lag}": df[col].shift(lag)
        for col in KEY_METRICS
        for lag in LAG_PERIODS
    }
    return _attach(df, new_cols)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # min_periods=window (not 1) -- a 3-month "rolling mean" computed from 1
    # real data point isn't a rolling mean, it's noise wearing a rolling
    # mean's name. Leaves NaN during warm-up, which is correct and handled
    # explicitly in Phase 4 (drop or impute), not silently smoothed over here.
    new_cols = {}
    for col in KEY_METRICS:
        for window in ROLLING_WINDOWS:
            new_cols[f"{col}_roll{window}_mean"] = df[col].rolling(window, min_periods=window).mean()
            new_cols[f"{col}_roll{window}_std"] = df[col].rolling(window, min_periods=window).std()
    return _attach(df, new_cols)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    def safe_div(a, b):
        return a / b.replace(0, np.nan)

    new_cols = {
        "ratio_hires_to_resignations": safe_div(df["new_hires_total"], df["resignations_total"]),
        "ratio_billable_to_unbillable_fte": safe_div(df["billable_fte_total"], df["unbillable_fte_total"]),
        "utilization_rate": safe_div(df["billable_hours"], df["total_logged_hours"]),
        "ratio_over_to_under_allocated": safe_div(df["over_allocated_count"], df["under_allocated_count"]),
        "ratio_extensions_billable_to_unbillable": safe_div(df["extensions_billable"], df["extensions_unbillable"]),
        "net_hire_flow": df["new_hires_total"] - df["resignations_total"],
        "completion_on_time_rate": safe_div(df["completions_on_time"], df["completions_total"]),
    }
    return _attach(df, new_cols)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {
        "interact_peak_season_new_projects": df["is_peak_season"] * df["new_projects_total"],
        "interact_notice_headcount_pulse_delta": df["notice_period_headcount"] * df["notice_cohort_pulse_delta"].fillna(0),
        "interact_over_allocated_workload": df["over_allocated_count"] * df["avg_q5_workload"],
        "interact_low_season_free_pool": df["is_low_season"] * df["free_pool_headcount"],
    }
    return _attach(df, new_cols)


def add_calendar_encoding(df: pd.DataFrame) -> pd.DataFrame:
    month_num = df["month"].dt.month
    # Cyclical sin/cos encoding instead of 12 one-hot dummies -- keeps the
    # calendar signal to 2 columns instead of 12, which matters when n=24
    # rows total (Section 3 of the plan caps the final feature set at ~12).
    new_cols = {
        "month_sin": np.sin(2 * np.pi * month_num / 12),
        "month_cos": np.cos(2 * np.pi * month_num / 12),
        "time_index": np.arange(len(df)),  # raw month-index, for any model that wants an explicit trend term
    }
    return _attach(df, new_cols)


def main():
    df = load_base_frame()
    n_raw_cols = len(df.columns)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ratio_features(df)
    df = add_interaction_features(df)
    df = add_calendar_encoding(df)

    out_path = DATA_DIR / "engineered_features.csv"
    df.to_csv(out_path, index=False)

    print(f"Base columns: {n_raw_cols}")
    print(f"Total columns after engineering: {len(df.columns)}")
    print(f"Rows: {len(df)}")
    print(f"Wrote: {out_path}")

    print("\n--- NaN warm-up check (expected from lags/rolling windows) ---")
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print(f"{len(nan_counts)} columns have NaNs (all should be lag/rolling warm-up or edge-of-window notice-cohort fields)")
    print("Worst 5:")
    print(nan_counts.head(5))

    print("\n--- Sample of engineered columns (first 3 rows with lag6 populated, i.e. row 6+) ---")
    sample_cols = ["month", "total_active_headcount", "total_active_headcount_lag1",
                    "total_active_headcount_lag6", "total_active_headcount_roll3_mean",
                    "ratio_hires_to_resignations", "net_hire_flow", "month_sin", "month_cos"]
    print(df.loc[6:8, sample_cols].to_string(index=False))


if __name__ == "__main__":
    main()
