"""
generate_weekly_pulse.py

Generates dummy "Weekly Pulse" survey data from an existing timesheet_details
export, and writes it out as an Excel file.

Logic:
  - Read the timesheet CSV.
  - For each employee, find every ISO week in which they logged at least one
    timesheet entry with status APPROVED or SAVED (i.e. they "filled their
    timesheet" that week).
  - Generate one Weekly Pulse response per employee per such week, answering
    the 5 standard pulse questions on a 4-point numeric scale:
        4 = Strongly agree, 3 = Agree, 2 = Disagree, 1 = Strongly disagree
    (see the "question_legend" and "answer_scale_legend" sheets for the
    mapping back to human-readable text).
  - Answers are randomized but weighted to look like a real survey: mostly
    positive, with a per-employee "sentiment bias" so a handful of employees
    consistently skew lower (useful later for burnout/attrition-risk demos).

Requirements:
    pip install pandas openpyxl

Usage:
    python generate_weekly_pulse.py --input timesheet_details.csv --output weekly_pulse_dummy.xlsx
    (both args are optional; see DEFAULT_INPUT / DEFAULT_OUTPUT below)
"""

import argparse
import hashlib
import random
import uuid
from datetime import timedelta

import pandas as pd

DEFAULT_INPUT = "data/Transformed/04_Timesheet_Details_clean.csv"
DEFAULT_OUTPUT = "weekly_pulse_dummy.xlsx"

# Only weeks where the employee's timesheet reached one of these statuses
# count as "filled" -- draft/rejected rows don't trigger a pulse.
QUALIFYING_STATUSES = {"APPROVED", "SAVED"}

QUESTIONS = [
    ("q1_inspired_motivated", "I feel inspired and motivated by my current project(s)."),
    ("q2_valued_supported", "I feel valued and supported by my team members and project manager(s)."),
    ("q3_feedback_growth", "I am getting enough feedback and growth opportunities."),
    ("q4_cdm_guidance", "I am getting the guidance and career development from my (CDM) that I need."),
    ("q5_workload_sustainable", "My current workload is sustainable."),
]

# Numeric 4-point scale (higher = more positive). Index-aligned with BASE_WEIGHTS.
ANSWER_SCALE = [4, 3, 2, 1]
ANSWER_LABELS = {4: "Strongly agree", 3: "Agree", 2: "Disagree", 1: "Strongly disagree"}

# Base weights for a healthy-skewed team (sums to 1.0), aligned with ANSWER_SCALE
BASE_WEIGHTS = [0.32, 0.45, 0.16, 0.07]

RNG_SEED = 42  # change or remove for different random output each run


def stable_employee_bias(employee_id: str) -> float:
    """Deterministic per-employee bias in [-1, 1] derived from their ID, so the
    same employee always gets the same 'tends to answer more positively /
    negatively' tendency across runs, without storing extra state."""
    h = hashlib.md5(employee_id.encode()).hexdigest()
    # Map first 8 hex chars to a float in [0, 1], then rescale to [-1, 1]
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return (frac * 2) - 1


def biased_weights(bias: float) -> list[float]:
    """Shift BASE_WEIGHTS toward the positive or negative end of the scale
    based on an employee's bias. bias > 0 => more positive answers,
    bias < 0 => more negative answers."""
    w = list(BASE_WEIGHTS)
    shift = 0.18 * bias  # magnitude of the effect
    # move mass between "Strongly agree/Agree" and "Disagree/Strongly disagree"
    w[0] = max(0.02, w[0] + shift)
    w[1] = max(0.02, w[1] + shift * 0.5)
    w[2] = max(0.02, w[2] - shift * 0.5)
    w[3] = max(0.02, w[3] - shift)
    total = sum(w)
    return [x / total for x in w]


def iso_week_bounds(d: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (Monday, Sunday) for the ISO week containing date d."""
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=6)
    return start.normalize(), end.normalize()


def main():
    parser = argparse.ArgumentParser(description="Generate dummy weekly pulse data from timesheet data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to timesheet CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write the pulse Excel file")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Reading timesheet data from {args.input} ...")
    ts = pd.read_csv(args.input, dtype=str)
    ts.columns = [c.strip() for c in ts.columns]

    required_cols = {"employee_id", "manager_id", "project_id", "date", "status"}
    missing = required_cols - set(ts.columns)
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    # Dates in the sample are DD-MM-YYYY
    ts["date_parsed"] = pd.to_datetime(
    ts["date"],
    errors="coerce"
    )
    ts["time"] = pd.to_numeric(ts["time"], errors="coerce").fillna(0.0)

    # Drop rows with missing/placeholder employee_id -- these are not real
    # employees and must not be aggregated together under a shared fake key
    # (e.g. "0"), which would sum unrelated people's hours into one bogus record.
    ts["employee_id"] = ts["employee_id"].astype(str).str.strip()
    invalid_emp_mask = ts["employee_id"].isin(["", "0", "nan", "NaN", "None"])
    n_invalid = invalid_emp_mask.sum()
    if n_invalid:
        print(f"Warning: dropping {n_invalid} timesheet rows with missing/invalid employee_id.")
    ts = ts[~invalid_emp_mask].copy()

    ts = ts[ts["date_parsed"].notna() & ts["status"].isin(QUALIFYING_STATUSES)].copy()
    if ts.empty:
        raise ValueError("No qualifying timesheet rows found (check status values / date format).")

    ts["week_start"], ts["week_end"] = zip(*ts["date_parsed"].map(iso_week_bounds))

    print("Grouping into employee/week submissions ...")
    grouped = (
        ts.groupby(["employee_id", "week_start", "week_end"])
        .agg(
            manager_id=("manager_id", "first"),
            hours_logged=("time", "sum"),
            project_id=("project_id", lambda s: s.value_counts().idxmax() if s.notna().any() else "UNKNOWN"),
        )
        .reset_index()
        .sort_values(["employee_id", "week_start"])
    )
    print(f"Found {grouped['employee_id'].nunique()} employees across {len(grouped)} employee-weeks.")

    records = []
    for _, row in grouped.iterrows():
        bias = stable_employee_bias(row["employee_id"])
        weights = biased_weights(bias)

        # Numeric answers (4=Strongly agree ... 1=Strongly disagree), not text.
        answers = {key: random.choices(ANSWER_SCALE, weights=weights, k=1)[0] for key, _ in QUESTIONS}

        # Pulse submitted a couple of days after the week ends, like a real check-in
        submitted_on = row["week_end"] + timedelta(days=random.randint(1, 3))

        records.append(
            {
                "pulse_surrogate_key": uuid.uuid4().hex,
                "employee_id": row["employee_id"],
                "manager_id": row["manager_id"],
                "project_id": row["project_id"],
                # Real date values (not formatted strings) so Excel/consumers get proper dates.
                "week_start_date": row["week_start"],
                "week_end_date": row["week_end"],
                "hours_logged_that_week": round(row["hours_logged"], 2),
                **answers,
                "progress_pct": 100,
                "status": "SUBMITTED",
                "submitted_on": submitted_on,
                "created_at": submitted_on,
                "updated_at": submitted_on,
                "data_loaded_at": submitted_on,
            }
        )

    out_df = pd.DataFrame.from_records(records)

    date_cols = [
        "week_start_date", "week_end_date",
        "submitted_on", "created_at", "updated_at", "data_loaded_at",
    ]
    for col in date_cols:
        out_df[col] = pd.to_datetime(out_df[col])

    # Nice column order / readable headers for the questions
    col_order = [
        "pulse_surrogate_key", "employee_id", "manager_id", "project_id",
        "week_start_date", "week_end_date", "hours_logged_that_week",
        *[key for key, _ in QUESTIONS],
        "progress_pct", "status", "submitted_on", "created_at", "updated_at", "data_loaded_at",
    ]
    out_df = out_df[col_order]

    print(f"Writing {len(out_df)} weekly pulse rows to {args.output} ...")
    with pd.ExcelWriter(args.output, engine="openpyxl", datetime_format="DD-MM-YYYY") as writer:
        out_df.to_excel(writer, sheet_name="weekly_pulse", index=False)

        # Legend sheet mapping column keys to the actual survey question text
        legend = pd.DataFrame(QUESTIONS, columns=["column_key", "question_text"])
        legend.to_excel(writer, sheet_name="question_legend", index=False)

        # Legend sheet mapping numeric scores back to their meaning
        scale_legend = pd.DataFrame(
            sorted(ANSWER_LABELS.items(), key=lambda kv: -kv[0]),
            columns=["score", "meaning"],
        )
        scale_legend.to_excel(writer, sheet_name="answer_scale_legend", index=False)

    print("Done.")


if __name__ == "__main__":
    main()