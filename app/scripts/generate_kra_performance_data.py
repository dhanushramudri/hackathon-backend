"""
generate_kra_performance_data.py

Generates dummy "KRA/Performance Review" data -- the half-yearly appraisal
cycle format used by JMAN's real internal Performance Edge / KRA-KPI Forms
tool (Projects/People/Products/Sales KRA line items, each with a real-style
Goal + Appraisee Rating + Appraiser Rating + score, a Total Score, and a
BE/ME/MEE/EE performance band). The hackathon team never provided a real
export of this data, so -- same posture as generate_hr_feedback_data.py and
weekly_pulse_generator.py -- this is a clearly-labeled synthetic dataset,
grounded in real employee/manager pairs (never an invented reviewer) rather
than random pairings or fabricated org structure.

Logic:
  - Read the real employees CSV. Only employees with a real
    manager_employee_id get cycles -- the appraiser is always that real
    manager, never fabricated (same rule generate_hr_feedback_data.py uses
    for its reviewer).
  - Each qualifying employee gets 1-2 real-cadence half-year cycles (H1/H2)
    counting back from the app's "current" date, skipping cycles that would
    start before the employee's real date_of_join.
  - Each cycle uses a fixed KRA template (Projects/People/Products/Sales,
    weights summing to 100 so 4 rating levels x 100 weight = 400, matching
    the real tool's 100-400 total-score scale and its real BE/ME/MEE/EE
    bands) -- the specific KRA wording is illustrative, not JMAN's real
    template, which this repo has no access to.
  - Appraisee/appraiser ratings are 1-4, randomized but weighted positive,
    with the same deterministic per-employee sentiment bias technique as
    weekly_pulse_generator.py / generate_hr_feedback_data.py so one
    employee's cycles read consistently rather than as pure per-row noise.
  - The most recent cycle for anyone whose form_end_date hasn't passed yet
    is left "in flight" (Appraisee Submit / Management Review) rather than
    Closed, matching the real tool's workflow states.

Usage:
    python -m app.scripts.generate_kra_performance_data
    (run from backend/, writes data/Transformed/12_Performance_Cycles_dummy.csv
    and data/Transformed/13_Performance_KRA_Items_dummy.csv)
"""
import hashlib
import random
import uuid
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
TRANSFORMED_DIR = DATA_ROOT / "Transformed"
EMPLOYEES_CSV = TRANSFORMED_DIR / "01_Employee_Details_clean.csv"
CYCLES_OUTPUT_CSV = TRANSFORMED_DIR / "12_Performance_Cycles_dummy.csv"
KRA_ITEMS_OUTPUT_CSV = TRANSFORMED_DIR / "13_Performance_KRA_Items_dummy.csv"

RNG_SEED = 11

# The app's fictional "current date" is well into 2026 (see other generators'
# TRAINING_WINDOW_END-style constants) -- cycles are generated counting back
# from here, never into the future beyond the current in-flight cycle.
CURRENT_DATE = pd.Timestamp("2026-08-12")
MAX_CYCLES_PER_EMPLOYEE = 2
RATING_LEVELS = [1, 2, 3, 4]
BASE_RATING_WEIGHTS = [0.05, 0.15, 0.45, 0.35]  # mostly ME/MEE, matching a real positively-skewed review distribution

PERFORMANCE_BANDS = [
    (100, 250, "BE", "Below Expectations"),
    (251, 325, "ME", "Meeting Expectations"),
    (326, 375, "MEE", "Marginally Exceeding Expectations"),
    (376, 400, "EE", "Exceeding Expectations"),
]

STATUS_FLOW = ["Goal set", "KRA Agreed", "Appraisee Submit", "Appraiser Submit", "Management Review", "Reviewer Intervention", "Reviewer Submit", "Closed"]

# KRA template: (category, kra_name, weight, kra_kpi_description, goal_template).
# Weights sum to 100 across the 4 categories -- illustrative wording, not a
# copy of JMAN's real internal template (this repo has no access to that).
KRA_TEMPLATE = [
    ("Projects", "Productivity", 15, "Ensure consistent on-time delivery with quality across assigned tasks for the cycle.", "I will complete assigned tasks on time with quality, ensuring high productivity levels."),
    ("Projects", "Project Leakage", 15, "Deviation in project deliverables and timelines vs. the agreed project plan.", "I will ensure there are no material deviations from the agreed project plan."),
    ("People", "Resource Utilization", 15, "Real billable/allocated utilization across the entire cycle.", "I will maintain strong utilization by actively participating in projects and delivering on schedule."),
    ("People", "Workshops Conducted", 5, "Internal knowledge-sharing workshops led or co-led this cycle.", "I will organize and conduct at least one internal workshop to share knowledge."),
    ("People", "Non-commercial Contribution", 5, "Coaching, mentoring, or supporting peers outside of billable work.", "I will support at least one peer or teammate through coaching or guidance."),
    ("People", "Certification", 5, "Professional certifications or structured learning completed this cycle.", "I will prioritize continuous learning by pursuing a relevant certification."),
    ("Products", "Product/IP Contribution", 10, "Contribution to internal accelerators, tooling, or reusable IP.", "I will contribute to at least one internal product or reusable asset."),
    ("Products", "Innovation & Improvement", 10, "Process or delivery improvements proposed and implemented.", "I will identify and help implement at least one delivery improvement."),
    ("Sales", "Client Relationship", 10, "Strength of client relationship and stakeholder feedback this cycle.", "I will maintain a strong, responsive relationship with client stakeholders."),
    ("Sales", "Business Development Support", 10, "Support provided toward proposals, pitches, or account growth.", "I will support business development activity where the opportunity arises."),
]

APPRAISEE_RATING_TEXT = {
    1: "Fell short of the goal this cycle -- did not consistently meet the expected standard.",
    2: "Partially met the goal -- some progress, but inconsistent across the cycle.",
    3: "Consistently met the goal, delivering to the expected standard throughout the cycle.",
    4: "Exceeded the goal -- consistently strong delivery well beyond the expected standard.",
}
APPRAISER_RATING_TEXT = {
    1: "Did not meet expectations on this KRA this cycle -- needs clear improvement next cycle.",
    2: "Below the expected standard on this KRA -- some progress made, more consistency needed.",
    3: "Agreed -- met expectations on this KRA for the cycle.",
    4: "Exceeded expectations on this KRA -- strong, consistent delivery worth recognizing.",
}

OVERALL_FEEDBACK = {
    "positive": (
        "{emp} has had a strong cycle, consistently delivering across their key result areas and taking real "
        "ownership of their work. Client and team feedback has been positive throughout."
    ),
    "neutral": (
        "{emp} is meeting expectations this cycle overall. With continued consistency across the areas noted "
        "below, their impact should keep growing next cycle."
    ),
    "constructive": (
        "{emp} has put in effort this cycle but fell short of expectations in a few key areas. We'd like to see "
        "clear, sustained improvement next cycle, with more regular check-ins in the meantime."
    ),
}
AREAS_OF_IMPROVEMENT = {
    "positive": "Continue building on current strengths; consider taking on a stretch assignment or mentoring a junior team member next cycle.",
    "neutral": "Focus on consistency across all KRAs, and look for opportunities to contribute beyond immediate project work.",
    "constructive": "Prioritize hitting agreed timelines and communicating blockers earlier; a closer check-in cadence with your manager is recommended next cycle.",
}


def stable_employee_bias(employee_id: str) -> float:
    """Same deterministic per-employee bias trick used by
    weekly_pulse_generator.py / generate_hr_feedback_data.py."""
    h = hashlib.md5(employee_id.encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return (frac * 2) - 1


def biased_rating_weights(bias: float) -> list[float]:
    w = list(BASE_RATING_WEIGHTS)
    shift = 0.12 * bias
    w[0] = max(0.01, w[0] - shift * 0.5)
    w[1] = max(0.01, w[1] - shift * 0.5)
    w[2] = max(0.01, w[2] + shift * 0.5)
    w[3] = max(0.01, w[3] + shift)
    total = sum(w)
    return [x / total for x in w]


def band_for_score(score: int) -> tuple[str, str]:
    for low, high, code, label in PERFORMANCE_BANDS:
        if low <= score <= high:
            return code, label
    return PERFORMANCE_BANDS[-1][2], PERFORMANCE_BANDS[-1][3]


def half_year_cycles_before(as_of: pd.Timestamp, n: int) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Real half-year (H1 Apr-Sep / H2 Oct-Mar) cycle windows, most recent
    first, going back n cycles from as_of."""
    cycles = []
    year, month = as_of.year, as_of.month
    if 4 <= month <= 9:
        cur_start, cur_end, cur_label = pd.Timestamp(year, 4, 1), pd.Timestamp(year, 9, 30), f"H1 Apr{year % 100:02d} - Sep{year % 100:02d}"
    elif month >= 10:
        cur_start, cur_end, cur_label = pd.Timestamp(year, 10, 1), pd.Timestamp(year + 1, 3, 31), f"H2 Oct{year % 100:02d} - Mar{(year + 1) % 100:02d}"
    else:
        cur_start, cur_end, cur_label = pd.Timestamp(year - 1, 10, 1), pd.Timestamp(year, 3, 31), f"H2 Oct{(year - 1) % 100:02d} - Mar{year % 100:02d}"

    start, end, label = cur_start, cur_end, cur_label
    for _ in range(n):
        cycles.append((start, end, label))
        if start.month == 4:
            start, end = pd.Timestamp(start.year - 1, 10, 1), pd.Timestamp(start.year, 3, 31)
            label = f"H2 Oct{(start.year) % 100:02d} - Mar{end.year % 100:02d}"
        else:
            start, end = pd.Timestamp(start.year, 4, 1), pd.Timestamp(start.year, 9, 30)
            label = f"H1 Apr{start.year % 100:02d} - Sep{start.year % 100:02d}"
    return cycles


def main() -> None:
    random.seed(RNG_SEED)

    print("Reading real employee data ...")
    employees = pd.read_csv(EMPLOYEES_CSV, dtype=str)
    employees.columns = [c.strip() for c in employees.columns]
    for col in employees.select_dtypes(include="object").columns:
        employees[col] = employees[col].str.strip()
    employees["date_of_join"] = pd.to_datetime(employees["date_of_join"], format="%d-%m-%Y", errors="coerce")

    qualifying = employees[employees["manager_employee_id"].notna() & (employees["manager_employee_id"] != "")]
    print(f"{len(qualifying)} of {len(employees)} employees have a real manager and qualify for cycles.")

    cycle_records: list[dict] = []
    kra_records: list[dict] = []

    for _, emp in qualifying.iterrows():
        emp_id = emp["employee_id"]
        manager_id = emp["manager_employee_id"]
        job_name = emp.get("job_name") or "Employee"
        join_date = emp.get("date_of_join")

        candidate_cycles = half_year_cycles_before(CURRENT_DATE, MAX_CYCLES_PER_EMPLOYEE)
        bias = stable_employee_bias(emp_id)
        weights = biased_rating_weights(bias)

        for start, end, label in candidate_cycles:
            if pd.notna(join_date) and join_date > end:
                continue  # not yet employed during this cycle

            in_flight = end >= CURRENT_DATE
            cycle_id = uuid.uuid4().hex
            published_on = start + pd.Timedelta(days=random.randint(35, 45))
            form_end_date = end

            item_scores = []
            for category, kra_name, weight, kra_kpi, goal in KRA_TEMPLATE:
                appraisee_rating = random.choices(RATING_LEVELS, weights=weights, k=1)[0]
                # Appraiser rating tracks the appraisee's self-rating but isn't
                # identical -- a real manager sometimes agrees, sometimes
                # adjusts by one level either way.
                appraiser_rating = max(1, min(4, appraisee_rating + random.choice([-1, 0, 0, 0, 1])))
                appraiser_score = weight * appraiser_rating
                item_scores.append(appraiser_score)
                kra_records.append({
                    "cycle_id": cycle_id,
                    "category": category,
                    "kra_name": kra_name,
                    "weight": weight,
                    "kra_kpi_description": kra_kpi,
                    "goal_text": goal,
                    "appraisee_rating_text": APPRAISEE_RATING_TEXT[appraisee_rating],
                    "appraisee_score": weight * appraisee_rating,
                    "appraiser_rating_text": APPRAISER_RATING_TEXT[appraiser_rating],
                    "appraiser_score": appraiser_score,
                })

            total_score = sum(item_scores) if not in_flight else None
            if in_flight:
                # Cycle still open -- no final appraiser scores/total yet, matches
                # the real tool showing "--" for an in-progress cycle.
                status = random.choice(["Appraisee Submit", "Management Review"])
                performance_code = performance_label = None
                tier = "neutral"
            else:
                status = "Closed"
                performance_code, performance_label = band_for_score(int(total_score))
                tier = "positive" if total_score >= 326 else ("neutral" if total_score >= 251 else "constructive")

            cycle_records.append({
                "cycle_id": cycle_id,
                "employee_id": emp_id,
                "appraiser_employee_id": manager_id,
                "designation": job_name,
                "form_name": job_name,
                "cycle_label": label,
                "published_on": published_on,
                "form_end_date": form_end_date,
                "status": status,
                "total_score": total_score,
                "performance_rating_code": performance_code,
                "performance_rating_label": performance_label,
                "overall_appraiser_feedback": None if in_flight else OVERALL_FEEDBACK[tier].format(emp=emp_id),
                "overall_areas_of_improvement": None if in_flight else AREAS_OF_IMPROVEMENT[tier],
            })

    cycles_df = pd.DataFrame.from_records(cycle_records).sort_values(["employee_id", "published_on"], ascending=[True, False])
    cycles_df["published_on"] = cycles_df["published_on"].dt.strftime("%Y-%m-%d")
    cycles_df["form_end_date"] = cycles_df["form_end_date"].dt.strftime("%Y-%m-%d")

    kra_df = pd.DataFrame.from_records(kra_records)

    print(f"Writing {len(cycles_df)} performance cycles to {CYCLES_OUTPUT_CSV} ...")
    cycles_df.to_csv(CYCLES_OUTPUT_CSV, index=False)
    print(f"Writing {len(kra_df)} KRA line items to {KRA_ITEMS_OUTPUT_CSV} ...")
    kra_df.to_csv(KRA_ITEMS_OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
