"""
generate_hr_feedback_data.py

Generates dummy "HR Feedback" data -- real HR/PM performance review check-ins
on an employee's work on a specific real project, tied to real (employee,
project) pairs drawn from the real allocation data (never invented pairs),
reviewed by a real employee (the project's real reporter/approver, or the
employee's real manager -- never an invented name), grounded in the project's
real tech CoE. The hackathon team never provided real feedback records, so
this mirrors the same posture as weekly_pulse_generator.py: a clearly-labeled
synthetic dataset, grounded in real staffing/org data rather than random
employee/project pairs or fabricated reviewer identities.

Each review reads like a real short performance write-up: 1-3 themes, each
with a "Positives" paragraph and an "Areas for Growth" paragraph, wrapped in a
greeting/sign-off -- not a single generic one-line comment.

Logic:
  - Read the real allocations + projects + employees CSVs.
  - For each allocation row with a real start/end date and a duration of at
    least MIN_DURATION_DAYS, generate periodic HR check-ins roughly every
    CHECKIN_INTERVAL_WEEKS, up to MAX_FEEDBACK_PER_ALLOCATION per allocation --
    capped at "today" so nothing is dated in the future, and never more than
    a few weeks past the allocation's own end date.
  - The reviewer is a real employee: the project's real reporter_employee_id,
    falling back to approver_employee_id, falling back to the reviewed
    employee's own manager_employee_id -- never fabricated.
  - Ratings are randomized on a 1-5 scale but weighted to look like a real
    review distribution (mostly positive), with a per-employee "sentiment
    bias" (same deterministic hash-based technique as weekly_pulse_generator)
    so a given employee's feedback reads consistently across their projects
    instead of pure per-row noise.

Usage:
    python -m app.scripts.generate_hr_feedback_data
    (run from backend/, writes data/Transformed/11_HR_Feedback_dummy.csv)
"""
import hashlib
import random
import uuid
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
TRANSFORMED_DIR = DATA_ROOT / "Transformed"
ALLOCATIONS_CSV = TRANSFORMED_DIR / "03_Project_Allocation_clean.csv"
PROJECTS_CSV = TRANSFORMED_DIR / "02_Project_Details_clean.csv"
EMPLOYEES_CSV = TRANSFORMED_DIR / "01_Employee_Details_clean.csv"
OUTPUT_CSV = TRANSFORMED_DIR / "11_HR_Feedback_dummy.csv"

RNG_SEED = 7

MIN_DURATION_DAYS = 14
CHECKIN_INTERVAL_WEEKS = 8
MAX_FEEDBACK_PER_ALLOCATION = 3
POST_END_GRACE_WEEKS = 3

REVIEWER_ROLES_WEIGHTED = [
    ("Project Manager", 0.35),
    ("Engagement Manager", 0.30),
    ("Delivery Lead", 0.20),
    ("Client Stakeholder", 0.10),
    ("HR Business Partner", 0.05),
]

RATING_SCALE = [5, 4, 3, 2, 1]
BASE_RATING_WEIGHTS = [0.28, 0.40, 0.20, 0.08, 0.04]

THEME_NAMES = [
    "Technical Delivery",
    "Communication",
    "Client Delivery",
    "Teamwork & Collaboration",
    "Ownership & Initiative",
]

# Weighted count of themes per review -- mostly 2, sometimes 1 or 3, so
# reviews vary in length like real ones do.
THEME_COUNT_WEIGHTS = [0.20, 0.55, 0.25]

THEME_CONTENT: dict[str, dict[str, list[str]]] = {
    "Technical Delivery": {
        "strengths": [
            "Delivered high-quality work consistently and caught edge cases early.",
            "Already a solid, dependable engineer -- has completed a range of different technical tasks in a short amount of time.",
            "Strong technical execution; output required minimal rework from the team.",
        ],
        "achievements": [
            "Took ownership of a tricky environment/setup issue and stuck with it until it was resolved.",
            "Delivered a meaningful feature end-to-end with limited guidance, even in an unfamiliar part of the stack.",
        ],
        "growth": [
            "A couple of deliverables needed more rework than expected this cycle -- worth tightening up estimates and self-review before handoff.",
            "Needs closer check-ins on estimates; a couple of milestones slipped this cycle.",
        ],
        "light_growth": [
            "No specific areas for growth at this time -- keep up the strong execution.",
            "Encourage taking on a technical \"spike\" or two and suggesting ideas for future improvements.",
        ],
    },
    "Communication": {
        "strengths": [
            "Clear, proactive communicator -- keeps stakeholders updated without being chased.",
            "Explains technical concepts well to non-technical stakeholders.",
        ],
        "achievements": [
            "Engagement in retros and stand-ups from day one has been genuinely appreciated by the team.",
        ],
        "growth": [
            "Needs to work on proactively sharing updates rather than responding only when asked.",
            "Status updates could come earlier -- flag blockers as soon as they appear rather than waiting for the next sync.",
        ],
        "light_growth": [
            "No specific areas for growth at this time.",
            "Keep up the proactive updates -- it's noticed and appreciated.",
        ],
    },
    "Client Delivery": {
        "strengths": [
            "Provided great support across multiple demanding project workstreams at the same time, progressing both to a high standard.",
            "Client specifically called out their professionalism and responsiveness this cycle.",
        ],
        "achievements": [
            "Balanced delivery across two demanding engagements without dropping quality on either.",
        ],
        "growth": [
            "Could be more proactive in managing client expectations on scope and timelines.",
            "A client raised a concern about response time on a couple of requests -- worth a closer look.",
        ],
        "light_growth": [
            "No specific areas for growth at this time.",
            "Continue the strong client engagement -- it's landing well.",
        ],
    },
    "Teamwork & Collaboration": {
        "strengths": [
            "Very new to the team but got up to speed on everything incredibly quickly.",
            "Well-liked by the team; consistently helps unblock others.",
            "Engagement in retros and sprint ceremonies right from the start has made a real difference to the team.",
        ],
        "achievements": [
            "Jumped in to help a teammate work through a blocker without being asked.",
        ],
        "growth": [
            "Some friction with a teammate was reported this cycle -- worth a quick check-in.",
            "Could collaborate more actively with other workstreams beyond the immediate team.",
        ],
        "light_growth": [
            "For now, we just want continued strong engagement in retros and sprints.",
            "No specific areas for growth at this time.",
        ],
    },
    "Ownership & Initiative": {
        "strengths": [
            "Highly dedicated and determined to complete assigned tasks, showing strong ownership when given responsibility.",
            "Willing to take on challenges regardless of prior experience, and persistent in resolving issues.",
        ],
        "achievements": [
            "Proactively identified and fixed a process gap without being asked.",
        ],
        "growth": [
            "Needs more prompting to pick up ownership outside assigned tasks -- could be more proactive rather than waiting for direction.",
            "Should work on managing frustration during challenging situations and staying composed while troubleshooting.",
        ],
        "light_growth": [
            "No specific areas for growth at this time -- keep pushing into new areas.",
            "Continue taking initiative as you have been.",
        ],
    },
}

OVERALL_IMPACT = {
    "positive": (
        "{emp} is a hard-working, dependable contributor who consistently pushes to complete tasks and pick up "
        "new technologies. Glad to have them on the team."
    ),
    "neutral": (
        "{emp} is meeting expectations this cycle. With a bit more consistency in the areas noted above, "
        "their impact will keep growing."
    ),
    "constructive": (
        "{emp} is putting in effort but this cycle fell short of expectations in a few areas. We'd like to see "
        "clear improvement next cycle, with more regular check-ins in the meantime."
    ),
}


def stable_employee_bias(employee_id: str) -> float:
    """Same deterministic per-employee bias trick as weekly_pulse_generator --
    the same employee always reads a bit more positive/negative across every
    project, instead of pure row-level noise."""
    h = hashlib.md5(employee_id.encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return (frac * 2) - 1


def biased_rating_weights(bias: float) -> list[float]:
    w = list(BASE_RATING_WEIGHTS)
    shift = 0.15 * bias
    w[0] = max(0.01, w[0] + shift)
    w[1] = max(0.01, w[1] + shift * 0.5)
    w[3] = max(0.01, w[3] - shift * 0.5)
    w[4] = max(0.01, w[4] - shift)
    total = sum(w)
    return [x / total for x in w]


def rating_tier(rating: int) -> str:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "constructive"


def weighted_reviewer_role() -> str:
    roles = [r for r, _ in REVIEWER_ROLES_WEIGHTED]
    weights = [w for _, w in REVIEWER_ROLES_WEIGHTED]
    return random.choices(roles, weights=weights, k=1)[0]


def pick_reviewer(project_row: pd.Series | None, manager_by_employee: dict, employee_id: str) -> str | None:
    """Real employee only -- project's real reporter, then real approver,
    then the reviewed employee's real manager. Never a fabricated name."""
    if project_row is not None:
        for col in ("reporter_employee_id", "approver_employee_id"):
            val = project_row.get(col)
            if pd.notna(val) and val:
                return val
    manager = manager_by_employee.get(employee_id)
    return manager if pd.notna(manager) and manager else None


def build_theme_block(theme: str, tier: str) -> str:
    bank = THEME_CONTENT[theme]
    if tier == "constructive":
        positives = random.choice(bank["strengths"])
        growth = " ".join(random.sample(bank["growth"], k=min(2, len(bank["growth"]))))
    elif tier == "neutral":
        pool = bank["strengths"] + bank["achievements"]
        positives = random.choice(pool)
        growth = random.choice(bank["growth"])
    else:  # positive
        pool = bank["strengths"] + bank["achievements"]
        positives = " ".join(random.sample(pool, k=min(2, len(pool))))
        growth = random.choice(bank["light_growth"]) if random.random() < 0.7 else random.choice(bank["growth"])
    return f"{theme}\nPositives:\n{positives}\n\nAreas for Growth:\n{growth}"


def build_full_review(employee_id: str, project_id: str, reviewer_role: str, themes: list[str], tier: str, rating: int) -> tuple[str, str]:
    theme_blocks = [build_theme_block(t, tier) for t in themes]
    opening = f"Hi {employee_id},\n\nPlease find feedback for your work on {project_id}."
    closing = f"Overall Impact\n{OVERALL_IMPACT[tier].format(emp=employee_id)}\n\nBest,\n{reviewer_role}"
    full_text = "\n\n".join([opening, *theme_blocks, closing])
    first_positive_line = theme_blocks[0].split("Positives:\n", 1)[1].split("\n\n", 1)[0]
    summary = f"{themes[0]}: {first_positive_line}"
    if len(summary) > 160:
        summary = summary[:157].rstrip() + "..."
    return full_text, summary


def main() -> None:
    random.seed(RNG_SEED)

    print(f"Reading allocation, project, and employee data ...")
    alloc = pd.read_csv(ALLOCATIONS_CSV, dtype=str)
    alloc.columns = [c.strip() for c in alloc.columns]
    for col in alloc.select_dtypes(include="object").columns:
        alloc[col] = alloc[col].str.strip()

    projects = pd.read_csv(PROJECTS_CSV, dtype=str)
    projects.columns = [c.strip() for c in projects.columns]
    for col in projects.select_dtypes(include="object").columns:
        projects[col] = projects[col].str.strip()
    projects_by_code = projects.set_index("project_code")

    employees = pd.read_csv(EMPLOYEES_CSV, dtype=str)
    employees.columns = [c.strip() for c in employees.columns]
    for col in employees.select_dtypes(include="object").columns:
        employees[col] = employees[col].str.strip()
    manager_by_employee = employees.set_index("employee_id")["manager_employee_id"].to_dict()

    alloc["allocated_start_date"] = pd.to_datetime(alloc["allocated_start_date"], errors="coerce")
    alloc["allocated_end_date"] = pd.to_datetime(alloc["allocated_end_date"], errors="coerce")
    alloc = alloc.dropna(subset=["employee_id", "project_id", "allocated_start_date", "allocated_end_date"])
    alloc = alloc[alloc["employee_id"] != ""]

    today = pd.Timestamp.now().normalize()
    duration_days = (alloc["allocated_end_date"] - alloc["allocated_start_date"]).dt.days
    qualifying = alloc[
        (duration_days >= MIN_DURATION_DAYS)
        & (alloc["allocated_start_date"] <= today)
    ].copy()
    print(f"{len(qualifying)} of {len(alloc)} allocation rows qualify for at least one check-in.")

    records: list[dict] = []
    for _, row in qualifying.iterrows():
        emp_id = row["employee_id"]
        project_id = row["project_id"]
        start = row["allocated_start_date"]
        end = row["allocated_end_date"]
        latest_allowed = min(end + pd.Timedelta(weeks=POST_END_GRACE_WEEKS), today)

        project_row = projects_by_code.loc[project_id] if project_id in projects_by_code.index else None
        reviewer_employee_id = pick_reviewer(project_row, manager_by_employee, emp_id)
        if reviewer_employee_id is None or reviewer_employee_id == emp_id:
            # No real reviewer identity available for this project/employee, or
            # the only candidate would be reviewing themselves -- skip rather
            # than fabricate a reviewer.
            continue

        bias = stable_employee_bias(emp_id)
        weights = biased_rating_weights(bias)

        checkin_dates: list[pd.Timestamp] = []
        for k in range(1, MAX_FEEDBACK_PER_ALLOCATION + 1):
            candidate = start + pd.Timedelta(weeks=CHECKIN_INTERVAL_WEEKS * k)
            if candidate > latest_allowed:
                break
            checkin_dates.append(candidate)
        if not checkin_dates:
            early = min(start + pd.Timedelta(weeks=2), latest_allowed)
            if early >= start:
                checkin_dates.append(early)

        for fb_date in checkin_dates:
            rating = random.choices(RATING_SCALE, weights=weights, k=1)[0]
            tier = rating_tier(rating)
            n_themes = random.choices([1, 2, 3], weights=THEME_COUNT_WEIGHTS, k=1)[0]
            themes = random.sample(THEME_NAMES, k=n_themes)
            reviewer_role = weighted_reviewer_role()
            full_text, summary_comment = build_full_review(emp_id, project_id, reviewer_role, themes, tier, rating)
            would_recommend = (
                "Yes" if rating >= 4 else ("No" if rating <= 2 else random.choice(["Yes", "Yes", "No"]))
            )
            records.append(
                {
                    "feedback_id": uuid.uuid4().hex,
                    "employee_id": emp_id,
                    "project_id": project_id,
                    "feedback_date": fb_date,
                    "reviewer_employee_id": reviewer_employee_id,
                    "reviewer_role": reviewer_role,
                    "rating": rating,
                    "would_recommend": would_recommend,
                    "theme_tags": ";".join(themes),
                    "summary_comment": summary_comment,
                    "full_text": full_text,
                }
            )

    out_df = pd.DataFrame.from_records(records).sort_values(["employee_id", "feedback_date"])
    out_df["feedback_date"] = out_df["feedback_date"].dt.strftime("%Y-%m-%d")
    print(f"Writing {len(out_df)} HR feedback rows to {OUTPUT_CSV} ...")
    out_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
