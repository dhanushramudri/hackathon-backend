"""Weekly Pulse survey (5 questions asked at timesheet submission -- see
weekly_pulse_generator.py for the synthetic dataset, real one never provided
by the hackathon team). Scale is 1 (strongly disagree) .. 4 (strongly agree).

Only q1/q2/q5 (inspiration, feeling valued, workload -- the project/team-
related ones) drive the flag, per an explicit call: "even one" of these three
landing on Disagree/Strongly disagree (score <=2) in a recent response is
enough to flag -- not an average. q3/q4 (feedback/growth, CDM guidance) are
still recorded and shown in the per-response detail, just not part of the
trigger. Employee-level flag is called "Not happy"; project-level keeps the
existing "Pulse risk" naming (Health page), same trigger rule.
"""
import pandas as pd

from app.core.adapter import get_adapter

NOT_HAPPY_QUESTIONS = ["q1_inspired_motivated", "q2_valued_supported", "q5_workload_sustainable"]
ALL_QUESTIONS = ["q1_inspired_motivated", "q2_valued_supported", "q3_feedback_growth", "q4_cdm_guidance", "q5_workload_sustainable"]
QUESTION_LABELS = {
    "q1_inspired_motivated": "Inspired/motivated by project",
    "q2_valued_supported": "Valued/supported by team",
    "q3_feedback_growth": "Feedback & growth",
    "q4_cdm_guidance": "CDM guidance",
    "q5_workload_sustainable": "Workload sustainable",
}
SCALE_MEANING = {4: "Strongly agree", 3: "Agree", 2: "Disagree", 1: "Strongly disagree"}
RECENT_WEEKS = 4
# Score at/below this on ANY of NOT_HAPPY_QUESTIONS = Disagree or Strongly
# disagree -- the flag trigger. Not an average.
DISAGREE_MAX_SCORE = 2

def _recent_pulse(weeks: int = RECENT_WEEKS) -> pd.DataFrame:
    df = get_adapter().get_weekly_pulse()
    if df.empty:
        return df
    # Anchored on the data's own latest submission, not wall-clock "now" --
    # this is a fixed synthetic snapshot (see weekly_pulse_generator.py) that
    # stops well before "today" ever ticks past it, so a calendar-relative
    # window would silently exclude every row once real time moves on.
    latest = df["week_start_date"].max()
    cutoff = latest - pd.Timedelta(weeks=weeks)
    return df[df["week_start_date"] >= cutoff]

def _is_bad_row(df: pd.DataFrame) -> pd.Series:
    return (df[NOT_HAPPY_QUESTIONS] <= DISAGREE_MAX_SCORE).any(axis=1)

def get_project_pulse_table(weeks: int = RECENT_WEEKS) -> pd.DataFrame:
    """One row per project_id -- precomputed once for the whole health report,
    same groupby-then-merge pattern every other per-project signal uses."""
    recent = _recent_pulse(weeks)
    if recent.empty:
        return pd.DataFrame(columns=["pulse_avg_score", "pulse_response_count", "pulse_employee_count", "is_pulse_risk"])
    recent = recent.assign(_is_bad=_is_bad_row(recent))
    grouped = recent.groupby("project_id")
    avg_score = grouped[NOT_HAPPY_QUESTIONS].mean().mean(axis=1).round(2).rename("pulse_avg_score")
    response_count = grouped.size().rename("pulse_response_count")
    employee_count = grouped["employee_id"].nunique().rename("pulse_employee_count")
    is_risk = grouped["_is_bad"].any().rename("is_pulse_risk")
    return pd.concat([avg_score, response_count, employee_count, is_risk], axis=1)

def get_project_pulse_detail(project_code: str, weeks: int = RECENT_WEEKS) -> dict | None:
    recent = _recent_pulse(weeks)
    rows = recent[recent["project_id"] == project_code]
    if rows.empty:
        return None
    scores = {q: round(rows[q].mean(), 2) for q in ALL_QUESTIONS}
    project_avg = round(sum(scores[q] for q in NOT_HAPPY_QUESTIONS) / len(NOT_HAPPY_QUESTIONS), 2)
    return {
        "response_count": int(len(rows)),
        "distinct_employees": int(rows["employee_id"].nunique()),
        "avg_score": project_avg,
        "scores": scores,
        "is_pulse_risk": bool(_is_bad_row(rows).any()),
        "worst_question": QUESTION_LABELS[min(NOT_HAPPY_QUESTIONS, key=lambda q: scores[q])],
        "window_weeks": weeks,
    }

def get_employee_pulse_table(weeks: int = RECENT_WEEKS) -> pd.DataFrame:
    """One row per employee_id -- Wellbeing's "Not happy" signal, separate
    from the existing timesheet-hours burnout flag."""
    recent = _recent_pulse(weeks)
    if recent.empty:
        return pd.DataFrame(columns=["is_not_happy"])
    recent = recent.assign(_is_bad=_is_bad_row(recent))
    is_not_happy = recent.groupby("employee_id")["_is_bad"].any().rename("is_not_happy")
    return is_not_happy.to_frame()

def get_employee_pulse_detail(employee_id: str, weeks: int = RECENT_WEEKS) -> dict | None:
    """Full recent-response record for one employee -- the "proof" behind the
    Not happy flag: every response in the window, most recent first, all 5
    answers (score + plain-English meaning) plus whether that specific
    response is what triggered the flag."""
    recent = _recent_pulse(weeks)
    rows = recent[recent["employee_id"] == employee_id].sort_values("week_start_date", ascending=False)
    if rows.empty:
        return None
    responses = []
    for _, r in rows.iterrows():
        responses.append({
            "week_start_date": r["week_start_date"].strftime("%Y-%m-%d"),
            "project_id": r["project_id"],
            "is_not_happy": bool(any(r[q] <= DISAGREE_MAX_SCORE for q in NOT_HAPPY_QUESTIONS)),
            "answers": {
                q: {"score": int(r[q]), "meaning": SCALE_MEANING[int(r[q])], "is_not_happy_question": q in NOT_HAPPY_QUESTIONS}
                for q in ALL_QUESTIONS
            },
        })
    return {
        "is_not_happy": any(resp["is_not_happy"] for resp in responses),
        "responses": responses,
        "window_weeks": weeks,
    }

if __name__ == "__main__":
    fake = pd.DataFrame({
        "employee_id": ["E1", "E1", "E2"],
        "project_id": ["P1", "P1", "P1"],
        "q1_inspired_motivated": [1, 4, 4],
        "q2_valued_supported": [4, 4, 4],
        "q3_feedback_growth": [4, 4, 4],
        "q4_cdm_guidance": [4, 4, 4],
        "q5_workload_sustainable": [4, 4, 4],
    })
    # E1's FIRST response has q1=1 (strongly disagree) -- one bad question in
    # one response is enough, even though E1's second response is all 4s.
    bad = _is_bad_row(fake)
    assert bad.tolist() == [True, False, False], "only the row with a disagree/strongly-disagree answer should flag"
    e1_flagged = fake[fake["employee_id"] == "E1"].pipe(_is_bad_row).any()
    assert e1_flagged, "E1 must be flagged from their one bad response even though their other response is fine"
    e2_flagged = fake[fake["employee_id"] == "E2"].pipe(_is_bad_row).any()
    assert not e2_flagged, "E2 has no disagree/strongly-disagree answers and must not be flagged"
    print("pulse_engine self-check OK")
