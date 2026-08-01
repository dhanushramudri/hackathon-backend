"""HR Feedback -- real HR/PM performance review check-ins on real (employee,
project) engagements (see generate_hr_feedback_data.py for the synthetic
dataset, real one never provided by the hackathon team, grounded in real
allocation/project/org data including the real reviewing employee).

Purely a manual-review "proof" surface for the Employee Profile modal: when
the recommendation engine suggests a candidate, the resource manager can open
this tab and read what HR/PMs have actually said about that person's real
project history. Deliberately NOT wired into recommendation_service.py or
scoring.py -- this data must never influence ranking/scoring, only manual
human review.
"""
import pandas as pd

from app.core.adapter import get_adapter

RATING_SCALE = [5, 4, 3, 2, 1]


def _primary_coe(tech_coe: str | None) -> str | None:
    """Projects sometimes carry multiple tech CoEs joined with ';' (e.g.
    "Data Engineering;BI and Reporting") -- take the first as the project's
    primary CoE for filtering, same convention used for display elsewhere."""
    if not tech_coe or pd.isna(tech_coe):
        return None
    return tech_coe.split(";")[0].strip() or None


def _feedback_with_project_info() -> pd.DataFrame:
    adapter = get_adapter()
    feedback = adapter.get_hr_feedback()
    if feedback.empty:
        return feedback
    projects = adapter.get_projects()
    merged = feedback.merge(
        projects[["project_code", "client_id", "tech_coe"]],
        left_on="project_id", right_on="project_code", how="left",
    )
    merged["coe"] = merged["tech_coe"].apply(_primary_coe)
    merged["themes"] = merged["theme_tags"].fillna("").apply(lambda s: [t for t in s.split(";") if t])
    return merged


def _theme_averages(rows: pd.DataFrame) -> dict[str, float]:
    if rows.empty:
        return {}
    exploded = rows[["rating", "themes"]].explode("themes").dropna(subset=["themes"])
    if exploded.empty:
        return {}
    return {theme: round(float(sub["rating"].mean()), 2) for theme, sub in exploded.groupby("themes")}


def get_employee_feedback(
    employee_id: str,
    weeks_back: int | None = None,
    coe: str | None = None,
    project_id: str | None = None,
    reviewer_employee_id: str | None = None,
    theme: str | None = None,
    ratings: list[int] | None = None,
) -> dict:
    all_feedback = _feedback_with_project_info()
    if all_feedback.empty:
        return _empty_result(employee_id)

    employee_rows = all_feedback[all_feedback["employee_id"] == employee_id]
    if employee_rows.empty:
        return _empty_result(employee_id)

    # Filter option lists reflect what's actually available for THIS employee,
    # unfiltered -- so a dropdown never offers a choice that would silently
    # zero out the list.
    available_coes = sorted(employee_rows["coe"].dropna().unique().tolist())
    available_projects = sorted(employee_rows["project_id"].dropna().unique().tolist())
    available_themes = sorted({t for themes in employee_rows["themes"] for t in themes})
    reviewer_lookup = (
        employee_rows[["reviewer_employee_id", "reviewer_role"]]
        .drop_duplicates(subset=["reviewer_employee_id"])
        .sort_values("reviewer_employee_id")
    )
    available_reviewers = [
        {"employee_id": r["reviewer_employee_id"], "role": r["reviewer_role"]}
        for _, r in reviewer_lookup.iterrows()
    ]

    rows = employee_rows
    if weeks_back is not None:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(weeks=weeks_back)
        rows = rows[rows["feedback_date"] >= cutoff]
    if coe is not None:
        rows = rows[rows["coe"] == coe]
    if project_id is not None:
        rows = rows[rows["project_id"] == project_id]
    if reviewer_employee_id is not None:
        rows = rows[rows["reviewer_employee_id"] == reviewer_employee_id]
    if theme is not None:
        rows = rows[rows["themes"].apply(lambda ts: theme in ts)]
    if ratings:
        rows = rows[rows["rating"].isin(ratings)]

    rows = rows.sort_values("feedback_date", ascending=False)

    entries = []
    for _, r in rows.iterrows():
        entries.append(
            {
                "feedback_id": r["feedback_id"],
                "project_id": r["project_id"],
                "client_id": r.get("client_id") if pd.notna(r.get("client_id")) else None,
                "coe": r.get("coe") if pd.notna(r.get("coe")) else None,
                "feedback_date": r["feedback_date"].strftime("%Y-%m-%d") if pd.notna(r["feedback_date"]) else None,
                "reviewer_employee_id": r["reviewer_employee_id"],
                "reviewer_role": r["reviewer_role"],
                "rating": int(r["rating"]),
                "would_recommend": r["would_recommend"] == "Yes",
                "themes": r["themes"],
                "summary_comment": r["summary_comment"],
                "full_text": r["full_text"],
            }
        )

    if len(rows) > 0:
        avg_rating = round(float(rows["rating"].mean()), 2)
        would_recommend_pct = round(float((rows["would_recommend"] == "Yes").mean()) * 100, 1)
        rating_breakdown = {str(score): int((rows["rating"] == score).sum()) for score in RATING_SCALE}
        theme_averages = _theme_averages(rows)
    else:
        avg_rating = None
        would_recommend_pct = None
        rating_breakdown = {str(score): 0 for score in RATING_SCALE}
        theme_averages = {}

    return {
        "employee_id": employee_id,
        "total_response_count": int(len(employee_rows)),
        "response_count": int(len(rows)),
        "distinct_project_count": int(rows["project_id"].nunique()),
        "avg_rating": avg_rating,
        "would_recommend_pct": would_recommend_pct,
        "rating_breakdown": rating_breakdown,
        "theme_averages": theme_averages,
        "available_coes": available_coes,
        "available_projects": available_projects,
        "available_themes": available_themes,
        "available_reviewers": available_reviewers,
        "entries": entries,
    }


def _empty_result(employee_id: str) -> dict:
    return {
        "employee_id": employee_id,
        "total_response_count": 0,
        "response_count": 0,
        "distinct_project_count": 0,
        "avg_rating": None,
        "would_recommend_pct": None,
        "rating_breakdown": {str(score): 0 for score in RATING_SCALE},
        "theme_averages": {},
        "available_coes": [],
        "available_projects": [],
        "available_themes": [],
        "available_reviewers": [],
        "entries": [],
    }
