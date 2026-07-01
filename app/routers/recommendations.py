from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.recommendation_service import (
    RowIndexOutOfRange,
    get_coverage_summary,
    get_project_team_recommendation,
    get_recommendations,
    get_recommendations_for_pipeline_row,
    list_deals,
)
from app.services.semantic_match_service import get_semantic_match_suggestions

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Upper bound is the real org-wide headcount ballpark, not an arbitrary round number --
# high enough that "show everyone" genuinely means everyone, while still rejecting a
# garbage/typo'd value (e.g. a stray extra zero) before it reaches the scoring loop.
MAX_TOP_N = 2000


class ProjectTeamRequest(BaseModel):
    row_indices: list[int]
    top_n: int = 15


@router.get("/coverage-summary")
def coverage_summary() -> dict:
    return get_coverage_summary()


@router.get("/deals")
def deals_list() -> list:
    return list_deals()


@router.post("/project-team")
def project_team(req: ProjectTeamRequest) -> dict:
    top_n = max(1, min(MAX_TOP_N, req.top_n))
    return get_project_team_recommendation(req.row_indices, top_n=top_n)


@router.get("/pipeline-row/{row_index}")
def for_pipeline_row(row_index: int, top_n: int = Query(default=15, ge=1, le=MAX_TOP_N)) -> dict:
    try:
        return get_recommendations_for_pipeline_row(row_index, top_n=top_n)
    except RowIndexOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/pipeline-row/{row_index}/semantic-match")
def semantic_match(row_index: int) -> dict:
    try:
        return get_semantic_match_suggestions(row_index)
    except RowIndexOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/search")
def search(
    skillset_text: str,
    likely_start_date: str,
    requested_pct: str = "100",
    top_n: int = Query(default=15, ge=1, le=MAX_TOP_N),
) -> dict:
    return get_recommendations(skillset_text, likely_start_date, requested_pct, top_n=top_n)


