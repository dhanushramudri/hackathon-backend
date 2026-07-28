from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.recommendation_service import (
    RowIndexOutOfRange,
    get_backfill_candidates,
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
    include_skill: bool = True
    include_competency: bool = True
    include_availability: bool = True
    include_category_match: bool = False
    include_project_count: bool = False


@router.get("/coverage-summary")
def coverage_summary() -> dict:
    return get_coverage_summary()


@router.get("/deals")
def deals_list() -> list:
    return list_deals()


@router.post("/project-team")
def project_team(req: ProjectTeamRequest) -> dict:
    top_n = max(1, min(MAX_TOP_N, req.top_n))
    return get_project_team_recommendation(
        req.row_indices, top_n=top_n,
        include_skill=req.include_skill, include_competency=req.include_competency,
        include_availability=req.include_availability,
        include_category_match=req.include_category_match, include_project_count=req.include_project_count,
    )


@router.get("/pipeline-row/{row_index}")
def for_pipeline_row(
    row_index: int,
    top_n: int = Query(default=15, ge=1, le=MAX_TOP_N),
    include_skill: bool = Query(default=True),
    include_competency: bool = Query(default=True),
    include_availability: bool = Query(default=True),
    include_category_match: bool = Query(default=False),
    include_project_count: bool = Query(default=False),
    include_below_capacity: bool = Query(default=False),
) -> dict:
    try:
        return get_recommendations_for_pipeline_row(
            row_index, top_n=top_n,
            include_skill=include_skill, include_competency=include_competency, include_availability=include_availability,
            include_category_match=include_category_match, include_project_count=include_project_count,
            include_below_capacity=include_below_capacity,
        )
    except RowIndexOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/pipeline-row/{row_index}/semantic-match")
def semantic_match(row_index: int) -> dict:
    try:
        return get_semantic_match_suggestions(row_index)
    except RowIndexOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/backfill")
def backfill(
    employee_id: str,
    source_project_id: str,
    top_n: int = Query(default=15, ge=1, le=50),
) -> dict:
    """Find replacement candidates if an employee is pulled from a project."""
    return get_backfill_candidates(employee_id, source_project_id, top_n=top_n)


@router.get("/search")
def search(
    skillset_text: str,
    likely_start_date: str,
    requested_pct: str = "100",
    top_n: int = Query(default=15, ge=1, le=MAX_TOP_N),
    solution: str | None = Query(default=None, description="Deal proposition category (proposition_coe vocabulary) to match track record against"),
    include_skill: bool = Query(default=True),
    include_competency: bool = Query(default=True),
    include_availability: bool = Query(default=True),
    include_category_match: bool = Query(default=False),
    include_project_count: bool = Query(default=False),
    min_relevant_projects: float = Query(default=0.0, ge=0.0),
    min_total_projects: int = Query(default=0, ge=0),
) -> dict:
    result = get_recommendations(
        skillset_text, likely_start_date, requested_pct, top_n=top_n,
        requested_solution=solution,
        include_skill=include_skill, include_competency=include_competency, include_availability=include_availability,
        include_category_match=include_category_match, include_project_count=include_project_count,
    )
    if min_relevant_projects > 0 or min_total_projects > 0:
        result["candidates"] = [
            c for c in result["candidates"]
            if c["relevant_project_count"] >= min_relevant_projects
            and c["total_projects"] >= min_total_projects
        ]
    return result
