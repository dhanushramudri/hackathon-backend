from fastapi import APIRouter, HTTPException, Query

from app.services.recommendation_service import (
    RowIndexOutOfRange,
    get_coverage_summary,
    get_recommendations,
    get_recommendations_for_pipeline_row,
)
from app.services.semantic_match_service import get_semantic_match_suggestions

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Upper bound is the real org-wide headcount ballpark, not an arbitrary round number --
# high enough that "show everyone" genuinely means everyone, while still rejecting a
# garbage/typo'd value (e.g. a stray extra zero) before it reaches the scoring loop.
MAX_TOP_N = 2000

@router.get("/coverage-summary")
def coverage_summary() -> dict:
    return get_coverage_summary()

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
