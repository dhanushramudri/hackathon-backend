from fastapi import APIRouter, HTTPException

from app.services.recommendation_service import (
    RowIndexOutOfRange,
    get_coverage_summary,
    get_recommendations,
    get_recommendations_for_pipeline_row,
)
from app.services.semantic_match_service import get_semantic_match_suggestions

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

@router.get("/coverage-summary")
def coverage_summary() -> dict:
    return get_coverage_summary()

@router.get("/pipeline-row/{row_index}")
def for_pipeline_row(row_index: int) -> dict:
    try:
        return get_recommendations_for_pipeline_row(row_index)
    except RowIndexOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/pipeline-row/{row_index}/semantic-match")
def semantic_match(row_index: int) -> dict:
    try:
        return get_semantic_match_suggestions(row_index)
    except RowIndexOutOfRange as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.get("/search")
def search(skillset_text: str, likely_start_date: str, requested_pct: str = "100") -> dict:
    return get_recommendations(skillset_text, likely_start_date, requested_pct)
