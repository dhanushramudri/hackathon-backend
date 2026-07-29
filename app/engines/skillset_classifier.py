import pandas as pd

from app.core.adapter import get_adapter
from app.engines.coe_taxonomy import resolve_coe_label

def classify_skillset_with_proof(skillset_text: str | None) -> tuple[list[str], list[dict]]:
    """Returns (categories, proof_rows). categories is resolved to the 5
    canonical CoE labels via coe_taxonomy -- the reference sheet's coe_skill
    column has inconsistent spellings across rows (e.g. "Data Science & AI"
    vs "AI & ML"), so raw values are never returned directly; proof_rows still
    carries the original raw coe_skill/coe_skills_list so callers that need to
    show their work have the real row, not just the resolved label."""
    if not skillset_text or not str(skillset_text).strip():
        return [], []
    sheet = get_adapter().get_pipeline_skillset()
    norm = str(skillset_text).strip().lower()
    matches = sheet[sheet["skills_combined"].astype(str).str.strip().str.lower() == norm]
    if matches.empty:
        return [], []
    raw_categories = matches["coe_skill"].dropna().tolist()
    categories = sorted({resolve_coe_label(c) for c in raw_categories if resolve_coe_label(c)})
    proof = [
        {
            "coe_skill": r["coe_skill"] if pd.notna(r["coe_skill"]) else None,
            "coe_skills_list": r["coe_skills_list"] if pd.notna(r["coe_skills_list"]) else None,
            "skills_combined": r["skills_combined"] if pd.notna(r["skills_combined"]) else None,
        }
        for _, r in matches.iterrows()
    ]
    return categories, proof

def classify_skillset(skillset_text: str | None) -> list[str]:
    categories, _ = classify_skillset_with_proof(skillset_text)
    return categories