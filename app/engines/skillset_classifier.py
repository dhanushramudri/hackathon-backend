import pandas as pd

from app.core.adapter import get_adapter

def classify_skillset_with_proof(skillset_text: str | None) -> tuple[list[str], list[dict]]:
    """Returns (categories, proof_rows). categories is the same exact-match result
    classify_skillset() has always returned. proof_rows is the literal reference-sheet
    row(s) that matched -- including coe_skills_list alongside coe_skill, since the two
    columns don't always agree in the source spreadsheet, so callers that need to show
    their work (not just the resulting label) have the real row to point at."""
    if not skillset_text or not str(skillset_text).strip():
        return [], []
    sheet = get_adapter().get_pipeline_skillset()
    norm = str(skillset_text).strip().lower()
    matches = sheet[sheet["skills_combined"].astype(str).str.strip().str.lower() == norm]
    if matches.empty:
        return [], []
    categories = sorted(matches["coe_skill"].dropna().unique().tolist())
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
