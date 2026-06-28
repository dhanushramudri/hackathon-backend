from app.core.adapter import get_adapter

def classify_skillset(skillset_text: str | None) -> list[str]:
    if not skillset_text or not str(skillset_text).strip():
        return []
    sheet = get_adapter().get_pipeline_skillset()
    norm = str(skillset_text).strip().lower()
    matches = sheet[sheet["skills_combined"].astype(str).str.strip().str.lower() == norm]
    if matches.empty:
        return []
    return sorted(matches["coe_skill"].dropna().unique().tolist())
