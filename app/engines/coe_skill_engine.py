import pandas as pd

from app.core.adapter import get_adapter

TOP_N_SKILLS = 12

COE_SKILL_MAP: dict[str, dict] = {
    "Data Engineering": {"skill_coes": ["Data Engineering"], "confidence": "medium"},
    "AI & ML": {"skill_coes": ["Data Science & AI"], "confidence": "medium"},
    "Full Stack Engineering": {"skill_coes": ["Full Stack"], "confidence": "medium"},
    "TechOps & Automation": {"skill_coes": ["Techops & automation", "Techops & Automation"], "confidence": "medium"},
    "BI & Reporting": {"skill_coes": ["Power BI & Consulting", "Consulting"], "confidence": "low"},
}

GENERIC_SKILL_COES = {
    "Delivery", "Billable", "Billable ", "Support", "IT Support", "HR", "Finance",
    "Legal", "Support - Legal", "People", "People Team", "US Non-Billable (Back Office)",
    "GTM",
}

def _aggregate_skills(rows: pd.DataFrame, top_n: int) -> list[dict]:
    if rows.empty:
        return []
    grouped = (
        rows.groupby(["skill", "subskill"])
        .agg(
            employee_count=("employee_id", "nunique"),
            avg_score=("score", "mean"),
            common_experience=("experience", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        )
        .reset_index()
        .sort_values("employee_count", ascending=False)
        .head(top_n)
    )
    return [
        {
            "skill": r["skill"],
            "subskill": r["subskill"],
            "employee_count": int(r["employee_count"]),
            "avg_score": round(float(r["avg_score"]), 2),
            "common_experience": r["common_experience"],
        }
        for _, r in grouped.iterrows()
    ]

def derive_skills_for_coes(coes: list[str], top_n: int = TOP_N_SKILLS) -> dict:
    adapter = get_adapter()
    skills = adapter.get_skills()
    real_skills = skills[(~skills["coe"].isin(GENERIC_SKILL_COES)) & (skills["score"] > 0)]

    org_wide_fallback = _aggregate_skills(real_skills, top_n)

    by_coe: dict[str, dict] = {}
    combined_seen: set[tuple] = set()
    combined: list[dict] = []
    for coe in coes:
        mapping = COE_SKILL_MAP.get(coe, {"skill_coes": [], "confidence": "none"})
        skill_coes = mapping["skill_coes"]
        if skill_coes:
            matched = real_skills[real_skills["coe"].isin(skill_coes)]
            by_coe[coe] = {
                "skills": _aggregate_skills(matched, top_n),
                "confidence": mapping["confidence"],
                "matched_skill_coes": skill_coes,
                "fallback": None,
            }
        else:
            by_coe[coe] = {
                "skills": org_wide_fallback,
                "confidence": "none",
                "matched_skill_coes": [],
                "fallback": "no_direct_coe_skill_data",
            }
        for s in by_coe[coe]["skills"]:
            key = (s["skill"], s["subskill"])
            if key not in combined_seen:
                combined_seen.add(key)
                combined.append(s)

    combined.sort(key=lambda s: -s["employee_count"])
    return {"by_coe": by_coe, "combined": combined[:top_n]}
