"""Canonical CoE vocabulary and alias resolution.

CoE labels flow in from multiple sources -- the employee roster's home-CoE
field (skills df "coe" column, canonicalized via COE_SKILL_MAP in
coe_skill_engine.py), the pipeline skillset reference sheet's coe_skill
column, and free-text solution/tech_coe labels on deals -- and none of them
are guaranteed to use the same literal strings (e.g. "Data Science & AI" vs
"AI & ML" vs "Gen AI" all mean the same CoE).

COE_SKILL_MAP is the authoritative, maintained mapping (same one
employee_coe.py canonicalizes against), so it always wins on overlap here.
_SUPPLEMENTARY_ALIASES only fills in variants seen in OTHER sources that
COE_SKILL_MAP doesn't cover at all -- e.g. "TechOps and MS" is a
tech_coe/project-label spelling (see SKILLSET_CATEGORY_TO_TECH_COE in
recommendation_service.py), a different namespace than skill_coe, so it
won't ever show up in COE_SKILL_MAP itself. Add entries here as new
spellings turn up in non-skill_coe sources, rather than patching individual
comparisons elsewhere in the codebase.

Run this file directly (`python -m app.engines.coe_taxonomy`) to audit the
live data for any CoE-ish label that still doesn't resolve to one of the 5
canonical categories -- this is how new unmapped spellings (like the
"TechOps and Automation" / "Platform Engineering" cases) get caught before
they cause a silent ranking miss, instead of waiting for someone to notice
a wrong recommendation in the UI.
"""
import re

from app.engines.coe_skill_engine import COE_SKILL_MAP

DATA_ENGINEERING = "Data Engineering"
AI_ML = "AI & ML"
FULL_STACK = "Full Stack Engineering"
TECHOPS = "TechOps & Automation"
BI_REPORTING = "BI & Reporting"

CANONICAL_COES: frozenset[str] = frozenset({
    DATA_ENGINEERING, AI_ML, FULL_STACK, TECHOPS, BI_REPORTING,
})

def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())

# Fallback spellings NOT covered by COE_SKILL_MAP's skill_coes lists --
# "Data Science & AI" and "Consulting" etc. are already in COE_SKILL_MAP so
# they're deliberately NOT duplicated here; only genuinely-uncovered variants
# belong in this dict.
_SUPPLEMENTARY_ALIASES: dict[str, str] = {
    "data science": AI_ML,
    "ds/ai": AI_ML,
    "gen ai": AI_ML,
    "genai": AI_ML,
    "machine learning": AI_ML,
    "ml": AI_ML,
    "techops and ms": TECHOPS,          # project/tech_coe label, not a skill_coe
    "techops and automation": TECHOPS,  # pipeline-sheet spelling uses "and", COE_SKILL_MAP uses "&"
    "platform engineering": TECHOPS,    # deal-level solution/tech label seen on TechOps-flavored deals
}

def _build_alias_map() -> dict[str, str]:
    merged = dict(_SUPPLEMENTARY_ALIASES)
    # COE_SKILL_MAP is authoritative -- overlay it last so it wins on any key
    # both dicts define. Also register each canonical name against itself so
    # e.g. resolve_coe_label("ai & ml") returns "AI & ML" even though that
    # exact string isn't in anyone's skill_coes list.
    for canonical, mapping in COE_SKILL_MAP.items():
        for variant in mapping.get("skill_coes", []):
            merged[_norm(variant)] = canonical
        merged[_norm(canonical)] = canonical
    return merged

_ALIASES: dict[str, str] = _build_alias_map()

def resolve_coe_label(value: str | None) -> str | None:
    """Returns the canonical label (e.g. "AI & ML") for any recognized
    spelling variant. Falls back to the original value (title-cased if it was
    all-lowercase, otherwise left as-is) so callers still get a stable,
    displayable value even for a label this map hasn't seen yet -- same
    fallback behavior employee_coe.py has always used."""
    if not value:
        return None
    key = _norm(value)
    mapped = _ALIASES.get(key)
    if mapped:
        return mapped
    cleaned = str(value).strip()
    return cleaned.title() if cleaned.islower() else cleaned

def normalize_coe_label(value: str | None) -> str:
    """Comparison-ready form: whitespace/casing collapsed AND alias-resolved.
    Use this (not a bare .strip().lower()) anywhere two CoE labels from
    different sources are compared for equality."""
    if not value:
        return ""
    return _norm(resolve_coe_label(value) or "")

def is_known_coe(value: str | None) -> bool:
    """True if value resolves to one of the 5 canonical CoEs (as opposed to
    falling back to the cleaned-but-unrecognized original string). Useful for
    callers that want to log/flag an unrecognized CoE label rather than
    silently treating it as its own category."""
    return resolve_coe_label(value) in CANONICAL_COES


# ── One-off data audit (not used by the app at runtime) ─────────────────────
# Everything below only runs when this file is executed directly:
#     python -m app.engines.coe_taxonomy
# It is NOT imported or called anywhere else in the codebase, so it adds zero
# overhead/risk to normal app startup or request handling.

def _audit_pipeline_skillset_coe_skill() -> list[str]:
    from app.core.adapter import get_adapter
    sheet = get_adapter().get_pipeline_skillset()
    raw_labels = sheet["coe_skill"].dropna().unique()
    return sorted({l for l in raw_labels if not is_known_coe(l)})

def _audit_skills_df_coe() -> list[str]:
    """Home-CoE source (employee_coe.py reads this same column). Note:
    GENERIC_SKILL_COES entries (Delivery, Billable, HR, etc.) will show up
    here as "unresolved" -- that's expected, since those rows get filtered
    out by GENERIC_SKILL_COES before reaching _canonicalize anyway. Don't add
    those to _SUPPLEMENTARY_ALIASES."""
    from app.core.adapter import get_adapter
    skills = get_adapter().get_skills()
    raw_labels = skills["coe"].dropna().unique()
    return sorted({l for l in raw_labels if not is_known_coe(l)})

def _audit_pipeline_forecast_solution() -> list[str]:
    """Deal-level solution/tech field -- this is where "Platform Engineering"
    came from. Adjust the column name if your pipeline_forecast schema uses a
    different field (e.g. "tech_coe", "proposition_coe")."""
    from app.core.adapter import get_adapter
    pipeline = get_adapter().get_pipeline_forecast()
    col = "solution" if "solution" in pipeline.columns else None
    if col is None:
        return []
    raw_labels = pipeline[col].dropna().unique()
    return sorted({l for l in raw_labels if not is_known_coe(l)})

def _run_audit() -> None:
    print("Unresolved labels in pipeline_skillset.coe_skill:")
    print(_audit_pipeline_skillset_coe_skill())
    print()
    print("Unresolved labels in skills.coe (employee home-CoE source):")
    print(_audit_skills_df_coe())
    print()
    print("Unresolved labels in pipeline_forecast.solution (deal-level):")
    print(_audit_pipeline_forecast_solution())

if __name__ == "__main__":
    _run_audit()