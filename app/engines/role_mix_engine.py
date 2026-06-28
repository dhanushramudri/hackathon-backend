
import pandas as pd

from app.core.adapter import get_adapter

DND_TACTICAL_BUILD = "D&D Tactical Build"

_DND_TEMPLATE = {
    "Principal": 0.25,
    "Technical Architect": 0.25,
    "Associate Consultant": 1.0,
    "Consultant": 0.5,
    "Solution Consultant": 0.5,
    "Senior Software Engineer": 1.0,
}

DOCX_CATEGORY_MAP: dict[str, dict] = {
    DND_TACTICAL_BUILD: {"docx_given": True},
    "Build Phase - Tactical Build": {"type_of_project": "Client Project"},
    "Build Phase - Enterprise Platform Build": {"type_of_project": "Client Project"},
    "Build Phase - Data Platform Build": {"type_of_project": "Client Project", "tech_coe_any": ["Data Engineering"]},
    "Data Science Projects": {"type_of_project": "Client Project", "tech_coe_any": ["Data Science", "Data Science & ML"]},
    "AI Projects": {"type_of_project": "Client Project", "tech_coe_any": ["Gen AI", "Data Science & ML"]},
    "MS projects": {"type_of_project": "Managed Services"},
    "Full stack Projects": {"type_of_project": "Client Project", "tech_coe_any": ["Full Stack Engineering"]},
    "Value creation Projects": {"proposition_coe_any": ["Value Creation"]},
}

def _primary_coe(tech_coe: str | None) -> str:
    if not tech_coe or pd.isna(tech_coe):
        return "Unknown"
    return str(tech_coe).split(";")[0]

def _real_completed_merged() -> pd.DataFrame:
    adapter = get_adapter()
    projects = adapter.get_projects()
    allocations = adapter.get_allocations()
    employees = adapter.get_employees()

    real = projects[
        (projects["date_source"].isin(["given", "derived_allocation"])) & (projects["project_status"] == "COMPLETE")
    ].copy()
    real["tech_coe_primary"] = real["tech_coe"].apply(_primary_coe)

    merged = (
        real.merge(allocations, left_on="project_code", right_on="project_id")
        .merge(employees[["employee_id", "job_name"]], on="employee_id", how="left")
    )
    merged = merged.dropna(subset=["job_name"])
    merged["fte"] = merged["allocation_by_percentage"] / 100.0
    return merged

COMMON_ROLE_PREVALENCE_PCT = 40.0

def _aggregate_role_mix_detailed(group: pd.DataFrame) -> dict:
    n_projects = group["project_code"].nunique()
    roles = []
    role_mix_fte: dict[str, float] = {}
    for designation, rows in group.groupby("job_name"):
        n_projects_with_role = rows["project_code"].nunique()
        prevalence_pct = round(100 * n_projects_with_role / n_projects, 0)
        typical_pct = float(rows["allocation_by_percentage"].mode().iat[0])
        heads_per_project = rows.groupby("project_code")["employee_id"].nunique()
        headcount = max(1, round(heads_per_project.mean()))
        roles.append(
            {
                "designation": designation,
                "headcount": int(headcount),
                "typical_pct": typical_pct,
                "prevalence_pct": prevalence_pct,
                "common": bool(prevalence_pct >= COMMON_ROLE_PREVALENCE_PCT),
            }
        )
        role_mix_fte[designation] = round(headcount * typical_pct / 100, 2)
    roles.sort(key=lambda r: -r["prevalence_pct"])
    expected_headcount_common = sum(r["headcount"] for r in roles if r["common"])
    return {
        "roles": roles,
        "role_mix": role_mix_fte,
        "expected_headcount_common": expected_headcount_common,
        "sample_size": int(n_projects),
        "source": "derived_empirical",
    }

def _docx_template_to_roles(template: dict[str, float]) -> list[dict]:
    return [
        {"designation": d, "headcount": 1, "typical_pct": round(fte * 100, 1), "prevalence_pct": None, "common": True}
        for d, fte in template.items()
    ]

def build_role_mix_templates() -> dict[tuple[str, str], dict]:
    merged = _real_completed_merged()
    templates: dict[tuple[str, str], dict] = {}
    for (type_of_project, coe), group in merged.groupby(["type_of_project", "tech_coe_primary"]):
        templates[(type_of_project, coe)] = _aggregate_role_mix_detailed(group)
    return templates

def get_role_mix_by_category(category: str) -> dict:
    spec = DOCX_CATEGORY_MAP.get(category)
    if spec is None:
        return {"role_mix": {}, "roles": [], "sample_size": 0, "source": "unknown_category"}
    if spec.get("docx_given"):
        return {"role_mix": _DND_TEMPLATE, "roles": _docx_template_to_roles(_DND_TEMPLATE), "sample_size": None, "source": "docx_given"}

    merged = _real_completed_merged()
    mask = pd.Series(True, index=merged.index)
    if "type_of_project" in spec:
        mask &= merged["type_of_project"] == spec["type_of_project"]
    if "tech_coe_any" in spec:
        mask &= merged["tech_coe"].fillna("").apply(lambda v: any(k in v for k in spec["tech_coe_any"]))
    if "proposition_coe_any" in spec:
        mask &= merged["proposition_coe"].fillna("").apply(lambda v: any(k in v for k in spec["proposition_coe_any"]))

    filtered = merged[mask]
    if filtered.empty:
        return {"role_mix": {}, "roles": [], "sample_size": 0, "source": "no_data"}
    result = _aggregate_role_mix_detailed(filtered)
    result["resolved_via"] = spec
    return result

def list_docx_categories() -> list[dict]:
    return [{"category": name, **get_role_mix_by_category(name)} for name in DOCX_CATEGORY_MAP]

CANONICAL_COE_MAP: dict[str, list[str]] = {
    "Data Engineering": ["Data Engineering"],
    "BI & Reporting": ["BI and Reporting"],
    "AI & ML": ["Data Science & ML", "Gen AI", "Data Science", "DS/AI", "Software Development and LLMs"],
    "Full Stack Engineering": ["Full Stack Engineering"],
    "TechOps & Automation": ["TechOps and Automation", "TechOps And MS"],
}

def canonical_project_coe(tech_coe: str | None) -> str | None:
    if not tech_coe or pd.isna(tech_coe):
        return None
    v = str(tech_coe)
    for canonical, aliases in CANONICAL_COE_MAP.items():
        if any(a in v for a in aliases):
            return canonical
    return None

def list_coes() -> list[dict]:
    real_complete = _real_completed_merged()[["project_code", "tech_coe"]].drop_duplicates("project_code")
    tech_coe = real_complete["tech_coe"].fillna("")
    result = []
    for canonical, raw_aliases in CANONICAL_COE_MAP.items():
        sample_size = int(tech_coe.apply(lambda v: any(a in v for a in raw_aliases)).sum())
        result.append({"coe": canonical, "sample_size": sample_size})
    return sorted(result, key=lambda c: -c["sample_size"])

def get_role_mix_by_coes(coes: list[str], type_of_project: str | None = None) -> dict:
    if not coes:
        return {"role_mix": {}, "roles": [], "sample_size": 0, "source": "no_coes_selected", "matched_project_codes": []}

    raw_aliases = [alias for coe in coes for alias in CANONICAL_COE_MAP.get(coe, [coe])]
    merged = _real_completed_merged()
    mask = merged["tech_coe"].fillna("").apply(lambda v: any(a in v for a in raw_aliases))
    if type_of_project:
        mask &= merged["type_of_project"] == type_of_project

    filtered = merged[mask]
    if filtered.empty:
        return {"role_mix": {}, "roles": [], "sample_size": 0, "source": "no_data", "matched_project_codes": []}
    result = _aggregate_role_mix_detailed(filtered)
    result["matched_project_codes"] = sorted(filtered["project_code"].unique().tolist())[:10]
    return result

def get_role_mix(type_of_project: str, tech_coe: str | None = None, templates: dict | None = None) -> dict:
    if type_of_project == DND_TACTICAL_BUILD:
        dnd_roles = _docx_template_to_roles(_DND_TEMPLATE)
        return {
            "role_mix": _DND_TEMPLATE,
            "roles": dnd_roles,
            "expected_headcount_common": sum(r["headcount"] for r in dnd_roles),
            "sample_size": None,
            "source": "docx_given",
        }

    if templates is None:
        templates = build_role_mix_templates()
    coe = _primary_coe(tech_coe)

    if (type_of_project, coe) in templates:
        return templates[(type_of_project, coe)]

    same_type = [v for (t, _), v in templates.items() if t == type_of_project]
    if same_type:
        best = max(same_type, key=lambda v: v["sample_size"])
        return {**best, "source": "derived_empirical_type_fallback"}

    if templates:
        best = max(templates.values(), key=lambda v: v["sample_size"])
        return {**best, "source": "derived_empirical_org_fallback"}

    return {"role_mix": {}, "sample_size": 0, "source": "no_data"}

def list_role_mix_templates() -> list[dict]:
    templates = build_role_mix_templates()
    out = [{"type_of_project": DND_TACTICAL_BUILD, "tech_coe": None, **{"role_mix": _DND_TEMPLATE, "sample_size": None, "source": "docx_given"}}]
    for (type_of_project, coe), v in templates.items():
        out.append({"type_of_project": type_of_project, "tech_coe": coe, **v})
    return out
