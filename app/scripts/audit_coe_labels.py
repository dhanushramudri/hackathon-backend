"""One-off diagnostic: lists every raw CoE-ish label found in the data that
doesn't resolve to one of the 5 canonical CoEs via coe_taxonomy. Run this
manually (not part of the app/test suite) whenever the underlying pipeline
sheet, skills data, or deal solution/tech_coe fields get new rows, to catch
unmapped spellings before they cause a silent ranking miss (a request whose
CoE never matches any candidate's home CoE, same bug class as the
"Data Science & AI" / "TechOps and Automation" / "Platform Engineering" cases).

Usage (from project root, with the app's venv active):
    python -m scripts.audit_coe_labels
"""
from app.core.adapter import get_adapter
from app.engines.coe_taxonomy import is_known_coe


def audit_pipeline_skillset_coe_skill() -> list[str]:
    sheet = get_adapter().get_pipeline_skillset()
    raw_labels = sheet["coe_skill"].dropna().unique()
    return sorted({l for l in raw_labels if not is_known_coe(l)})


def audit_skills_df_coe() -> list[str]:
    """Home-CoE source (employee_coe.py reads this same column)."""
    skills = get_adapter().get_skills()
    raw_labels = skills["coe"].dropna().unique()
    return sorted({l for l in raw_labels if not is_known_coe(l)})


def audit_pipeline_forecast_solution() -> list[str]:
    """Deal-level solution/tech field -- this is where "Platform Engineering"
    came from. Column name may differ; adjust to match your pipeline_forecast
    schema (e.g. "solution", "tech_coe", "proposition_coe")."""
    pipeline = get_adapter().get_pipeline_forecast()
    col = "solution" if "solution" in pipeline.columns else None
    if col is None:
        return []
    raw_labels = pipeline[col].dropna().unique()
    return sorted({l for l in raw_labels if not is_known_coe(l)})


if __name__ == "__main__":
    print("Unresolved labels in pipeline_skillset.coe_skill:")
    print(audit_pipeline_skillset_coe_skill())
    print()
    print("Unresolved labels in skills.coe (employee home-CoE source):")
    print(audit_skills_df_coe())
    print()
    print("Unresolved labels in pipeline_forecast.solution (deal-level):")
    print(audit_pipeline_forecast_solution())