"""Fallback required-skills inference for pipeline demand rows that have no
real skillset text of their own.

~53% of real pipeline_forecast rows (156/293) have a blank `skillset` field --
HubSpot deals where nobody typed a skills list. Left alone, every candidate
for that role scores bucket="not_assessed" (ranked by availability/competency
only) and AI Semantic Match short-circuits with "no skillset". This module
infers a real, defensible required-skills list instead, in priority order:

  1. The deal's real `solution` field (proposition_coe vocabulary) is bridged
     to a canonical tech_coe via real historical COMPLETE/ACTIVE projects that
     share that same real proposition_coe (majority vote) -- the same
     real-data bridge experience_engine.py uses per-employee, aggregated here
     at the population level instead. That canonical CoE's real, curated
     skill list (COE_Skills_Mapping.csv) becomes the inferred skillset.
  2. When no solution is set, or it doesn't resolve to any canonical CoE with
     real project history, the same sentence-transformer embedding model used
     for AI Semantic Match / semantic skill scoring elsewhere in this app
     picks the nearest-matching CoE by comparing whatever real text IS on the
     row (decoded role code + comments) against each CoE's real skill list --
     "closest matching skills we could find", explicitly labeled as such.
  3. If there's truly nothing to go on (embedding unavailable, or no CoE
     clears the minimum similarity bar), the Cross-Functional/Core Consulting
     Skills row from COE_Skills_Mapping.csv -- the org's own real
     "applies to every CoE" baseline -- is the last resort.

Every inference carries a `source`/`confidence` field so callers/UI never
present this as if HubSpot/the deal author specified it themselves.
"""
from collections import Counter

import pandas as pd

from app.core.adapter import get_adapter
from app.engines import embedding_engine
from app.engines.coe_taxonomy import CANONICAL_COES, resolve_coe_label
from app.engines.resource_code_decoder import decode_resource_code
from app.engines.role_mix_engine import canonical_project_coe

REAL_STATUSES = frozenset({"COMPLETE", "ACTIVE"})
TOP_N_INFERRED_SKILLS = 15
CROSS_FUNCTIONAL_COE_LABEL = "Cross-Functional / Core Consulting Skills (applies across all COEs)"
# Minimum embedding cosine similarity to treat a CoE as a genuine match rather
# than a coin-flip guess -- below this, "closest matching skills" would be
# more misleading than useful, so the Cross-Functional baseline is used instead.
MIN_EMBEDDING_CONFIDENCE = 0.25


def _solution_to_coe_map() -> dict[str, str]:
    """Real proposition_coe (the same vocabulary as a pipeline deal's real
    `solution` field) -> the canonical tech_coe that real COMPLETE/ACTIVE
    projects sharing that proposition_coe most often carry. Population-level
    version of the per-employee bridge in experience_engine.py. Votes one
    canonical CoE per project (via canonical_project_coe on the whole raw
    tech_coe field), matching the exact same whole-string-match convention
    role_mix_engine.list_coes()/get_role_mix_by_coes() already use for this
    field -- not a second, differently-behaved way of reading it."""
    adapter = get_adapter()
    projects = adapter.get_projects()
    real = projects[projects["project_status"].isin(REAL_STATUSES)]

    mapping: dict[str, str] = {}
    for solution, group in real.groupby("proposition_coe"):
        if pd.isna(solution) or not str(solution).strip():
            continue
        # pd.notna(), not a bare truthy check: Series.apply() silently upcasts
        # canonical_project_coe's `None` returns to float NaN when collecting
        # them into the output Series (an object-dtype -> mixed-type pandas
        # quirk) -- and `nan` is truthy in plain Python, so `if canon` would
        # silently let it through as a spurious "canonical" CoE.
        counts: Counter = Counter(
            canon for canon in group["tech_coe"].apply(canonical_project_coe) if pd.notna(canon)
        )
        if counts:
            mapping[str(solution).strip()] = counts.most_common(1)[0][0]
    return mapping


def _coe_skill_lists() -> dict[str, list[str]]:
    """Real per-CoE required-skill lists from COE_Skills_Mapping.csv, keyed by
    the same 5 canonical CoE names used everywhere else in this app (plus a
    "_cross_functional" bucket for the always-applicable baseline row). Two
    raw rows (Platform Engineering, TechOps and Automation) both resolve to
    the same canonical "TechOps & Automation" -- their real skill lists are
    unioned rather than one silently overwriting the other."""
    adapter = get_adapter()
    df = adapter.get_coe_skills_mapping()

    by_coe: dict[str, list[str]] = {}
    cross_functional: list[str] = []
    for row in df.itertuples(index=False):
        raw_label = str(row.coe).strip()
        skills = [s.strip() for s in str(row.skills_combined).split(",") if s.strip()]
        if raw_label == CROSS_FUNCTIONAL_COE_LABEL:
            cross_functional = skills
            continue
        canonical = resolve_coe_label(raw_label)
        if canonical not in CANONICAL_COES:
            # "BI & Reporting, Consulting" resolves via resolve_coe_label's
            # title-case fallback to itself, not to a canonical name -- map it
            # explicitly rather than silently dropping a real 7-skill list.
            if "bi" in raw_label.lower() and "report" in raw_label.lower():
                canonical = "BI & Reporting"
            else:
                continue
        existing = by_coe.setdefault(canonical, [])
        for s in skills:
            if s not in existing:
                existing.append(s)
    by_coe["_cross_functional"] = cross_functional
    return by_coe


_coe_embedding_index: dict | None = None
_coe_embedding_fingerprint: tuple | None = None


def _coe_reference_embedding_index() -> dict | None:
    """One embedding vector per canonical CoE, encoded from its real
    COE_Skills_Mapping.csv skill list -- cached until the reference file
    changes (mirrors embedding_engine's own fingerprint-cache pattern)."""
    global _coe_embedding_index, _coe_embedding_fingerprint
    skill_lists = _coe_skill_lists()
    fingerprint = tuple(sorted((k, tuple(v)) for k, v in skill_lists.items()))
    if _coe_embedding_index is not None and _coe_embedding_fingerprint == fingerprint:
        return _coe_embedding_index

    index: dict = {}
    for coe, skills in skill_lists.items():
        if coe == "_cross_functional" or not skills:
            continue
        vec = embedding_engine.embed_jobspec(", ".join(skills))
        if vec is not None:
            index[coe] = vec

    _coe_embedding_index = index
    _coe_embedding_fingerprint = fingerprint
    return index or None


def infer_skills_for_coes(tech_coes) -> list[str]:
    """Real per-CoE required skills (COE_Skills_Mapping.csv) for a set/list of
    raw tech_coe-vocabulary strings (e.g. Forecast spec `coes`/category
    aliases), resolved to canonical CoEs and unioned. Used to backfill a
    required_skills list when a caller only knows *which CoE* a role belongs
    to, not an explicit skillset -- without this, skill/competency scoring is
    skipped entirely (skill_index stays None) and every candidate shows 0%."""
    skill_lists = _coe_skill_lists()
    canon_coes = {canonical_project_coe(t) for t in tech_coes}
    canon_coes.discard(None)
    phrases: list[str] = []
    for coe in canon_coes:
        for s in skill_lists.get(coe, []):
            if s not in phrases:
                phrases.append(s)
    return phrases[:TOP_N_INFERRED_SKILLS]


def infer_required_skills_for_pipeline_row(
    resources_requested,
    solution: str | None,
    comments=None,
) -> dict:
    """Returns a dict always safe to feed straight into get_recommendations as
    skillset_text: {skillset_text, required_skills, source, matched_coe,
    confidence, detail}. Never returns an empty skillset_text -- worst case
    falls through to the real Cross-Functional baseline skills."""
    skill_lists = _coe_skill_lists()

    # 1) Solution -> real historical CoE
    if solution:
        coe = _solution_to_coe_map().get(str(solution).strip())
        if coe and skill_lists.get(coe):
            phrases = skill_lists[coe][:TOP_N_INFERRED_SKILLS]
            return {
                "skillset_text": ", ".join(phrases),
                "required_skills": phrases,
                "source": "coe_mapping",
                "matched_coe": coe,
                "confidence": "medium",
                "detail": (
                    f"No skillset was given for this role -- inferred from real historical "
                    f"'{solution}' engagements, which consistently drew on {coe} skills, "
                    f"cross-referenced against JMAN's real {coe} skill list."
                ),
            }

    # 2) Embedding: nearest CoE by whatever real text IS on the row
    designations = decode_resource_code(resources_requested) if resources_requested else []
    query_parts = list(designations)
    if not designations and resources_requested:
        query_parts.append(str(resources_requested))
    if isinstance(comments, str) and comments.strip():
        query_parts.append(comments.strip())
    if solution:
        query_parts.append(str(solution).strip())
    query_text = " | ".join(p for p in query_parts if p)

    if query_text:
        job_vec = embedding_engine.embed_jobspec(query_text)
        coe_index = _coe_reference_embedding_index()
        if job_vec is not None and coe_index:
            sims = embedding_engine.batch_cosine_similarity(job_vec, coe_index, use_pinecone=False)
            if sims:
                best_coe, best_score = max(sims.items(), key=lambda kv: kv[1])
                if best_score >= MIN_EMBEDDING_CONFIDENCE and skill_lists.get(best_coe):
                    phrases = skill_lists[best_coe][:TOP_N_INFERRED_SKILLS]
                    return {
                        "skillset_text": ", ".join(phrases),
                        "required_skills": phrases,
                        "source": "embedding_match",
                        "matched_coe": best_coe,
                        "confidence": "low",
                        "match_score_pct": round(best_score * 100, 1),
                        "detail": (
                            f"No real proposition/CoE link was found for this role -- nearest match via "
                            f"AI semantic similarity ({round(best_score * 100)}% confidence) to real "
                            f"{best_coe} work, based on the requested role and any notes on the deal."
                        ),
                    }

    # 3) Last resort: the org's own real cross-CoE baseline skills
    fallback = skill_lists.get("_cross_functional") or []
    return {
        "skillset_text": ", ".join(fallback),
        "required_skills": fallback,
        "source": "org_fallback",
        "matched_coe": None,
        "confidence": "very_low",
        "detail": (
            "No CoE or historical match could be inferred for this role -- showing JMAN's general "
            "cross-CoE consulting skills as a baseline; ranked mostly by availability/competency."
        ),
    }
