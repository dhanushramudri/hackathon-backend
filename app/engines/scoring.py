import re

import pandas as pd

SKILL_WEIGHT = 0.5
COMPETENCY_WEIGHT = 0.3
AVAILABILITY_WEIGHT = 0.2

ELIGIBLE_THRESHOLD = 0.6
TRAINABLE_THRESHOLD = 0.3

IMPUTED_SKILL_DISCOUNT = 0.6
MAX_SUBSKILL_WORDS = 4

def parse_requested_pct(raw) -> float:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 100.0
    match = re.search(r"\d+", str(raw))
    return float(match.group()) if match else 100.0

_ITEM_SEPARATOR = re.compile(r"\.\s*,\s*")
_LABEL_SPLIT = re.compile(r"\s+-\s+")

def tokenize_skillset(text: str) -> list[str]:
    if not text or pd.isna(text):
        return []
    raw = str(text).strip()

    if " - " in raw:
        items = _ITEM_SEPARATOR.split(raw) if _ITEM_SEPARATOR.search(raw) else re.split(r"[,;]", raw)
        labels = []
        for item in items:
            item = item.strip().strip(".")
            if not item:
                continue
            label = _LABEL_SPLIT.split(item, maxsplit=1)[0].strip()
            if label:
                labels.append(label.lower())
        if labels:
            return labels

    return [p.strip().lower() for p in re.split(r"[,;]", raw) if p.strip()]

MAX_ENRICHED_PHRASES = 20

def enrich_required_phrases(required_phrases: list[str], skillset_ref: pd.DataFrame) -> list[str]:
    if skillset_ref.empty or not required_phrases:
        return required_phrases
    by_category = skillset_ref.groupby(skillset_ref["coe_skill"].fillna("").str.lower())["skills_combined"].apply(
        lambda s: ", ".join(str(v) for v in s if pd.notna(v))
    )

    enriched = list(required_phrases)
    for phrase in required_phrases:
        for coe_skill, combined_skills in by_category.items():
            if coe_skill and (phrase.lower() in coe_skill or coe_skill in phrase.lower()):
                enriched.extend(tokenize_skillset(combined_skills))
    return list(dict.fromkeys(enriched))[:MAX_ENRICHED_PHRASES]

_skill_index_cache: dict[str, dict[str, float]] | None = None
_skill_index_fingerprint: tuple | None = None

def _fingerprint(skills_df: pd.DataFrame) -> tuple:
    return (len(skills_df), int(pd.util.hash_pandas_object(skills_df, index=False).sum()))

def build_employee_skill_index(skills_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    global _skill_index_cache, _skill_index_fingerprint

    fingerprint = _fingerprint(skills_df)
    if _skill_index_cache is not None and fingerprint == _skill_index_fingerprint:
        return _skill_index_cache

    subskill_word_count = skills_df["subskill"].astype(str).str.split().str.len()
    subskill_specific = skills_df["subskill"].where(subskill_word_count <= MAX_SUBSKILL_WORDS)

    weight = skills_df["skill_source"].eq("observed").map({True: 1.0, False: IMPUTED_SKILL_DISCOUNT})
    long = pd.DataFrame(
        {
            "employee_id": pd.concat([skills_df["employee_id"]] * 3, ignore_index=True),
            "weight": pd.concat([weight] * 3, ignore_index=True),
            "text": pd.concat(
                [skills_df["skill"], subskill_specific, skills_df["coe_skill"]], ignore_index=True
            ),
        }
    )
    long = long.dropna(subset=["text"])
    long["tokens"] = long["text"].astype(str).str.lower().str.split(r"\W+")
    long = long.explode("tokens")
    long = long[long["tokens"].str.len() > 2]

    best = long.groupby(["employee_id", "tokens"])["weight"].max()

    index: dict[str, dict[str, float]] = {}
    for (emp_id, tok), w in best.items():
        index.setdefault(emp_id, {})[tok] = w

    _skill_index_cache = index
    _skill_index_fingerprint = fingerprint
    return index

COMMON_TOKEN_THRESHOLD = 0.4

def common_skill_tokens(index: dict[str, dict[str, float]], threshold: float = COMMON_TOKEN_THRESHOLD) -> frozenset:
    if not index:
        return frozenset()
    counts: dict[str, int] = {}
    for tokens in index.values():
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
    n = len(index)
    return frozenset(tok for tok, c in counts.items() if c / n > threshold)

def score_skill_match(
    required_phrases: list[str], employee_tokens: dict[str, float], common_tokens: frozenset = frozenset()
) -> dict:
    if not required_phrases:
        return {"score": 0.5, "matched": [], "missing": [], "confidence": "no_requirement"}

    matched, missing, weights = [], [], []
    for phrase in required_phrases:
        phrase_tokens = [t for t in re.split(r"\W+", phrase.lower()) if len(t) > 2]
        specific_tokens = [t for t in phrase_tokens if t not in common_tokens]
        candidate_tokens = specific_tokens or phrase_tokens
        best_weight = max((employee_tokens.get(tok, 0.0) for tok in candidate_tokens), default=0.0)
        if best_weight > 0:
            matched.append(phrase)
            weights.append(best_weight)
        else:
            missing.append(phrase)

    score = sum(weights) / len(required_phrases) if required_phrases else 0.0
    confidence = "observed" if weights and min(weights) == 1.0 else ("imputed" if weights else "no_match")
    return {"score": float(round(min(score, 1.0), 3)), "matched": matched, "missing": missing, "confidence": confidence}

def composite_score(skill_score: float, competency_score: float, availability_score: float) -> float:
    return float(
        round(
            skill_score * SKILL_WEIGHT + competency_score * COMPETENCY_WEIGHT + availability_score * AVAILABILITY_WEIGHT,
            3,
        )
    )

def bucket(skill_score: float, confidence: str | None = None) -> str:
    if confidence == "no_requirement":
        return "not_assessed"
    score = skill_score
    if score >= ELIGIBLE_THRESHOLD:
        return "eligible"
    if score >= TRAINABLE_THRESHOLD:
        return "trainable"
    return "gap"

def staffing_signal(bucket_value: str) -> str:
    if bucket_value == "eligible":
        return "redeploy"
    if bucket_value == "trainable":
        return "redeploy_with_training"
    if bucket_value == "not_assessed":
        return "not_assessed"
    return "hire"

def explain_candidate(
    employee_id: str,
    job_name: str | None,
    bucket_value: str,
    skill_result: dict,
    competency_score: float,
    available_pct: float,
    requested_pct: float,
    meets_requested_capacity: bool,
) -> str:
    matched = skill_result["matched"]
    missing = skill_result["missing"]
    n_required = len(matched) + len(missing)

    if n_required == 0:
        skill_clause = "no specific skills were requested for this role, so skill fit could not be assessed"
    else:
        skill_clause = f"matches {len(matched)} of {n_required} required skill(s)"
        if matched:
            skill_clause += f" ({', '.join(matched)})"
        if missing:
            skill_clause += f", missing {', '.join(missing)}"
        confidence_note = {
            "observed": "based on directly observed skill records",
            "imputed": "based on inferred/peer-imputed skill records, lower confidence",
            "no_match": "no overlapping skill records found",
        }.get(skill_result["confidence"], "")
        if confidence_note:
            skill_clause += f" ({confidence_note})"

    competency_clause = f"competency score {competency_score:.2f}/1.00"
    availability_clause = f"{available_pct:.0f}% available" + (
        f", meets the requested {requested_pct:.0f}%"
        if meets_requested_capacity
        else f", below the requested {requested_pct:.0f}%"
    )
    bucket_clause = {
        "eligible": "a strong internal fit -- recommended for direct redeployment",
        "trainable": "a partial fit -- redeployable with targeted upskilling",
        "gap": "not a viable internal fit on verified skills -- a hire signal, not a redeploy",
        "not_assessed": "skill fit not assessed -- no skillset was specified for this role; ranked by availability and competency only",
    }[bucket_value]

    name = f"{employee_id} ({job_name})" if job_name else employee_id
    return f"{name} {skill_clause}; {competency_clause}; {availability_clause}. Overall: {bucket_clause}."
