"""
Sentiment analysis engine for WSR (Weekly Status Report) comments.

Primary model: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
  — transformer-based, understands professional language context and negation.
  — Ideal for business/project management communications (not social media).

Secondary / comparison: VADER (Valence Aware Dictionary and sEntiment Reasoner)
  — lexicon + rule-based, fast, interpretable.
  — Surfaced alongside BERT score so reviewers can see both signals.

Choice rationale: project status comments are professional communications where
context ("we faced issues but team worked hard and we are back on track") matters
more than word frequency — BERT handles that correctly; VADER would score the
negative keywords and miss the recovery sentiment.
"""

import logging
from functools import lru_cache

logger = logging.getLogger("resourceiq.sentiment")

# ── VADER (lightweight, always available) ────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _vader_available = True
except ImportError:
    _vader_available = False
    logger.warning("vaderSentiment not installed — VADER scores will be None")


def _vader_score(text: str) -> dict | None:
    if not _vader_available or not text.strip():
        return None
    s = _vader.polarity_scores(text)
    return {
        "compound": round(s["compound"], 3),
        "pos": round(s["pos"], 3),
        "neu": round(s["neu"], 3),
        "neg": round(s["neg"], 3),
    }


# ── DistilBERT (primary — loaded once, cached) ───────────────────────────────
@lru_cache(maxsize=1)
def _load_bert_pipeline():
    try:
        from transformers import pipeline
        pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=128,
        )
        logger.info("DistilBERT sentiment pipeline loaded")
        return pipe
    except Exception as e:
        logger.warning("DistilBERT unavailable: %s — falling back to VADER only", e)
        return None


def _bert_score(text: str) -> dict | None:
    if not text.strip():
        return None
    pipe = _load_bert_pipeline()
    if pipe is None:
        return None
    try:
        result = pipe(text[:512])[0]
        label = result["label"].lower()          # "positive" | "negative"
        confidence = round(result["score"], 3)
        # Convert to compound-style score: positive → +confidence, negative → -confidence
        compound = confidence if label == "positive" else -confidence
        return {"label": label, "confidence": confidence, "compound": round(compound, 3)}
    except Exception as e:
        logger.warning("DistilBERT inference failed: %s", e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_comment(text: str) -> dict:
    """
    Analyze a single WSR comment. Returns combined BERT + VADER scores.
    Primary label/compound comes from BERT; VADER provided as comparison.
    """
    if not text or not str(text).strip():
        return {
            "label": "neutral",
            "compound": 0.0,
            "risk_signal": "none",
            "bert": None,
            "vader": None,
        }

    text = str(text).strip()
    bert = _bert_score(text)
    vader = _vader_score(text)

    # Primary signal: BERT if available, else VADER compound
    if bert is not None:
        label = bert["label"]
        compound = bert["compound"]
    elif vader is not None:
        c = vader["compound"]
        label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
        compound = c
    else:
        label = "neutral"
        compound = 0.0

    risk_signal = (
        "high" if label == "negative" and abs(compound) >= 0.6 else
        "medium" if label == "negative" else
        "none"
    )

    return {
        "label": label,
        "compound": compound,
        "risk_signal": risk_signal,
        "bert": bert,
        "vader": vader,
    }


def summarize_project_sentiment(comments_with_dates: list[dict]) -> dict:
    """
    Given a list of {"date": str, "comment": str} dicts (newest last),
    return an aggregated sentiment summary with trend.
    """
    scored = []
    for entry in comments_with_dates:
        txt = entry.get("comment", "")
        if not txt or not str(txt).strip():
            continue
        result = analyze_comment(txt)
        scored.append({
            "date": entry.get("date"),
            "comment": txt,
            **result,
        })

    if not scored:
        return {
            "has_data": False,
            "label": None,
            "compound": None,
            "trend": None,
            "risk_signal": "none",
            "latest_comment": None,
            "recent_scores": [],
        }

    compounds = [s["compound"] for s in scored]
    avg = round(sum(compounds) / len(compounds), 3)
    latest = scored[-1]

    # Trend: compare last 2 vs prior 2
    trend = None
    if len(scored) >= 4:
        recent_avg = sum(compounds[-2:]) / 2
        prior_avg = sum(compounds[-4:-2]) / 2
        diff = recent_avg - prior_avg
        if diff > 0.15:
            trend = "improving"
        elif diff < -0.15:
            trend = "deteriorating"
        else:
            trend = "stable"
    elif len(scored) >= 2:
        diff = compounds[-1] - compounds[0]
        trend = "improving" if diff > 0.15 else "deteriorating" if diff < -0.15 else "stable"

    return {
        "has_data": True,
        "label": latest["label"],
        "compound": latest["compound"],
        "avg_compound": avg,
        "trend": trend,
        "risk_signal": latest["risk_signal"],
        "latest_comment": latest["comment"],
        "recent_scores": [
            {"date": s["date"], "label": s["label"], "compound": s["compound"], "comment": s["comment"]}
            for s in scored[-5:]
        ],
    }
