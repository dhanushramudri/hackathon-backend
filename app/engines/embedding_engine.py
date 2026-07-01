"""
Semantic embedding engine for skill-profile matching.

Architecture
------------
Layer 1 (encoding)  : sentence-transformers — converts text to 384-dim unit vectors.
Layer 2 (storage)   : Pinecone when PINECONE_API_KEY + PINECONE_HOST are set in .env,
                      otherwise an in-process numpy dict (fast enough for ≤ 50 000 employees).

Flow
----
  Offline  : encode each employee's skill profile → upsert to Pinecone (or cache in RAM).
             Re-encoded automatically when the skills data fingerprint changes.
  At query : encode the job description (1 call, ~20 ms) → Pinecone query OR numpy matmul
             → semantic similarity scores for every employee in ~1–5 ms.

Fallback chain
--------------
  Pinecone configured  → use Pinecone (persistent, scalable, survives restarts)
  Pinecone not set     → numpy in-memory (rebuilt each restart, fine for the demo)
  sentence-transformers not installed → None everywhere; callers fall back to word-token matching.

Recommendation service integration
-----------------------------------
  Calling code passes emp_embedding_index=None for single-row calls (built here on demand,
  cached internally) or passes a pre-built dict/None for multi-role batch calls.
  A None return means "embeddings unavailable" — caller silently uses word matching only.
"""

import hashlib
import logging
import os
import pathlib
import threading
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, ~90 MB on first download, then cached locally
_PINECONE_INDEX_NAME = "resourceiq-skills"
_PINECONE_DIMENSION = 384
_PINECONE_METRIC = "cosine"

# Disk cache lives two levels above app/ (i.e. next to requirements.txt) so uvicorn's
# WatchFiles doesn't see writes and trigger a reload loop.
_DISK_CACHE_DIR = pathlib.Path(__file__).parents[3] / ".embedding_cache"
_DISK_CACHE_DIR.mkdir(exist_ok=True)

# ── encoding model ────────────────────────────────────────────────────────────
_model = None
_model_available: Optional[bool] = None   # None=untried, True=ready, False=unavailable


def _get_model():
    global _model, _model_available
    if _model_available is False:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        logger.info("Loading embedding model %s (downloads once, then cached)…", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        _model_available = True
        logger.info("Embedding model ready.")
        return _model
    except ImportError:
        logger.warning("sentence-transformers not installed — run: pip install sentence-transformers")
        _model_available = False
        return None
    except Exception as exc:
        logger.warning("Embedding model failed to load (%s). Using word-token matching.", exc)
        _model_available = False
        return None


# ── Pinecone connection ───────────────────────────────────────────────────────
_pinecone_index = None
_pinecone_ready: Optional[bool] = None


def _get_pinecone_index():
    global _pinecone_index, _pinecone_ready
    if _pinecone_ready is False:
        return None
    if _pinecone_index is not None:
        return _pinecone_index

    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_HOST")
    if not api_key:
        logger.info("PINECONE_API_KEY not set — using in-memory numpy vector store.")
        _pinecone_ready = False
        return None

    try:
        from pinecone import Pinecone, ServerlessSpec  # noqa: PLC0415
        pc = Pinecone(api_key=api_key)

        # If host is provided, connect directly to the index (fastest path)
        if host:
            _pinecone_index = pc.Index(host=host)
            _pinecone_ready = True
            logger.info("Connected to Pinecone index via host URL.")
            return _pinecone_index

        # Otherwise create/reuse by name
        existing = [idx["name"] for idx in pc.list_indexes()]
        if _PINECONE_INDEX_NAME not in existing:
            logger.info("Creating Pinecone index '%s'…", _PINECONE_INDEX_NAME)
            pc.create_index(
                name=_PINECONE_INDEX_NAME,
                dimension=_PINECONE_DIMENSION,
                metric=_PINECONE_METRIC,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _pinecone_index = pc.Index(_PINECONE_INDEX_NAME)
        _pinecone_ready = True
        logger.info("Connected to Pinecone index '%s'.", _PINECONE_INDEX_NAME)
        return _pinecone_index

    except ImportError:
        logger.warning("pinecone not installed — run: pip install pinecone")
        _pinecone_ready = False
        return None
    except Exception as exc:
        logger.warning("Pinecone connection failed (%s). Falling back to numpy.", exc)
        _pinecone_ready = False
        return None


# ── employee text builder ─────────────────────────────────────────────────────

def _employee_skill_text(emp_id: str, skills_df: pd.DataFrame) -> str:
    emp_rows = skills_df[skills_df["employee_id"] == emp_id]
    scores = pd.to_numeric(emp_rows["score"], errors="coerce").fillna(0)
    observed_mask = emp_rows["skill_source"].eq("observed") & (scores > 0)
    rows = emp_rows[observed_mask] if observed_mask.any() else emp_rows[scores > 0]
    if rows.empty:
        return ""
    parts: list[str] = []
    seen: set[str] = set()
    for _, r in rows.iterrows():
        for field in ("skill", "subskill", "coe_skill"):
            val = str(r.get(field) or "").strip()
            if val and val.lower() != "nan":
                key = val.lower()
                if key not in seen:
                    seen.add(key)
                    parts.append(val)
    return " | ".join(parts)


def _skills_fingerprint(skills_df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(skills_df, index=False).values.tobytes()
    ).hexdigest()


# ── disk cache helpers ───────────────────────────────────────────────────────

def _disk_cache_path(fingerprint: str) -> pathlib.Path:
    return _DISK_CACHE_DIR / f"emb_{fingerprint}.npz"


def _load_from_disk(fingerprint: str) -> Optional[dict[str, np.ndarray]]:
    path = _disk_cache_path(fingerprint)
    if not path.exists():
        return None
    try:
        # Stored as two arrays: 'ids' (string array) + 'matrix' (N×384 float32).
        # Single np.load call decompresses everything at once — much faster than
        # one decompress per employee-key.
        data = np.load(str(path), allow_pickle=False)
        ids: list[str] = data["ids"].tolist()
        matrix: np.ndarray = data["matrix"]
        result = {eid: matrix[i] for i, eid in enumerate(ids)}
        logger.info("Loaded %d employee embeddings from disk cache.", len(result))
        return result
    except Exception as exc:
        logger.warning("Disk cache load failed (%s). Re-encoding.", exc)
        return None


def _save_to_disk(fingerprint: str, index: dict[str, np.ndarray]) -> None:
    path = _disk_cache_path(fingerprint)
    try:
        ids = np.array(list(index.keys()))
        matrix = np.stack(list(index.values())).astype(np.float32)
        np.savez_compressed(str(path), ids=ids, matrix=matrix)
        # Remove stale cache files
        for old in _DISK_CACHE_DIR.glob("emb_*.npz"):
            if old != path:
                old.unlink(missing_ok=True)
        logger.info("Saved %d employee embeddings to disk cache.", len(index))
    except Exception as exc:
        logger.warning("Disk cache save failed (%s). Continuing in-memory only.", exc)


# ── in-memory numpy fallback cache ───────────────────────────────────────────
_numpy_cache: Optional[dict[str, np.ndarray]] = None
_numpy_fingerprint: Optional[str] = None
_pinecone_synced_fingerprint: Optional[str] = None   # tracks what's in Pinecone
_build_lock = threading.Lock()  # prevents concurrent index builds on the same process


# ── public API ───────────────────────────────────────────────────────────────

def build_employee_embedding_index(skills_df: pd.DataFrame) -> Optional[dict[str, np.ndarray]]:
    """Pre-computes and caches skill embeddings for all employees.

    If Pinecone is configured: upserts vectors there (persistent across restarts).
    Otherwise: stores unit-norm vectors in a numpy dict in RAM.

    Returns the in-memory numpy dict regardless of backend (used for fast batch
    cosine similarity or as the fallback when Pinecone is not available).
    Returns None if the embedding model cannot be loaded.
    """
    global _numpy_cache, _numpy_fingerprint, _pinecone_synced_fingerprint

    model = _get_model()
    if model is None:
        return None

    fingerprint = _skills_fingerprint(skills_df)

    # Fast path 1: in-memory cache is current (no lock needed — pure read)
    if _numpy_cache is not None and _numpy_fingerprint == fingerprint:
        if _pinecone_synced_fingerprint != fingerprint:
            _try_sync_pinecone(_numpy_cache, fingerprint)
        return _numpy_cache

    # Slow path — one thread builds, all others wait, then get the cache hit
    with _build_lock:
        # Re-check inside lock: another thread may have built it while we waited
        if _numpy_cache is not None and _numpy_fingerprint == fingerprint:
            return _numpy_cache

        # Fast path 2: disk cache exists for this fingerprint (survives restarts)
        disk_result = _load_from_disk(fingerprint)
        if disk_result is not None:
            _numpy_cache = disk_result
            _numpy_fingerprint = fingerprint
            _try_sync_pinecone(disk_result, fingerprint)
            return disk_result

        # Encode all employees (first run or data changed)
        employee_ids = skills_df["employee_id"].unique().tolist()
        texts = [_employee_skill_text(eid, skills_df) for eid in employee_ids]
        valid = [(eid, txt) for eid, txt in zip(employee_ids, texts) if txt]

        if not valid:
            _numpy_cache = {}
            _numpy_fingerprint = fingerprint
            return _numpy_cache

        valid_ids, valid_texts = zip(*valid)
        logger.info("Encoding skill profiles for %d employees…", len(valid_ids))
        vecs: np.ndarray = model.encode(
            list(valid_texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        result = {eid: vecs[i] for i, eid in enumerate(valid_ids)}

        _numpy_cache = result
        _numpy_fingerprint = fingerprint
        logger.info("Embedding index ready (%d employees).", len(result))

        _save_to_disk(fingerprint, result)
        _try_sync_pinecone(result, fingerprint)
        return result


def _disable_pinecone(reason: str) -> None:
    global _pinecone_ready, _pinecone_index
    _pinecone_ready = False
    _pinecone_index = None
    logger.warning("Pinecone disabled for this session: %s. Using numpy vector store.", reason)


def _try_sync_pinecone(index: dict[str, np.ndarray], fingerprint: str) -> None:
    """Upserts all employee vectors to Pinecone (fire-and-forget, no exception raised)."""
    global _pinecone_synced_fingerprint
    pc_index = _get_pinecone_index()
    if pc_index is None:
        return
    try:
        # Upsert in batches of 100 (Pinecone recommended batch size)
        items = list(index.items())
        batch_size = 100
        total = 0
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            vectors = [{"id": eid, "values": vec.tolist()} for eid, vec in batch]
            pc_index.upsert(vectors=vectors)
            total += len(batch)
        _pinecone_synced_fingerprint = fingerprint
        logger.info("Synced %d employee vectors to Pinecone.", total)
    except Exception as exc:
        _disable_pinecone(str(exc))


_jobspec_cache: dict[str, np.ndarray] = {}  # text → unit vector; unbounded but ~150 pipeline rows max


def embed_jobspec(text) -> Optional[np.ndarray]:
    """Encodes a job description / skillset string into a unit-norm vector.
    Results are cached in-process so the same pipeline-row skillset text is only
    encoded once regardless of how many times it's queried across services."""
    model = _get_model()
    if model is None or not isinstance(text, str) or not text.strip():
        return None
    key = text.strip()
    cached = _jobspec_cache.get(key)
    if cached is not None:
        return cached
    vec: np.ndarray = model.encode([key], normalize_embeddings=True)
    _jobspec_cache[key] = vec[0]
    return vec[0]


def batch_cosine_similarity(
    job_vec: np.ndarray,
    emp_index: dict[str, np.ndarray],
    *,
    use_pinecone: bool = True,
) -> dict[str, float]:
    """Returns cosine similarity of job_vec against all employees.

    If Pinecone is available and use_pinecone=True: queries Pinecone for the top K
    matches (very fast, scales to millions). Employees not in the top K get score 0.

    Falls back to a single numpy matmul when Pinecone is unavailable (sub-ms for
    ≤ 50 000 employees — perfectly fast for production-scale JMAN org).
    """
    if not emp_index:
        return {}

    if use_pinecone:
        pc_index = _get_pinecone_index()
        if pc_index is not None:
            try:
                # top_k = all employees so ranking is identical to numpy mode
                top_k = min(len(emp_index), 10_000)
                result = pc_index.query(vector=job_vec.tolist(), top_k=top_k)
                scores = {m["id"]: float(np.clip(m["score"], 0.0, 1.0)) for m in result["matches"]}
                # Fill in 0 for any employee not returned (score below threshold)
                for eid in emp_index:
                    scores.setdefault(eid, 0.0)
                return scores
            except Exception as exc:
                _disable_pinecone(str(exc))

    # numpy fallback — single matmul, unit vectors → dot product = cosine similarity
    ids = list(emp_index.keys())
    matrix = np.stack([emp_index[eid] for eid in ids])   # (N, 384)
    sims: np.ndarray = np.clip(matrix @ job_vec, 0.0, 1.0)
    return dict(zip(ids, sims.tolist()))


def is_available() -> bool:
    """True when the embedding model is loaded and ready."""
    return _get_model() is not None
