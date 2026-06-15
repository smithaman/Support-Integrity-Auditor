# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/embeddings/sentence_encoder.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from config.constants import SEVERITY_ANCHORS
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════

# Module-level cache — model loaded once per process
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


def load_encoder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str     = None,
) -> SentenceTransformer:
    """
    Loads and caches a SentenceTransformer model.
    Returns cached instance if already loaded.

    Model options:
        BAAI/bge-small-en-v1.5  — 384-dim, fast, good accuracy (default)
        BAAI/bge-base-en-v1.5   — 768-dim, slower, higher accuracy
        all-MiniLM-L6-v2        — 384-dim, very fast, lower accuracy

    Args:
        model_name : HuggingFace model identifier
        device     : "cpu" | "cuda" | None (auto-detect)

    Returns:
        Loaded SentenceTransformer model
    """
    if model_name in _MODEL_CACHE:
        logger.debug(f"Using cached encoder: {model_name}")
        return _MODEL_CACHE[model_name]

    log_step(logger, f"Loading sentence encoder: {model_name}")

    try:
        model = SentenceTransformer(model_name, device=device)
        _MODEL_CACHE[model_name] = model

        # Log embedding dimension
        dim = model.get_sentence_embedding_dimension()
        log_success(logger, f"Encoder loaded — dim={dim} | device={model.device}")

        return model

    except Exception as e:
        raise RuntimeError(
            f"Failed to load encoder '{model_name}': {e}\n"
            f"Run: pip install sentence-transformers"
        )


def get_embedding_dim(model_name: str = "BAAI/bge-small-en-v1.5") -> int:
    """Returns the embedding dimension for a given model."""
    dim_map = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5":  768,
        "all-MiniLM-L6-v2":       384,
    }
    return dim_map.get(model_name, 384)


# ══════════════════════════════════════════════════════════════
#  ENCODING
# ══════════════════════════════════════════════════════════════

def encode_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int  = 64,
    normalize: bool  = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encodes a list of texts into normalized embedding vectors.

    Normalization is required for cosine similarity via
    dot product (FAISS IndexFlatIP).

    Args:
        texts         : List of text strings to encode
        model         : Loaded SentenceTransformer model
        batch_size    : Number of texts per batch
        normalize     : L2-normalize embeddings (required for cosine sim)
        show_progress : Show tqdm progress bar

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32
    """
    if not texts:
        raise ValueError("texts list is empty")

    logger.info(f"Encoding {len(texts):,} texts | batch_size={batch_size}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    # Ensure float32 — required by FAISS
    embeddings = embeddings.astype(np.float32)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def encode_single(
    text: str,
    model: SentenceTransformer,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encodes a single text string.
    Used during inference for one-ticket analysis.

    Returns:
        np.ndarray of shape (embedding_dim,), dtype float32
    """
    emb = model.encode(
        [text],
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return emb[0].astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  ANCHOR EMBEDDINGS
# ══════════════════════════════════════════════════════════════

def build_anchor_embeddings(
    model: SentenceTransformer,
    anchors: Dict = None,
) -> Dict[int, np.ndarray]:
    """
    Encodes severity anchor texts into embeddings.

    Supports both:
        str   → single anchor sentence (old format)
        list  → multiple anchor phrases (new format)

    For list anchors: encodes all phrases and averages
    their embeddings → more stable centroid representation.
    """
    if anchors is None:
        anchors = SEVERITY_ANCHORS

    log_step(logger, "Building severity anchor embeddings")

    anchor_embeddings = {}

    for level, anchor in anchors.items():

        if isinstance(anchor, list):
            # Encode all phrases and average → stable centroid
            phrase_embs = model.encode(
                anchor,
                normalize_embeddings = True,
                convert_to_numpy     = True,
            )
            # Average and re-normalize
            avg_emb = phrase_embs.mean(axis=0)
            norm    = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm
            anchor_embeddings[level] = avg_emb.astype(np.float32)

        elif isinstance(anchor, str):
            # Single sentence — encode directly
            emb = model.encode(
                [anchor],
                normalize_embeddings = True,
                convert_to_numpy     = True,
            )
            anchor_embeddings[level] = emb[0].astype(np.float32)

        else:
            raise ValueError(
                f"Anchor for level {level} must be str or list. "
                f"Got {type(anchor)}"
            )

        logger.debug(
            f"Anchor {level} — "
            f"{'list' if isinstance(anchor, list) else 'str'} "
            f"({len(anchor) if isinstance(anchor, list) else 1} phrases) "
            f"→ shape {anchor_embeddings[level].shape}"
        )

    log_success(
        logger,
        f"Built {len(anchor_embeddings)} anchor embeddings "
        f"({'multi-phrase' if isinstance(list(anchors.values())[0], list) else 'single-phrase'})"
    )
    return anchor_embeddings


# ══════════════════════════════════════════════════════════════
#  SIMILARITY
# ══════════════════════════════════════════════════════════════

def cosine_similarity_single(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """
    Computes cosine similarity between two normalized vectors.
    Since both are L2-normalized, dot product = cosine similarity.

    Args:
        vec_a : Normalized embedding vector
        vec_b : Normalized embedding vector

    Returns:
        Similarity score between -1.0 and 1.0
    """
    return float(np.dot(vec_a, vec_b))


def compute_anchor_similarities(
    ticket_emb: np.ndarray,
    anchor_embs: Dict[int, np.ndarray],
) -> Dict[int, float]:
    """
    Computes cosine similarity between a ticket embedding
    and all severity anchor embeddings.

    Args:
        ticket_emb  : Single ticket embedding (normalized)
        anchor_embs : Dict from build_anchor_embeddings()

    Returns:
        Dict mapping severity level → similarity score
        e.g. {1: 0.21, 2: 0.35, 3: 0.61, 4: 0.18}
    """
    return {
        level: cosine_similarity_single(ticket_emb, anchor_emb)
        for level, anchor_emb in anchor_embs.items()
    }


def compute_soft_severity_score(
    ticket_emb: np.ndarray,
    anchor_embs: Dict[int, np.ndarray],
) -> float:
    """
    Computes a soft weighted severity score (1.0–4.0).

    Uses similarity-weighted average instead of hard argmax.
    This preserves boundary information — a ticket equally
    similar to Level 2 and Level 3 anchors gets score ~2.5
    rather than being forced into one bucket.

    Args:
        ticket_emb  : Single normalized ticket embedding
        anchor_embs : Dict from build_anchor_embeddings()

    Returns:
        Continuous severity score between 1.0 and 4.0
    """
    sims  = compute_anchor_similarities(ticket_emb, anchor_embs)
    total = sum(sims.values())

    if total == 0:
        return 2.0  # fallback to Medium

    soft_score = sum(
        level * (sim / total)
        for level, sim in sims.items()
    )

    # Clamp to valid range
    return float(np.clip(soft_score, 1.0, 4.0))


def compute_soft_severity_batch(
    ticket_embs: np.ndarray,
    anchor_embs: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Vectorized soft severity scoring for a batch of ticket embeddings.
    Much faster than calling compute_soft_severity_score in a loop.

    Args:
        ticket_embs : np.ndarray of shape (N, dim) — normalized
        anchor_embs : Dict from build_anchor_embeddings()

    Returns:
        np.ndarray of shape (N,) — soft severity scores (1.0–4.0)
    """
    levels = sorted(anchor_embs.keys())                        # [1, 2, 3, 4]
    anchor_matrix = np.stack(
        [anchor_embs[l] for l in levels], axis=0
    ).astype(np.float32)                                       # (4, dim)

    # Dot product matrix: (N, dim) @ (dim, 4) → (N, 4) similarities
    sim_matrix = ticket_embs @ anchor_matrix.T                 # (N, 4)

    # Normalize rows so they sum to 1
    row_sums   = sim_matrix.sum(axis=1, keepdims=True)
    row_sums   = np.where(row_sums == 0, 1.0, row_sums)       # avoid div-by-zero
    weights    = sim_matrix / row_sums                         # (N, 4)

    # Weighted sum across levels
    level_arr  = np.array(levels, dtype=np.float32)            # [1, 2, 3, 4]
    scores     = weights @ level_arr                           # (N,)

    return np.clip(scores, 1.0, 4.0).astype(np.float32)