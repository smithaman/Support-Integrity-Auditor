"""
build_index.py
FAISS Index Builder
Loads ticket embeddings from outputs/embeddings/.
Builds a FAISS IndexFlatIP (cosine similarity on normalized vectors).
Saves index to outputs/faiss_index/.
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/retrieval/build_index.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, List, Optional, Tuple


import faiss
import numpy as np
import pandas as pd

from config.constants import (
    COL_TICKET_ID,
    COL_SUBJECT,
    COL_PRIORITY,
    COL_CATEGORY,
    COL_CHANNEL,
    COL_RT,
    COL_CUSTOMER_TIER,
    COL_INFERRED_SEV,
    COL_MISMATCH_LABEL,
    COL_MISMATCH_TYPE,
)
from src.utils.helpers import (
    load_config,
    save_json,
    load_json,
    ensure_dir,
)
from src.utils.logger import (
    get_sia_logger,
    log_step,
    log_success,
    log_warning,
)

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  INDEX BUILDER
# ══════════════════════════════════════════════════════════════

def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "IndexFlatIP",
) -> faiss.Index:
    """
    Builds a FAISS index from normalized ticket embeddings.

    Index type: IndexFlatIP (Inner Product)
    Since embeddings are L2-normalized, inner product = cosine similarity.
    This gives exact nearest neighbor search with no approximation.

    Why not IndexIVFFlat (approximate)?
    Our dataset is 20K tickets — small enough for exact search.
    IndexFlatIP is fast, accurate, and requires no training.

    Args:
        embeddings : np.ndarray (N, dim) — L2-normalized, float32
        index_type : FAISS index type (default IndexFlatIP)

    Returns:
        Built FAISS index with all embeddings added
    """
    log_step(logger, f"Building FAISS index — type={index_type}")

    # Validate embeddings
    if embeddings.dtype != np.float32:
        logger.debug("Converting embeddings to float32")
        embeddings = embeddings.astype(np.float32)

    if len(embeddings.shape) != 2:
        raise ValueError(
            f"Embeddings must be 2D. Got shape: {embeddings.shape}"
        )

    n_vectors, dimension = embeddings.shape
    logger.info(f"Building index — vectors={n_vectors:,} | dim={dimension}")

    # Verify normalization (required for cosine similarity)
    norms = np.linalg.norm(embeddings[:100], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        log_warning(
            logger,
            "Embeddings may not be L2-normalized. "
            "Normalizing now for correct cosine similarity."
        )
        faiss.normalize_L2(embeddings)

    # Build index
    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(
            f"Unknown index type '{index_type}'. "
            f"Use 'IndexFlatIP' or 'IndexFlatL2'."
        )

    # Add all vectors
    index.add(embeddings)

    logger.info(
        f"FAISS index built — "
        f"total_vectors={index.ntotal:,} | "
        f"dimension={index.d}"
    )

    log_success(logger, "FAISS index built successfully")
    return index


# ══════════════════════════════════════════════════════════════
#  INDEX PERSISTENCE
# ══════════════════════════════════════════════════════════════

def save_faiss_index(
    index: faiss.Index,
    index_path: str,
) -> None:
    """
    Saves FAISS index to disk.

    Args:
        index      : Built FAISS index
        index_path : File path to save (.index extension)
    """
    ensure_dir(Path(index_path).parent)
    faiss.write_index(index, index_path)
    log_success(logger, f"FAISS index saved → {index_path}")


def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Loads a saved FAISS index from disk.

    Args:
        index_path : Path to saved .index file

    Returns:
        Loaded FAISS index
    """
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            f"Run build_faiss_pipeline() first."
        )

    log_step(logger, f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)

    log_success(
        logger,
        f"FAISS index loaded — "
        f"vectors={index.ntotal:,} | dim={index.d}"
    )
    return index


# ══════════════════════════════════════════════════════════════
#  METADATA BUILDER
# ══════════════════════════════════════════════════════════════

def build_ticket_metadata(df: pd.DataFrame) -> List[Dict]:
    """
    Builds lightweight metadata for each ticket.
    Stored alongside the FAISS index for retrieval results.

    Each metadata record contains:
        index           : Row index (matches FAISS result index)
        ticket_id       : Original Ticket_ID
        subject         : Cleaned ticket subject
        priority        : Assigned priority label
        inferred_severity: Inferred severity (if available)
        mismatch_type   : Mismatch type (if available)
        category        : Issue category
        channel         : Ticket channel
        tier            : Customer tier
        resolution_time : Resolution time in hours

    Args:
        df : Preprocessed (and optionally pseudo-labeled) DataFrame

    Returns:
        List of metadata dicts, one per ticket
    """
    log_step(logger, "Building ticket metadata for FAISS retrieval")

    metadata = []

    for idx, row in df.iterrows():
        record = {
            "index":            int(idx),
            "ticket_id":        str(row.get(COL_TICKET_ID,    idx)),
            "subject":          str(row.get("Ticket_Subject", "")),
            "priority":         str(row.get(COL_PRIORITY,     "")),
            "category":         str(row.get(COL_CATEGORY,     "")),
            "channel":          str(row.get(COL_CHANNEL,      "")),
            "tier":             str(row.get(COL_CUSTOMER_TIER, "")),
            "resolution_time":  float(row.get(COL_RT, 0.0)),
        }

        # Add pseudo-label info if available
        if COL_INFERRED_SEV in df.columns:
            record["inferred_severity"] = str(row.get(COL_INFERRED_SEV, ""))
        if COL_MISMATCH_TYPE in df.columns:
            record["mismatch_type"] = str(row.get(COL_MISMATCH_TYPE, ""))
        if COL_MISMATCH_LABEL in df.columns:
            record["mismatch_label"] = int(row.get(COL_MISMATCH_LABEL, 0))

        metadata.append(record)

    log_success(logger, f"Metadata built for {len(metadata):,} tickets")
    return metadata


# ══════════════════════════════════════════════════════════════
#  INDEX VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_index(
    index: faiss.Index,
    embeddings: np.ndarray,
    n_test: int = 5,
) -> bool:
    """
    Validates the FAISS index by running test queries.

    Checks:
        - Index has correct number of vectors
        - Self-query returns index 0 with similarity ~1.0
        - Results are sorted by descending similarity

    Args:
        index      : Built FAISS index
        embeddings : Original embeddings used to build index
        n_test     : Number of test queries to run

    Returns:
        True if all checks pass
    """
    log_step(logger, "Validating FAISS index")

    all_pass = True

    # Check vector count
    if index.ntotal != len(embeddings):
        log_warning(
            logger,
            f"Index has {index.ntotal} vectors but "
            f"embeddings has {len(embeddings)} rows"
        )
        all_pass = False

    # Test self-queries
    test_indices = np.random.choice(len(embeddings), n_test, replace=False)

    for test_idx in test_indices:
        query = embeddings[test_idx:test_idx+1].astype(np.float32)
        similarities, indices = index.search(query, k=3)

        # Top result should be the query itself with similarity ~1.0
        top_idx  = int(indices[0][0])
        top_sim  = float(similarities[0][0])

        if top_idx != test_idx:
            log_warning(
                logger,
                f"Self-query test failed: query={test_idx} "
                f"but top result={top_idx}"
            )
            all_pass = False

        if abs(top_sim - 1.0) > 0.01:
            log_warning(
                logger,
                f"Self-similarity too low: {top_sim:.4f} "
                f"(expected ~1.0)"
            )
            all_pass = False

        # Check results are sorted descending
        if not all(
            similarities[0][i] >= similarities[0][i+1]
            for i in range(len(similarities[0])-1)
        ):
            log_warning(logger, "FAISS results not sorted by similarity")
            all_pass = False

    if all_pass:
        log_success(logger, f"Index validation passed ({n_test} test queries)")
    else:
        log_warning(logger, "Index validation had warnings — check above")

    return all_pass


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def build_faiss_pipeline(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    config_path: str = "config/config.yaml",
) -> faiss.Index:
    """
    Full FAISS index build pipeline.
    Called from src/pipeline/train_pipeline.py and build_faiss.py.

    Steps:
        1. Build FAISS index from ticket embeddings
        2. Validate index with test queries
        3. Save index to disk
        4. Save embeddings alongside index (for reuse)
        5. Save ticket metadata for retrieval results

    Args:
        df          : Preprocessed (optionally pseudo-labeled) DataFrame
        embeddings  : np.ndarray (N, dim) ticket embeddings
        config_path : Path to config.yaml

    Returns:
        Built FAISS index
    """
    cfg = load_config(config_path)

    index_path = cfg["faiss"]["index_path"]
    emb_path   = cfg["faiss"]["embeddings_path"]
    meta_path  = cfg["faiss"]["metadata_path"]
    index_type = cfg["faiss"]["index_type"]

    logger.info("Starting FAISS index build pipeline")

    # ── Step 1: Build index ───────────────────────────────────
    index = build_faiss_index(
        embeddings = embeddings,
        index_type = index_type,
    )

    # ── Step 2: Validate ──────────────────────────────────────
    validate_index(index, embeddings)

    # ── Step 3: Save index ────────────────────────────────────
    save_faiss_index(index, index_path)

    # ── Step 4: Save embeddings alongside index ───────────────
    ensure_dir(Path(emb_path).parent)
    np.save(emb_path, embeddings)
    log_success(logger, f"Embeddings saved → {emb_path} {embeddings.shape}")

    # ── Step 5: Save metadata ─────────────────────────────────
    metadata = build_ticket_metadata(df)
    save_json(metadata, meta_path)
    log_success(logger, f"Metadata saved → {meta_path}")

    log_success(logger, "FAISS index build pipeline complete")
    return index