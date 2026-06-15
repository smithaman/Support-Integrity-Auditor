# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/embeddings/generate_embeddings.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.constants import (
    COL_TICKET_ID,
    COL_COMBINED_TEXT,
)
from src.embeddings.sentence_encoder import (
    load_encoder,
    encode_texts,
    build_anchor_embeddings,
)
from src.utils.helpers import (
    load_config,
    save_numpy,
    load_numpy,
    save_json,
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
#  TICKET EMBEDDINGS
# ══════════════════════════════════════════════════════════════

def generate_ticket_embeddings(
    df: pd.DataFrame,
    model_name: str  = "BAAI/bge-small-en-v1.5",
    batch_size: int  = 64,
    save_path: str   = None,
    cache_path: str  = None,
    config_path: str = "config/config.yaml",
) -> Tuple[np.ndarray, object]:
    """
    Generates and saves embeddings for all tickets.

    Checks cache first — if embeddings already exist at
    cache_path and match the dataset size, loads from cache
    instead of re-encoding (saves time on reruns).

    Args:
        df          : Preprocessed DataFrame with combined_text column
        model_name  : SentenceTransformer model to use
        batch_size  : Encoding batch size
        save_path   : Where to save embeddings (.npy)
        cache_path  : Cache path to check before re-encoding
        config_path : Path to config.yaml

    Returns:
        (embeddings, model)
        embeddings : np.ndarray shape (N, dim), float32, normalized
        model      : Loaded SentenceTransformer (reused downstream)
    """
    cfg = load_config(config_path)

    model_name = model_name or cfg["embeddings"]["model_name"]
    batch_size = batch_size or cfg["embeddings"]["batch_size"]
    save_path  = save_path  or cfg["embeddings"]["save_path"]
    cache_path = cache_path or cfg["embeddings"]["cache_path"]

    n_tickets = len(df)

    # ── Check cache ───────────────────────────────────────────
    if cache_path and Path(cache_path).exists():
        log_step(logger, f"Cache found at {cache_path} — checking validity")
        try:
            cached = load_numpy(cache_path)
            if cached.shape[0] == n_tickets:
                log_success(logger, f"Cache valid ({cached.shape}) — skipping re-encoding")
                model = load_encoder(model_name)
                return cached, model
            else:
                log_warning(
                    logger,
                    f"Cache shape {cached.shape} doesn't match dataset size "
                    f"{n_tickets} — re-encoding"
                )
        except Exception as e:
            log_warning(logger, f"Cache load failed ({e}) — re-encoding")

    # ── Extract texts ─────────────────────────────────────────
    log_step(logger, "Extracting combined texts for encoding")

    if COL_COMBINED_TEXT not in df.columns:
        raise ValueError(
            f"Column '{COL_COMBINED_TEXT}' not found. "
            f"Run preprocess_pipeline() first."
        )

    texts = df[COL_COMBINED_TEXT].fillna("").tolist()
    logger.info(f"Encoding {n_tickets:,} tickets")

    # ── Load model ────────────────────────────────────────────
    model = load_encoder(model_name)

    # ── Encode ────────────────────────────────────────────────
    log_step(logger, "Encoding tickets")
    embeddings = encode_texts(
        texts=texts,
        model=model,
        batch_size=batch_size,
        normalize=True,
        show_progress=True,
    )

    # ── Save embeddings ───────────────────────────────────────
    if save_path:
        save_numpy(embeddings, save_path)
        log_success(logger, f"Embeddings saved → {save_path} {embeddings.shape}")

    # ── Save to cache ─────────────────────────────────────────
    if cache_path and cache_path != save_path:
        save_numpy(embeddings, cache_path)
        logger.debug(f"Cache updated → {cache_path}")

    return embeddings, model


# ══════════════════════════════════════════════════════════════
#  ANCHOR EMBEDDINGS
# ══════════════════════════════════════════════════════════════

def generate_anchor_embeddings(
    model,
    save_path: str = "outputs/embeddings/anchor_embeddings.npy",
    anchors: Dict  = None,
) -> Dict[int, np.ndarray]:
    """
    Generates and saves severity anchor embeddings.

    Anchors are small (4 vectors) — fast to generate.
    Saved separately from ticket embeddings.

    Args:
        model     : Loaded SentenceTransformer model
        save_path : Path to save anchor embeddings
        anchors   : Custom anchor dict (defaults to SEVERITY_ANCHORS)

    Returns:
        Dict mapping severity level (1–4) → embedding vector
    """
    log_step(logger, "Generating anchor embeddings")

    anchor_embs = build_anchor_embeddings(model=model, anchors=anchors)

    # Save as stacked numpy array with level metadata
    if save_path:
        levels    = sorted(anchor_embs.keys())
        stack     = np.stack([anchor_embs[l] for l in levels])
        save_numpy(stack, save_path)

        # Save level mapping alongside
        meta_path = save_path.replace(".npy", "_meta.json")
        save_json({"levels": levels}, meta_path)

        log_success(logger, f"Anchor embeddings saved → {save_path} {stack.shape}")

    return anchor_embs


# ══════════════════════════════════════════════════════════════
#  TICKET METADATA (for FAISS retrieval)
# ══════════════════════════════════════════════════════════════

def save_ticket_metadata(
    df: pd.DataFrame,
    save_path: str = "outputs/faiss_index/ticket_metadata.json",
) -> None:
    """
    Saves lightweight ticket metadata alongside FAISS index.
    Used during retrieval to return human-readable results.

    Stores per ticket:
        index        : row index (matches FAISS result index)
        ticket_id    : original Ticket_ID
        subject      : cleaned subject
        priority     : assigned priority label
        category     : issue category
        channel      : ticket channel
        tier         : customer tier
        resolution_time : resolution time in hours

    Args:
        df        : Preprocessed DataFrame
        save_path : Path to save metadata JSON
    """
    log_step(logger, "Saving ticket metadata for FAISS retrieval")

    metadata = []
    for idx, row in df.iterrows():
        metadata.append({
            "index":           int(idx),
            "ticket_id":       str(row.get(COL_TICKET_ID, idx)),
            "subject":         str(row.get("Ticket_Subject", "")),
            "priority":        str(row.get("Priority_Level", "")),
            "category":        str(row.get("Issue_Category", "")),
            "channel":         str(row.get("Ticket_Channel", "")),
            "tier":            str(row.get("Customer_Tier", "")),
            "resolution_time": float(row.get("Resolution_Time_Hours", 0)),
        })

    save_json(metadata, save_path)
    log_success(logger, f"Metadata saved → {save_path} ({len(metadata):,} records)")


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════

def embedding_pipeline(
    df: pd.DataFrame,
    config_path: str = "config/config.yaml",
) -> Tuple[np.ndarray, Dict[int, np.ndarray], object]:
    """
    Full embedding pipeline.
    Runs after feature_engineering_pipeline().

    Steps:
        1. Generate ticket embeddings (with cache check)
        2. Generate anchor embeddings
        3. Save ticket metadata for FAISS

    Args:
        df          : Feature-engineered DataFrame
        config_path : Path to config.yaml

    Returns:
        (ticket_embeddings, anchor_embeddings, model)
        ticket_embeddings : np.ndarray (N, dim)
        anchor_embeddings : Dict[int, np.ndarray]
        model             : Loaded SentenceTransformer
    """
    logger.info("Starting embedding pipeline")

    cfg = load_config(config_path)

    # Step 1 — Ticket embeddings
    ticket_embs, model = generate_ticket_embeddings(
        df=df,
        model_name=cfg["embeddings"]["model_name"],
        batch_size=cfg["embeddings"]["batch_size"],
        save_path=cfg["embeddings"]["save_path"],
        cache_path=cfg["embeddings"]["cache_path"],
        config_path=config_path,
    )

    # Step 2 — Anchor embeddings
    anchor_embs = generate_anchor_embeddings(
        model=model,
        save_path="outputs/embeddings/anchor_embeddings.npy",
    )

    # Step 3 — Ticket metadata for FAISS
    save_ticket_metadata(
        df=df,
        save_path=cfg["faiss"]["metadata_path"],
    )

    log_success(logger, "Embedding pipeline complete")
    logger.info(
        f"Ticket embeddings : {ticket_embs.shape} | "
        f"Anchors : {len(anchor_embs)} vectors"
    )

    return ticket_embs, anchor_embs, model