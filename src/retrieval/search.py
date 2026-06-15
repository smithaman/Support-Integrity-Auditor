"""
search.py
FAISS Semantic Search
Given a query ticket embedding, retrieves top-K most similar
historical tickets with similarity scores.
Used to provide comparative evidence in dossiers.
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/retrieval/search.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from config.constants import (
    COL_TICKET_ID,
    COL_PRIORITY,
    COL_CATEGORY,
    COL_CHANNEL,
    COL_RT,
    COL_CUSTOMER_TIER,
    COL_INFERRED_SEV,
    COL_MISMATCH_TYPE,
    MISMATCH_HIDDEN_CRISIS,
    MISMATCH_FALSE_ALARM,
)
from src.retrieval.build_index import load_faiss_index
from src.utils.helpers import load_config, load_json
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  SEARCHER CLASS
# ══════════════════════════════════════════════════════════════

class FAISSSearcher:
    """
    Manages FAISS index loading and semantic search operations.

    Loads the index and metadata once, then serves multiple
    search queries efficiently.

    Usage:
        searcher = FAISSSearcher()
        searcher.load()
        results = searcher.search(query_embedding, k=5)
    """

    def __init__(
        self,
        index_path: str    = None,
        metadata_path: str = None,
        config_path: str   = "config/config.yaml",
    ):
        self.config_path   = config_path
        cfg                = load_config(config_path)
        self.index_path    = index_path    or cfg["faiss"]["index_path"]
        self.metadata_path = metadata_path or cfg["faiss"]["metadata_path"]
        self.top_k         = cfg["faiss"]["top_k"]

        self.index    = None
        self.metadata = None
        self._loaded  = False

    def load(self) -> None:
        """
        Loads FAISS index and ticket metadata from disk.
        Must be called before search().
        """
        log_step(logger, "Loading FAISS searcher")

        self.index    = load_faiss_index(self.index_path)
        self.metadata = load_json(self.metadata_path)

        self._loaded = True
        log_success(
            logger,
            f"FAISSSearcher ready — "
            f"{self.index.ntotal:,} vectors | "
            f"{len(self.metadata):,} metadata records"
        )

    def _check_loaded(self) -> None:
        """Raises error if searcher not loaded."""
        if not self._loaded:
            raise RuntimeError(
                "FAISSSearcher not loaded. Call searcher.load() first."
            )

    # ── Core Search ───────────────────────────────────────────

    def search(
        self,
        query_emb: np.ndarray,
        k: int        = None,
        exclude_idx: int = None,
    ) -> List[Dict]:
        """
        Finds k most similar tickets to a query embedding.

        Args:
            query_emb   : np.ndarray (dim,) — normalized query embedding
            k           : Number of results (defaults to config top_k)
            exclude_idx : Row index to exclude (avoids self-match)

        Returns:
            List of result dicts sorted by similarity descending:
            [
                {
                    "rank":          1,
                    "index":         142,
                    "ticket_id":     "TKT-0142",
                    "subject":       "...",
                    "priority":      "High",
                    "category":      "Technical",
                    "channel":       "Email",
                    "tier":          "enterprise",
                    "resolution_time": 8.5,
                    "similarity":    0.923,
                    "inferred_severity": "Critical",  # if available
                    "mismatch_type": "Hidden Crisis",  # if available
                },
                ...
            ]
        """
        self._check_loaded()

        if k is None:
            k = self.top_k

        # Fetch extra results to allow excluding self-match
        fetch_k = k + (1 if exclude_idx is not None else 0)

        query = query_emb.astype(np.float32).reshape(1, -1)
        similarities, indices = self.index.search(query, fetch_k)

        results = []
        rank    = 1

        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue   # FAISS padding for short results

            if exclude_idx is not None and int(idx) == exclude_idx:
                continue   # Skip self-match

            if rank > k:
                break

            # Get metadata for this index
            meta = self._get_metadata(int(idx))

            result = {
                "rank":            rank,
                "index":           int(idx),
                "ticket_id":       meta.get("ticket_id",      str(idx)),
                "subject":         meta.get("subject",         ""),
                "priority":        meta.get("priority",        ""),
                "category":        meta.get("category",        ""),
                "channel":         meta.get("channel",         ""),
                "tier":            meta.get("tier",            ""),
                "resolution_time": meta.get("resolution_time", 0.0),
                "similarity":      round(float(sim),           4),
            }

            # Add optional fields if present
            if "inferred_severity" in meta:
                result["inferred_severity"] = meta["inferred_severity"]
            if "mismatch_type" in meta:
                result["mismatch_type"] = meta["mismatch_type"]
            if "mismatch_label" in meta:
                result["mismatch_label"] = meta["mismatch_label"]

            results.append(result)
            rank += 1

        return results

    def search_by_text(
        self,
        text: str,
        model,
        k: int = None,
    ) -> List[Dict]:
        """
        Searches by raw text string.
        Encodes the text first, then runs search().

        Used by Streamlit app for single-ticket analysis.

        Args:
            text  : Raw ticket text to search
            model : Loaded SentenceTransformer model
            k     : Number of results

        Returns:
            List of result dicts (same format as search())
        """
        from src.embeddings.sentence_encoder import encode_single

        self._check_loaded()

        query_emb = encode_single(text=text, model=model, normalize=True)
        return self.search(query_emb=query_emb, k=k)

    # ── Pattern Analysis ──────────────────────────────────────

    def analyze_similar_priorities(
        self,
        results: List[Dict],
        inferred_severity: str,
    ) -> Dict:
        """
        Analyzes priority patterns in similar ticket results.
        Used to generate FAISS evidence for the dossier.

        Checks:
            - What priority do most similar tickets have?
            - Does the dominant priority support the mismatch finding?
            - How many similar tickets were also mismatches?

        Args:
            results            : List from search()
            inferred_severity  : Inferred severity label for this ticket

        Returns:
            Dict with pattern analysis:
            {
                "dominant_priority":      "High",
                "dominant_count":         4,
                "total_similar":          5,
                "dominant_pct":           0.80,
                "supports_mismatch":      True,
                "n_similar_mismatches":   3,
                "pattern_description":    "4 of 5 similar tickets..."
            }
        """
        if not results:
            return {
                "dominant_priority":    "Unknown",
                "dominant_count":       0,
                "total_similar":        0,
                "dominant_pct":         0.0,
                "supports_mismatch":    False,
                "n_similar_mismatches": 0,
                "pattern_description":  "No similar tickets found",
            }

        priorities = [r["priority"] for r in results]
        priority_counts: Dict[str, int] = {}
        for p in priorities:
            priority_counts[p] = priority_counts.get(p, 0) + 1

        dominant       = max(priority_counts, key=priority_counts.get)
        dominant_count = priority_counts[dominant]
        total          = len(results)
        dominant_pct   = dominant_count / total

        # Check if dominant priority supports mismatch
        # (most similar tickets have different priority than assigned)
        supports = (dominant == inferred_severity)

        # Count similar tickets that were also mismatches
        n_mismatches = sum(
            1 for r in results
            if r.get("mismatch_label", 0) == 1
        )

        description = (
            f"{dominant_count} of {total} semantically similar tickets "
            f"were assigned {dominant} priority"
        )

        return {
            "dominant_priority":      dominant,
            "dominant_count":         dominant_count,
            "total_similar":          total,
            "dominant_pct":           round(dominant_pct, 3),
            "supports_mismatch":      supports,
            "n_similar_mismatches":   n_mismatches,
            "pattern_description":    description,
        }

    # ── Metadata Helper ───────────────────────────────────────

    def _get_metadata(self, idx: int) -> Dict:
        """
        Returns metadata record for a given FAISS index position.
        Falls back to empty dict if index out of range.
        """
        if self.metadata and idx < len(self.metadata):
            return self.metadata[idx]
        return {}


# ══════════════════════════════════════════════════════════════
#  STANDALONE SEARCH FUNCTIONS
# ══════════════════════════════════════════════════════════════

def search_similar_tickets(
    query_emb: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict],
    k: int         = 5,
    exclude_idx: int = None,
) -> List[Dict]:
    """
    Standalone search function (no FAISSSearcher class needed).
    Used when index and metadata are already loaded in memory.

    Args:
        query_emb   : Query embedding (normalized)
        index       : Loaded FAISS index
        metadata    : List of metadata dicts
        k           : Number of results
        exclude_idx : Index to exclude (self-match)

    Returns:
        List of result dicts
    """
    fetch_k = k + (1 if exclude_idx is not None else 0)
    query   = query_emb.astype(np.float32).reshape(1, -1)

    similarities, indices = index.search(query, fetch_k)

    results = []
    rank    = 1

    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        if exclude_idx is not None and int(idx) == exclude_idx:
            continue
        if rank > k:
            break

        meta = metadata[int(idx)] if int(idx) < len(metadata) else {}

        results.append({
            "rank":            rank,
            "index":           int(idx),
            "ticket_id":       meta.get("ticket_id",       str(idx)),
            "subject":         meta.get("subject",          ""),
            "priority":        meta.get("priority",         ""),
            "category":        meta.get("category",         ""),
            "channel":         meta.get("channel",          ""),
            "tier":            meta.get("tier",             ""),
            "resolution_time": meta.get("resolution_time",  0.0),
            "similarity":      round(float(sim),            4),
            "inferred_severity": meta.get("inferred_severity", ""),
            "mismatch_type":   meta.get("mismatch_type",    ""),
            "mismatch_label":  meta.get("mismatch_label",   0),
        })
        rank += 1

    return results


def load_searcher(config_path: str = "config/config.yaml") -> FAISSSearcher:
    """
    Convenience function — loads and returns a ready FAISSSearcher.

    Usage:
        searcher = load_searcher()
        results  = searcher.search(query_emb, k=5)
    """
    searcher = FAISSSearcher(config_path=config_path)
    searcher.load()
    return searcher