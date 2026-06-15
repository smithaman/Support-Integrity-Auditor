# # ─────────────────────────────────────────
# #  SIA — Support Integrity Auditor
# #  src/signals/signal1_semantic.py
# # ─────────────────────────────────────────

# from typing import Dict, Optional, Tuple

# import numpy as np
# import pandas as pd

# from config.constants import (
#     COL_COMBINED_TEXT,
#     COL_SEM_SCORE,
#     COL_CATEGORY,
#     SEVERITY_ANCHORS,
#     CATEGORY_SEVERITY_BIAS,
#     URGENCY_KEYWORDS,
# )
# from src.embeddings.sentence_encoder import (
#     load_encoder,
#     build_anchor_embeddings,
#     compute_soft_severity_batch,
# )
# from src.utils.helpers import load_config
# from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

# logger = get_sia_logger(__name__)


# # ══════════════════════════════════════════════════════════════
# #  NLP URGENCY SCORE (boosts Signal 1 spread)
# # ══════════════════════════════════════════════════════════════

# def compute_nlp_urgency_scores(texts: pd.Series) -> np.ndarray:
#     """
#     Computes rule-based urgency score (1–4) from ticket text.

#     Combines:
#         - Critical keyword density  → +0.8 per keyword (max 1.6)
#         - High keyword density      → +0.4 per keyword (max 0.8)
#         - Escalation phrases        → +0.6
#         - Negation patterns         → +0.4
#         - Fraud indicators          → +1.2 (strong boost)

#     Maps total boost to 1–4 scale:
#         boost=0    → score=1.5 (neutral Low/Medium)
#         boost=0.5  → score=2.0
#         boost=1.0  → score=2.5
#         boost=2.0  → score=3.5
#         boost=3.0+ → score=4.0
#     """
#     import re

#     critical_kws   = URGENCY_KEYWORDS.get("critical",   [])
#     high_kws       = URGENCY_KEYWORDS.get("high",       [])
#     escalation_kws = URGENCY_KEYWORDS.get("escalation", [])
#     negation_kws   = URGENCY_KEYWORDS.get("negation",   [])

#     fraud_patterns = [
#         r"\bfraud\b", r"\bunauthorized\b", r"\bhacked\b",
#         r"\bstolen\b", r"\bbreach\b", r"\bsecurity\s+incident\b",
#         r"\bdata\s+loss\b", r"\bidentity\s+theft\b",
#     ]

#     scores = []
#     for text in texts:
#         t = str(text).lower()
#         boost = 0.0

#         # Critical keywords
#         crit_count = sum(1 for kw in critical_kws if kw in t)
#         boost += min(crit_count * 0.8, 1.6)

#         # High keywords
#         high_count = sum(1 for kw in high_kws if kw in t)
#         boost += min(high_count * 0.4, 0.8)

#         # Escalation
#         esc_count = sum(1 for kw in escalation_kws if kw in t)
#         boost += min(esc_count * 0.6, 0.6)

#         # Negation
#         neg_count = sum(1 for kw in negation_kws if kw in t)
#         boost += min(neg_count * 0.4, 0.4)

#         # Fraud (strong boost)
#         fraud_count = sum(
#             1 for p in fraud_patterns
#             if re.search(p, t)
#         )
#         boost += min(fraud_count * 1.2, 1.2)

#         # Map boost → 1–4 scale
#         # boost=0 → 1.5, boost=3.0 → 4.0
#         score = 1.5 + (boost / 3.0) * 2.5
#         scores.append(float(np.clip(score, 1.0, 4.0)))

#     return np.array(scores, dtype=np.float32)


# # ══════════════════════════════════════════════════════════════
# #  CATEGORY BIAS
# # ══════════════════════════════════════════════════════════════

# def apply_category_bias(
#     semantic_scores: np.ndarray,
#     categories: pd.Series,
#     bias_weight: float = 0.2,
# ) -> np.ndarray:
#     """
#     Adjusts semantic scores using Issue_Category severity bias.
#     Fraud=4, Technical=3, Billing=2, Account/General=1
#     """
#     category_bias = np.array([
#         float(CATEGORY_SEVERITY_BIAS.get(cat, 2.0))
#         for cat in categories
#     ], dtype=np.float32)

#     adjusted = (
#         (1 - bias_weight) * semantic_scores +
#         bias_weight * category_bias
#     )

#     return np.clip(adjusted, 1.0, 4.0).astype(np.float32)


# # ══════════════════════════════════════════════════════════════
# #  SIGNAL 1 — HYBRID SEMANTIC + NLP
# # ══════════════════════════════════════════════════════════════

# def compute_signal1(
#     df: pd.DataFrame,
#     ticket_embs: np.ndarray            = None,
#     anchor_embs: Dict[int, np.ndarray] = None,
#     model                              = None,
#     use_category_bias: bool            = True,
#     category_bias_weight: float        = 0.2,
#     model_name: str                    = "BAAI/bge-small-en-v1.5",
#     config_path: str                   = "config/config.yaml",
# ) -> Tuple[np.ndarray, pd.DataFrame]:
#     """
#     Computes Signal 1 — Hybrid Semantic + NLP Severity Score.

#     Architecture:
#         Embedding similarity (BGE vs anchors) → raw_emb_score
#         NLP urgency features (keywords/fraud) → nlp_score
#         Category bias                         → bias

#         Final = 0.55 × raw_emb + 0.30 × nlp + 0.15 × category_bias

#     This hybrid approach:
#         - Widens score spread beyond embedding-only (2.38–2.57)
#         - Makes Signal 1 contribute meaningfully to fusion
#         - NLP features directly detect fraud/escalation
#         - Reduces RT dominance in fusion
#     """
#     cfg = load_config(config_path)

#     use_category_bias    = cfg["signal1"].get("use_category_bias",    use_category_bias)
#     category_bias_weight = cfg["signal1"].get("category_bias_weight", category_bias_weight)
#     model_name           = cfg["embeddings"]["model_name"]

#     logger.info("Computing Signal 1 — Hybrid Semantic + NLP Severity")

#     # ── Load encoder if needed ────────────────────────────────
#     if model is None:
#         log_step(logger, "Loading sentence encoder")
#         model = load_encoder(model_name)

#     # ── Encode tickets if needed ──────────────────────────────
#     if ticket_embs is None:
#         log_step(logger, "Encoding tickets for Signal 1")
#         from src.embeddings.sentence_encoder import encode_texts
#         ticket_embs = encode_texts(
#             texts         = df[COL_COMBINED_TEXT].fillna("").tolist(),
#             model         = model,
#             batch_size    = cfg["embeddings"]["batch_size"],
#             normalize     = True,
#             show_progress = True,
#         )

#     # ── Build anchor embeddings if needed ─────────────────────
#     if anchor_embs is None:
#         log_step(logger, "Building severity anchor embeddings")
#         anchor_embs = build_anchor_embeddings(model=model)

#     # ── Component 1: Embedding similarity (1–4) ───────────────
#     log_step(logger, "Computing embedding-anchor similarity scores")
#     emb_scores = compute_soft_severity_batch(
#         ticket_embs = ticket_embs,
#         anchor_embs = anchor_embs,
#     )

#     logger.info(
#         f"Embedding scores — "
#         f"min={emb_scores.min():.3f} | "
#         f"max={emb_scores.max():.3f} | "
#         f"mean={emb_scores.mean():.3f} | "
#         f"std={emb_scores.std():.3f}"
#     )

#     # ── Component 2: NLP urgency score (1–4) ─────────────────
#     log_step(logger, "Computing NLP urgency features")
#     nlp_scores = compute_nlp_urgency_scores(
#         df[COL_COMBINED_TEXT].fillna("")
#     )

#     logger.info(
#         f"NLP urgency scores — "
#         f"min={nlp_scores.min():.3f} | "
#         f"max={nlp_scores.max():.3f} | "
#         f"mean={nlp_scores.mean():.3f} | "
#         f"std={nlp_scores.std():.3f}"
#     )

#     # ── Component 3: Category bias (1–4) ─────────────────────
#     if use_category_bias and COL_CATEGORY in df.columns:
#         cat_bias = np.array([
#             float(CATEGORY_SEVERITY_BIAS.get(cat, 2.0))
#             for cat in df[COL_CATEGORY]
#         ], dtype=np.float32)
#     else:
#         cat_bias = np.full(len(df), 2.0, dtype=np.float32)

#     # ── Hybrid fusion ─────────────────────────────────────────
#     # Weights: embedding=0.55, nlp=0.30, category=0.15
#     W_EMB = 0.60
#     W_NLP = 0.25
#     W_CAT = 0.15

#     scores = (
#         W_EMB * emb_scores +
#         W_NLP * nlp_scores +
#         W_CAT * cat_bias
#     )
#     scores = np.clip(scores, 1.0, 4.0).astype(np.float32)

#     logger.info(
#         f"Hybrid Signal 1 scores — "
#         f"min={scores.min():.3f} | "
#         f"max={scores.max():.3f} | "
#         f"mean={scores.mean():.3f} | "
#         f"std={scores.std():.3f}"
#     )

#     # ── Add to DataFrame ──────────────────────────────────────
#     df = df.copy()
#     df[COL_SEM_SCORE] = scores

#     _log_score_distribution(scores, "Hybrid Signal 1")

#     log_success(logger, f"Signal 1 complete — {len(scores):,} scores computed")
#     return scores, df


# # ══════════════════════════════════════════════════════════════
# #  ABLATION
# # ══════════════════════════════════════════════════════════════

# def signal1_ablation(
#     df: pd.DataFrame,
#     ticket_embs: np.ndarray,
#     anchor_embs: Dict[int, np.ndarray],
# ) -> np.ndarray:
#     """Ablation — embedding only, no NLP boost."""
#     logger.info("Running Signal 1 ablation (embedding only)")
#     scores = compute_soft_severity_batch(
#         ticket_embs = ticket_embs,
#         anchor_embs = anchor_embs,
#     )
#     _log_score_distribution(scores, "Signal 1 Ablation")
#     return scores


# # ══════════════════════════════════════════════════════════════
# #  HELPERS
# # ══════════════════════════════════════════════════════════════

# def _log_score_distribution(scores: np.ndarray, label: str) -> None:
#     buckets = {
#         "Low (1.0–1.5)":      ((scores >= 1.0) & (scores < 1.5)).sum(),
#         "Medium (1.5–2.5)":   ((scores >= 1.5) & (scores < 2.5)).sum(),
#         "High (2.5–3.5)":     ((scores >= 2.5) & (scores < 3.5)).sum(),
#         "Critical (3.5–4.0)": ((scores >= 3.5) & (scores <= 4.0)).sum(),
#     }
#     logger.info(f"{label} distribution:")
#     for bucket, count in buckets.items():
#         pct = count / len(scores) * 100
#         logger.info(f"   {bucket:<22} {count:>6,}  ({pct:.1f}%)")


# def get_top_mismatch_candidates(
#     df: pd.DataFrame,
#     scores: np.ndarray,
#     top_n: int = 20,
# ) -> pd.DataFrame:
#     """Returns top-N tickets most likely to be mismatches."""
#     df = df.copy()
#     df[COL_SEM_SCORE] = scores
#     df["_sem_delta"]  = abs(scores - df["Priority_Numeric"].values)

#     top = df.nlargest(top_n, "_sem_delta")[
#         ["Ticket_ID", "Ticket_Subject", "Priority_Level",
#          COL_SEM_SCORE, "_sem_delta"]
#     ].copy()

#     top["Inferred_Label"] = top[COL_SEM_SCORE].apply(
#         lambda s: "Low"      if s < 1.5 else
#                   "Medium"   if s < 2.5 else
#                   "High"     if s < 3.5 else "Critical"
#     )

#     return top.rename(columns={"_sem_delta": "Delta"})

# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/signals/signal1_semantic.py
# ─────────────────────────────────────────

import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.constants import (
    COL_COMBINED_TEXT,
    COL_SEM_SCORE,
    COL_CATEGORY,
    SEVERITY_ANCHORS,
    CATEGORY_SEVERITY_BIAS,
    URGENCY_KEYWORDS,
)
from src.embeddings.sentence_encoder import (
    load_encoder,
    build_anchor_embeddings,
)
from src.utils.helpers import load_config
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  CONTRASTIVE EMBEDDING SCORER
# ══════════════════════════════════════════════════════════════

def compute_contrastive_scores(
    ticket_embs: np.ndarray,
    anchor_embs: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Contrastive severity scoring.

    Problem with soft weighted average:
        All anchors get similar similarity scores for support tickets
        because all ticket text lives in similar semantic space.
        Result: all scores cluster around 2.4–2.5

    Contrastive approach:
        1. Compute similarity to all 4 anchors
        2. Find top anchor (argmax) → base severity
        3. Compute gap = sim(top) - sim(second)
        4. Use gap to calibrate confidence
        5. Push scores toward extremes based on gap

    This produces:
        - Clear winners when one anchor dominates
        - Middle scores when anchors are close (genuine ambiguity)
        - Much wider spread overall
    """
    levels       = sorted(anchor_embs.keys())          # [1, 2, 3, 4]
    anchor_matrix = np.stack(
        [anchor_embs[l] for l in levels], axis=0
    ).astype(np.float32)                               # (4, dim)

    # Similarity matrix: (N, 4)
    sim_matrix = ticket_embs @ anchor_matrix.T

    scores = []
    for sims in sim_matrix:
        # Sort similarities descending
        sorted_idx  = np.argsort(sims)[::-1]
        top_idx     = sorted_idx[0]
        second_idx  = sorted_idx[1]

        top_level   = levels[top_idx]
        top_sim     = sims[top_idx]
        second_sim  = sims[second_idx]

        # Gap between top and second
        gap = float(top_sim - second_sim)

        # Base score = top anchor level
        base = float(top_level)

        # Calibrate with gap
        # Wide gap (>0.02)  → push toward top level (confident)
        # Narrow gap (<0.005) → stay in middle (uncertain)
        if gap > 0.02:
            # Confident — push toward top level
            calibrated = base + (gap * 20) * 0.3
        elif gap > 0.01:
            # Moderate confidence
            calibrated = base + (gap * 20) * 0.15
        else:
            # Low confidence — pull toward center (2.5)
            calibrated = base * 0.6 + 2.5 * 0.4

        scores.append(float(np.clip(calibrated, 1.0, 4.0)))

    return np.array(scores, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  NLP URGENCY SCORER
# ══════════════════════════════════════════════════════════════

def compute_nlp_urgency_scores(texts: pd.Series) -> np.ndarray:
    """
    Rule-based urgency score (1–4) from ticket text.

    Deliberately simple and transparent:
        No ML — pure pattern matching
        Every score traceable to specific keywords
        Used as supporting signal only (W=0.25)
    """
    critical_kws   = URGENCY_KEYWORDS.get("critical",   [])
    high_kws       = URGENCY_KEYWORDS.get("high",       [])
    escalation_kws = URGENCY_KEYWORDS.get("escalation", [])
    negation_kws   = URGENCY_KEYWORDS.get("negation",   [])

    fraud_patterns = [
        r"\bfraud\b",
        r"\bunauthorized\b",
        r"\bhacked\b",
        r"\bstolen\b",
        r"\bbreach\b",
        r"\bsecurity\s+incident\b",
        r"\bdata\s+loss\b",
        r"\bidentity\s+theft\b",
    ]

    scores = []
    for text in texts:
        t = str(text).lower()
        boost = 0.0

        # Critical keywords
        crit_count = sum(1 for kw in critical_kws if kw in t)
        boost += min(crit_count * 0.8, 1.6)

        # High keywords
        high_count = sum(1 for kw in high_kws if kw in t)
        boost += min(high_count * 0.4, 0.8)

        # Escalation
        esc_count = sum(1 for kw in escalation_kws if kw in t)
        boost += min(esc_count * 0.6, 0.6)

        # Negation
        neg_count = sum(1 for kw in negation_kws if kw in t)
        boost += min(neg_count * 0.4, 0.4)

        # Fraud (strong boost)
        fraud_count = sum(
            1 for p in fraud_patterns
            if re.search(p, t)
        )
        boost += min(fraud_count * 1.2, 1.2)

        # Map boost → 1–4
        # boost=0 → 1.5 (neutral)
        # boost=3 → 4.0 (maximum)
        score = 1.5 + (boost / 3.0) * 2.5
        scores.append(float(np.clip(score, 1.0, 4.0)))

    return np.array(scores, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  ANCHOR SIMILARITY DIAGNOSTICS
# ══════════════════════════════════════════════════════════════

def log_anchor_diagnostics(
    ticket_embs: np.ndarray,
    anchor_embs: Dict[int, np.ndarray],
    n_sample: int = 100,
) -> None:
    """
    Logs similarity gap diagnostics for a random sample of tickets.
    Helps diagnose anchor compression issues.

    Logs:
        mean similarity per anchor level
        mean gap between top and second anchor
        % of tickets where each level is top anchor
    """
    levels        = sorted(anchor_embs.keys())
    anchor_matrix = np.stack(
        [anchor_embs[l] for l in levels], axis=0
    ).astype(np.float32)

    # Sample
    idx     = np.random.choice(len(ticket_embs), min(n_sample, len(ticket_embs)), replace=False)
    sample  = ticket_embs[idx].astype(np.float32)
    sims    = sample @ anchor_matrix.T              # (sample, 4)

    # Mean similarity per level
    logger.info("Anchor similarity diagnostics (sample=100):")
    for i, level in enumerate(levels):
        logger.info(
            f"   Level {level} — "
            f"mean_sim={sims[:, i].mean():.4f} | "
            f"std={sims[:, i].std():.4f}"
        )

    # Gap analysis
    sorted_sims = np.sort(sims, axis=1)[:, ::-1]
    gaps        = sorted_sims[:, 0] - sorted_sims[:, 1]
    logger.info(
        f"Similarity gap (top vs second) — "
        f"mean={gaps.mean():.4f} | "
        f"min={gaps.min():.4f} | "
        f"max={gaps.max():.4f}"
    )

    # Top anchor distribution
    top_anchors = sims.argmax(axis=1)
    logger.info("Top anchor distribution:")
    for i, level in enumerate(levels):
        count = (top_anchors == i).sum()
        logger.info(f"   Level {level} is top: {count}/{n_sample} ({count/n_sample:.0%})")


# ══════════════════════════════════════════════════════════════
#  SIGNAL 1 — MAIN
# ══════════════════════════════════════════════════════════════

def compute_signal1(
    df: pd.DataFrame,
    ticket_embs: np.ndarray            = None,
    anchor_embs: Dict[int, np.ndarray] = None,
    model                              = None,
    use_category_bias: bool            = True,
    category_bias_weight: float        = 0.15,
    model_name: str                    = "BAAI/bge-small-en-v1.5",
    config_path: str                   = "config/config.yaml",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Signal 1 — Hybrid Contrastive Semantic + NLP Severity Score.

    Components:
        1. Contrastive embedding score  (W=0.60)
           Uses similarity gap to push scores toward extremes
        2. NLP urgency score            (W=0.25)
           Keyword + fraud + escalation features
        3. Category bias                (W=0.15)
           Domain prior: Fraud=4, Technical=3, Billing=2, Account=1

    Target spread: min~1.2, max~3.9, std>0.35
    Target mismatch rate: 15–25%
    """
    cfg = load_config(config_path)

    use_category_bias    = cfg["signal1"].get("use_category_bias",    use_category_bias)
    category_bias_weight = cfg["signal1"].get("category_bias_weight", category_bias_weight)
    model_name           = cfg["embeddings"]["model_name"]

    logger.info("Computing Signal 1 — Hybrid Contrastive Semantic + NLP")

    # ── Load encoder ──────────────────────────────────────────
    if model is None:
        log_step(logger, "Loading sentence encoder")
        model = load_encoder(model_name)

    # ── Encode tickets ────────────────────────────────────────
    if ticket_embs is None:
        log_step(logger, "Encoding tickets for Signal 1")
        from src.embeddings.sentence_encoder import encode_texts
        ticket_embs = encode_texts(
            texts         = df[COL_COMBINED_TEXT].fillna("").tolist(),
            model         = model,
            batch_size    = cfg["embeddings"]["batch_size"],
            normalize     = True,
            show_progress = True,
        )

    # ── Build anchors ─────────────────────────────────────────
    if anchor_embs is None:
        log_step(logger, "Building severity anchor embeddings")
        anchor_embs = build_anchor_embeddings(model=model)

    # ── Anchor diagnostics ────────────────────────────────────
    log_step(logger, "Running anchor similarity diagnostics")
    log_anchor_diagnostics(ticket_embs, anchor_embs, n_sample=100)

    # ── Component 1: Contrastive embedding score ──────────────
    log_step(logger, "Computing contrastive embedding scores")
    emb_scores = compute_contrastive_scores(ticket_embs, anchor_embs)

    logger.info(
        f"Contrastive embedding scores — "
        f"min={emb_scores.min():.3f} | "
        f"max={emb_scores.max():.3f} | "
        f"mean={emb_scores.mean():.3f} | "
        f"std={emb_scores.std():.3f}"
    )

    # ── Component 2: NLP urgency score ────────────────────────
    log_step(logger, "Computing NLP urgency scores")
    nlp_scores = compute_nlp_urgency_scores(
        df[COL_COMBINED_TEXT].fillna("")
    )

    logger.info(
        f"NLP urgency scores — "
        f"min={nlp_scores.min():.3f} | "
        f"max={nlp_scores.max():.3f} | "
        f"mean={nlp_scores.mean():.3f} | "
        f"std={nlp_scores.std():.3f}"
    )

    # ── Component 3: Category bias ────────────────────────────
    if use_category_bias and COL_CATEGORY in df.columns:
        cat_bias = np.array([
            float(CATEGORY_SEVERITY_BIAS.get(cat, 2.0))
            for cat in df[COL_CATEGORY]
        ], dtype=np.float32)
    else:
        cat_bias = np.full(len(df), 2.0, dtype=np.float32)

    # ── Hybrid fusion ─────────────────────────────────────────
    # Embedding remains primary signal
    # NLP provides keyword boost but doesn't dominate
    # Category adds domain prior
    W_EMB = 0.60
    W_NLP = 0.25
    W_CAT = 0.15

    scores = (
        W_EMB * emb_scores +
        W_NLP * nlp_scores +
        W_CAT * cat_bias
    )
    scores = np.clip(scores, 1.0, 4.0).astype(np.float32)

    logger.info(
        f"Final Hybrid Signal 1 — "
        f"min={scores.min():.3f} | "
        f"max={scores.max():.3f} | "
        f"mean={scores.mean():.3f} | "
        f"std={scores.std():.3f}"
    )

    # ── Add to DataFrame ──────────────────────────────────────
    df = df.copy()
    df[COL_SEM_SCORE] = scores

    _log_score_distribution(scores, "Hybrid Signal 1")

    log_success(logger, f"Signal 1 complete — {len(scores):,} scores")
    return scores, df


# ══════════════════════════════════════════════════════════════
#  ABLATION
# ══════════════════════════════════════════════════════════════

def signal1_ablation(
    df: pd.DataFrame,
    ticket_embs: np.ndarray,
    anchor_embs: Dict[int, np.ndarray],
) -> np.ndarray:
    """Ablation — contrastive embedding only, no NLP."""
    logger.info("Running Signal 1 ablation (contrastive embedding only)")
    scores = compute_contrastive_scores(ticket_embs, anchor_embs)
    _log_score_distribution(scores, "Signal 1 Ablation (embedding only)")
    return scores


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _log_score_distribution(scores: np.ndarray, label: str) -> None:
    buckets = {
        "Low (1.0–1.5)":      ((scores >= 1.0) & (scores < 1.5)).sum(),
        "Medium (1.5–2.5)":   ((scores >= 1.5) & (scores < 2.5)).sum(),
        "High (2.5–3.5)":     ((scores >= 2.5) & (scores < 3.5)).sum(),
        "Critical (3.5–4.0)": ((scores >= 3.5) & (scores <= 4.0)).sum(),
    }
    logger.info(f"{label} distribution:")
    for bucket, count in buckets.items():
        pct = count / len(scores) * 100
        logger.info(f"   {bucket:<22} {count:>6,}  ({pct:.1f}%)")


def get_top_mismatch_candidates(
    df: pd.DataFrame,
    scores: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Returns top-N tickets most likely to be mismatches."""
    df = df.copy()
    df[COL_SEM_SCORE] = scores
    df["_sem_delta"]  = abs(scores - df["Priority_Numeric"].values)
    top = df.nlargest(top_n, "_sem_delta")[
        ["Ticket_ID", "Ticket_Subject", "Priority_Level",
         COL_SEM_SCORE, "_sem_delta"]
    ].copy()
    top["Inferred_Label"] = top[COL_SEM_SCORE].apply(
        lambda s: "Low"    if s < 1.5 else
                  "Medium" if s < 2.5 else
                  "High"   if s < 3.5 else "Critical"
    )
    return top.rename(columns={"_sem_delta": "Delta"})