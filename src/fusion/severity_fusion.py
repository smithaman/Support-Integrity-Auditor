# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/fusion/severity_fusion.py
# ─────────────────────────────────────────

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.constants import (
    COL_SEM_SCORE,
    COL_RT_SCORE,
    COL_FUSED_SCORE,
    COL_INFERRED_SEV,
    COL_INFERRED_NUM,
    WEIGHT_SEMANTIC,
    WEIGHT_RESOLUTION,
    MISMATCH_THRESHOLD,
    PRIORITY_MAP,
    PRIORITY_INV,
)
from src.utils.helpers import load_config, score_to_label, check_mismatch_rate
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  CORE FUSION
# ══════════════════════════════════════════════════════════════

def weighted_fusion(
    semantic_scores: np.ndarray,
    rt_scores: np.ndarray,
    w_semantic: float    = WEIGHT_SEMANTIC,
    w_resolution: float  = WEIGHT_RESOLUTION,
) -> np.ndarray:
    """
    Fuses Signal 1 and Signal 2 via weighted average.

        Fused = w_semantic × Semantic + w_resolution × RT

    Default weights: 0.7 × Semantic + 0.3 × Resolution

    Weight rationale:
        - Semantic (BGE embeddings) directly reads ticket content
          and is the strongest signal → 70%
        - Resolution time is an indirect, noisy proxy → 30%
        - Weights sum to 1.0 (verified below)

    Args:
        semantic_scores : np.ndarray (N,) Signal 1 scores (1–4)
        rt_scores       : np.ndarray (N,) Signal 2 scores (1–4)
        w_semantic      : Weight for semantic signal (default 0.7)
        w_resolution    : Weight for resolution signal (default 0.3)

    Returns:
        np.ndarray (N,) fused scores (1.0–4.0), float32
    """
    # Validate weights
    weight_sum = w_semantic + w_resolution
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Fusion weights must sum to 1.0. "
            f"Got {w_semantic} + {w_resolution} = {weight_sum:.4f}"
        )

    # Validate shape
    if semantic_scores.shape != rt_scores.shape:
        raise ValueError(
            f"Score arrays must have same shape. "
            f"Got semantic={semantic_scores.shape}, rt={rt_scores.shape}"
        )

    fused = w_semantic * semantic_scores + w_resolution * rt_scores
    return np.clip(fused, 1.0, 4.0).astype(np.float32)


def scores_to_labels(fused_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts continuous fused scores to:
        1. Integer severity levels (1–4)
        2. String priority labels (Low/Medium/High/Critical)

    Boundary mapping:
        score < 1.5  → Low      (1)
        score < 2.5  → Medium   (2)
        score < 3.5  → High     (3)
        score >= 3.5 → Critical (4)

    Args:
        fused_scores : np.ndarray (N,) continuous scores (1–4)

    Returns:
        (numeric_levels, label_array)
        numeric_levels : np.ndarray (N,) int, values 1–4
        label_array    : np.ndarray (N,) str labels
    """
    numeric = np.round(np.clip(fused_scores, 1.0, 4.0)).astype(int)
    labels  = np.array([PRIORITY_INV[n] for n in numeric])
    return numeric, labels


# ══════════════════════════════════════════════════════════════
#  SIGNAL AGREEMENT
# ══════════════════════════════════════════════════════════════

def compute_signal_agreement(
    semantic_scores: np.ndarray,
    rt_scores: np.ndarray,
) -> Dict:
    """
    Computes pairwise agreement between Signal 1 and Signal 2.

    Agreement is measured at the label level (after rounding to 1–4).
    This is the "Pseudo-Label Signal Agreement" metric required
    by the evaluation criteria.

    Also computes Cohen's Kappa for a more robust agreement measure.

    Args:
        semantic_scores : np.ndarray (N,) Signal 1 scores
        rt_scores       : np.ndarray (N,) Signal 2 scores

    Returns:
        Dict with agreement metrics:
        {
            "exact_agreement":   0.62,   # fraction with same rounded label
            "within_1_level":    0.89,   # fraction within 1 severity level
            "cohen_kappa":       0.48,   # Cohen's kappa
            "mean_abs_diff":     0.71,   # mean |S1 - S2|
        }
    """
    s1_rounded = np.round(np.clip(semantic_scores, 1.0, 4.0)).astype(int)
    s2_rounded = np.round(np.clip(rt_scores,       1.0, 4.0)).astype(int)

    # Exact agreement
    exact = (s1_rounded == s2_rounded).mean()

    # Within 1 level
    within_1 = (np.abs(s1_rounded - s2_rounded) <= 1).mean()

    # Mean absolute difference (continuous)
    mean_abs_diff = np.abs(semantic_scores - rt_scores).mean()

    # Cohen's Kappa
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(s1_rounded, s2_rounded)
    except Exception:
        kappa = None

    agreement = {
        "exact_agreement": round(float(exact),        4),
        "within_1_level":  round(float(within_1),     4),
        "cohen_kappa":     round(float(kappa), 4) if kappa is not None else None,
        "mean_abs_diff":   round(float(mean_abs_diff), 4),
    }

    logger.info("Signal Agreement:")
    for k, v in agreement.items():
        logger.info(f"   {k:<22} {v}")

    return agreement


# ══════════════════════════════════════════════════════════════
#  ABLATION EXPERIMENTS
# ══════════════════════════════════════════════════════════════

def run_ablation(
    semantic_scores: np.ndarray,
    rt_scores: np.ndarray,
    assigned_numeric: np.ndarray,
    threshold: float = MISMATCH_THRESHOLD,
) -> Dict:
    """
    Runs all fusion configurations for the ablation table.

    Configurations tested:
        1. Signal 1 only  (Semantic)
        2. Signal 2 only  (RT)
        3. Equal fusion   (0.5 × S1 + 0.5 × S2)
        4. Final fusion   (0.7 × S1 + 0.3 × S2)  ← our config

    For each config, computes:
        - Mismatch rate (% of tickets flagged as mismatch)
        - Signal-assignment agreement (% where inferred == assigned)

    Args:
        semantic_scores  : Signal 1 scores
        rt_scores        : Signal 2 scores
        assigned_numeric : Assigned priority as numeric (1–4)
        threshold        : Mismatch threshold (default 1.5)

    Returns:
        Dict mapping config_name → metrics dict
    """
    logger.info("Running ablation experiments")

    configs = {
        "Signal 1 only (Semantic)":       (1.0,  0.0),
        "Signal 2 only (RT)":             (0.0,  1.0),
        "Equal fusion (0.5 + 0.5)":       (0.5,  0.5),
        "Final fusion (0.7 + 0.3)":       (0.7,  0.3),
    }

    results = {}

    for name, (w1, w2) in configs.items():
        fused   = w1 * semantic_scores + w2 * rt_scores
        fused   = np.clip(fused, 1.0, 4.0)
        rounded = np.round(fused).astype(int)

        delta        = np.abs(fused - assigned_numeric)
        mismatch     = (delta >= threshold)
        mismatch_rate = mismatch.mean()

        agreement = (rounded == assigned_numeric).mean()

        results[name] = {
            "mismatch_rate":       round(float(mismatch_rate), 4),
            "assign_agreement":    round(float(agreement),     4),
            "mean_fused_score":    round(float(fused.mean()),  4),
        }

        logger.info(
            f"   {name:<35} "
            f"mismatch={mismatch_rate:.1%} | "
            f"agreement={agreement:.1%}"
        )

    return results


# ══════════════════════════════════════════════════════════════
#  MAIN FUSION PIPELINE
# ══════════════════════════════════════════════════════════════

def fusion_pipeline(
    df: pd.DataFrame,
    semantic_scores: np.ndarray  = None,
    rt_scores: np.ndarray        = None,
    config_path: str             = "config/config.yaml",
) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """
    Full fusion pipeline.
    Runs after Signal 1 and Signal 2 have been computed.

    Steps:
        1. Load scores from DataFrame if not provided
        2. Validate both signal arrays
        3. Compute signal agreement (for ablation table)
        4. Fuse signals → continuous fused score
        5. Convert to integer levels and string labels
        6. Add all columns to DataFrame
        7. Check mismatch rate is in acceptable range

    Args:
        df              : DataFrame with Severity_Semantic + Severity_RT
        semantic_scores : Signal 1 scores (if not in df)
        rt_scores       : Signal 2 scores (if not in df)
        config_path     : Path to config.yaml

    Returns:
        (fused_scores, df, agreement_metrics)
        fused_scores      : np.ndarray (N,) fused scores (1–4)
        df                : DataFrame with fusion columns added
        agreement_metrics : Dict with signal agreement stats
    """
    cfg = load_config(config_path)

    w_sem = cfg["fusion"]["semantic_weight"]
    w_rt  = cfg["fusion"]["resolution_weight"]

    logger.info(
        f"Starting fusion pipeline — "
        f"weights: {w_sem} × Semantic + {w_rt} × RT"
    )

    # ── Load scores from DataFrame if not provided ────────────
    if semantic_scores is None:
        if COL_SEM_SCORE not in df.columns:
            raise ValueError(
                f"Column '{COL_SEM_SCORE}' not found. "
                f"Run compute_signal1() first."
            )
        semantic_scores = df[COL_SEM_SCORE].values.astype(np.float32)

    if rt_scores is None:
        if COL_RT_SCORE not in df.columns:
            raise ValueError(
                f"Column '{COL_RT_SCORE}' not found. "
                f"Run compute_signal2() first."
            )
        rt_scores = df[COL_RT_SCORE].values.astype(np.float32)

    # ── Step 1: Signal agreement ──────────────────────────────
    log_step(logger, "Computing signal agreement")
    agreement = compute_signal_agreement(semantic_scores, rt_scores)

    # ── Step 2: Fuse signals ──────────────────────────────────
    log_step(logger, f"Fusing signals ({w_sem} × Semantic + {w_rt} × RT)")
    fused_scores = weighted_fusion(
        semantic_scores = semantic_scores,
        rt_scores       = rt_scores,
        w_semantic      = w_sem,
        w_resolution    = w_rt,
    )

    logger.info(
        f"Fused scores — "
        f"min={fused_scores.min():.2f} | "
        f"max={fused_scores.max():.2f} | "
        f"mean={fused_scores.mean():.2f}"
    )

    # ── Step 3: Convert to labels ─────────────────────────────
    log_step(logger, "Converting scores to severity labels")
    inferred_numeric, inferred_labels = scores_to_labels(fused_scores)

    # ── Step 4: Add columns to DataFrame ─────────────────────
    df = df.copy()
    df[COL_FUSED_SCORE]   = fused_scores
    df[COL_INFERRED_NUM]  = inferred_numeric
    df[COL_INFERRED_SEV]  = inferred_labels

    # ── Step 5: Log inferred severity distribution ────────────
    logger.info("Inferred Severity Distribution:")
    for label in ["Low", "Medium", "High", "Critical"]:
        count = (df[COL_INFERRED_SEV] == label).sum()
        pct   = count / len(df) * 100
        logger.info(f"   {label:<10} {count:>6,}  ({pct:.1f}%)")

    # ── Step 6: Check mismatch rate preview ───────────────────
    if "Priority_Numeric" in df.columns:
        log_step(logger, "Previewing mismatch rate at threshold")
        delta        = np.abs(fused_scores - df["Priority_Numeric"].values)
        preview_rate = (delta >= MISMATCH_THRESHOLD).mean()
        logger.info(
            f"Preview mismatch rate at threshold={MISMATCH_THRESHOLD}: "
            f"{preview_rate:.1%}"
        )

        min_rate = cfg["fusion"]["target_mismatch_rate"]["min"]
        max_rate = cfg["fusion"]["target_mismatch_rate"]["max"]
        if not (min_rate <= preview_rate <= max_rate):
            log_warning(
                logger,
                f"Mismatch rate {preview_rate:.1%} outside target "
                f"{min_rate:.0%}–{max_rate:.0%}. "
                f"Consider adjusting fusion threshold in config.yaml"
            )

    log_success(logger, "Fusion pipeline complete")
    return fused_scores, df, agreement