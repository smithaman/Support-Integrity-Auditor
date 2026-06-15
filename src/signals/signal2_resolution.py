# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/signals/signal2_resolution.py
# ─────────────────────────────────────────

from typing import Tuple

import numpy as np
import pandas as pd

from config.constants import (
    COL_RT,
    COL_RT_SCORE,
    COL_PRIORITY,
    EXPECTED_RT,
    RT_PERCENTILES,
)
from src.utils.helpers import load_config
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  CORE SCORING METHODS
# ══════════════════════════════════════════════════════════════

def score_by_continuous_percentile(
    rt_values: np.ndarray,
    clip_lower: float = 0.05,
    clip_upper: float = 0.95,
) -> np.ndarray:
    """
    Converts resolution times to continuous severity scores (1–4)
    using percentile ranking.

    Why continuous instead of hard bins?
    Hard bins (0–25%→1, 25–50%→2 ...) lose boundary information.
    A ticket at the 24th percentile (score=1) and 26th percentile
    (score=2) would be treated very differently despite being
    nearly identical. Continuous scaling preserves this nuance.

    Steps:
        1. Clip outliers at clip_lower/clip_upper percentiles
        2. Compute percentile rank for each value (0–1)
        3. Scale to 1–4 range linearly

    Higher RT → higher percentile → higher severity score.
    This reflects the dataset insight:
        Critical tickets resolve in ~12hrs (fast)
        Low tickets resolve in ~45hrs (slow)

    Args:
        rt_values  : np.ndarray of resolution times (hours)
        clip_lower : Lower percentile to clip (removes outliers)
        clip_upper : Upper percentile to clip (removes outliers)

    Returns:
        np.ndarray (N,) of scores in range [1.0, 4.0], float32
    """
    rt = rt_values.astype(np.float64)

    # Clip outliers
    lower_bound = np.percentile(rt, clip_lower * 100)
    upper_bound = np.percentile(rt, clip_upper * 100)
    rt_clipped  = np.clip(rt, lower_bound, upper_bound)

    logger.debug(
        f"RT clipping — lower: {lower_bound:.1f}hrs | "
        f"upper: {upper_bound:.1f}hrs"
    )

    # Compute percentile rank (0–1) for each value
    # rank(pct=True) gives fraction of values <= current value
    rt_series   = pd.Series(rt_clipped)
    percentiles = rt_series.rank(pct=True).values   # 0.0 to 1.0

    # Scale to 1–4
    scores = 1.0 + (percentiles * 3.0)              # 1.0 to 4.0

    return np.clip(scores, 1.0, 4.0).astype(np.float32)


def score_by_priority_comparison(
    rt_values: np.ndarray,
    priorities: pd.Series,
) -> np.ndarray:
    """
    Alternative scoring: compares each ticket's RT against
    the expected RT window for its assigned priority.

    Logic:
        If RT >> expected for assigned priority → under-triaged → high score
        If RT << expected for assigned priority → over-triaged  → low score
        If RT within expected range             → consistent    → neutral score

    Expected RT windows (from dataset analysis):
        Critical : 0–4 hrs
        High     : 2–8 hrs
        Medium   : 8–24 hrs
        Low      : 24–72 hrs

    This is a complementary perspective to percentile scoring.
    Not used in main fusion but available for ablation.

    Args:
        rt_values  : np.ndarray of resolution times
        priorities : pd.Series of assigned priority labels

    Returns:
        np.ndarray (N,) scores in range [1.0, 4.0]
    """
    scores = np.zeros(len(rt_values), dtype=np.float32)

    for i, (rt, priority) in enumerate(zip(rt_values, priorities)):
        expected_low, expected_high = EXPECTED_RT.get(priority, (8, 24))

        if rt > expected_high * 2:
            # Far above expected — severe under-triaging
            scores[i] = 4.0
        elif rt > expected_high:
            # Above expected — likely under-triaged
            scores[i] = 3.0
        elif rt < expected_low * 0.5:
            # Far below expected — likely over-triaged
            scores[i] = 1.0
        elif rt < expected_low:
            # Below expected — possibly over-triaged
            scores[i] = 2.0
        else:
            # Within expected range — consistent
            # Use midpoint of priority's numeric range
            from config.constants import PRIORITY_MAP
            scores[i] = float(PRIORITY_MAP.get(priority, 2))

    return np.clip(scores, 1.0, 4.0)


# ══════════════════════════════════════════════════════════════
#  SIGNAL 2 — MAIN COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_signal2(
    df: pd.DataFrame,
    method: str      = "continuous",
    clip_lower: float = 0.05,
    clip_upper: float = 0.95,
    config_path: str = "config/config.yaml",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Computes Signal 2 — Resolution Time Severity Score.

    Method options:
        "continuous" : Percentile-based continuous scaling (default)
                       Best for fusion — smooth 1–4 range
        "comparison" : Priority-vs-RT comparison scoring
                       Better for explainability

    Args:
        df          : Preprocessed DataFrame with Resolution_Time_Hours
        method      : Scoring method ("continuous" or "comparison")
        clip_lower  : Lower percentile clip for outlier removal
        clip_upper  : Upper percentile clip for outlier removal
        config_path : Path to config.yaml

    Returns:
        (scores, df)
        scores : np.ndarray (N,) of RT severity scores (1.0–4.0)
        df     : DataFrame with Severity_RT column added
    """
    cfg = load_config(config_path)

    method     = cfg["signal2"].get("method",      method)
    clip_lower = cfg["signal2"].get("clip_lower",  clip_lower)
    clip_upper = cfg["signal2"].get("clip_upper",  clip_upper)

    logger.info(f"Computing Signal 2 — Resolution Time Severity (method={method})")

    # ── Validate column ───────────────────────────────────────
    if COL_RT not in df.columns:
        raise ValueError(
            f"Column '{COL_RT}' not found. "
            f"Run preprocess_pipeline() first."
        )

    rt_values = df[COL_RT].values.astype(np.float64)

    # ── Check for missing values ──────────────────────────────
    n_missing = np.isnan(rt_values).sum()
    if n_missing > 0:
        log_warning(logger, f"{n_missing} missing RT values — filling with median")
        median    = np.nanmedian(rt_values)
        rt_values = np.where(np.isnan(rt_values), median, rt_values)

    # Log RT statistics
    log_step(logger, "Resolution time statistics")
    logger.info(f"   Min  : {rt_values.min():.1f} hrs")
    logger.info(f"   Max  : {rt_values.max():.1f} hrs")
    logger.info(f"   Mean : {rt_values.mean():.1f} hrs")
    logger.info(f"   P25  : {np.percentile(rt_values, 25):.1f} hrs")
    logger.info(f"   P50  : {np.percentile(rt_values, 50):.1f} hrs")
    logger.info(f"   P75  : {np.percentile(rt_values, 75):.1f} hrs")

    # ── Compute scores ────────────────────────────────────────
    if method == "continuous":
        log_step(logger, "Scoring via continuous percentile ranking")
        scores = score_by_continuous_percentile(
            rt_values  = rt_values,
            clip_lower = clip_lower,
            clip_upper = clip_upper,
        )

    elif method == "comparison":
        log_step(logger, "Scoring via priority-RT comparison")
        if COL_PRIORITY not in df.columns:
            log_warning(
                logger,
                f"Priority column not found for comparison method — "
                f"falling back to continuous"
            )
            scores = score_by_continuous_percentile(rt_values)
        else:
            scores = score_by_priority_comparison(
                rt_values  = rt_values,
                priorities = df[COL_PRIORITY],
            )
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose 'continuous' or 'comparison'."
        )

    logger.info(
        f"RT severity scores — "
        f"min={scores.min():.2f} | "
        f"max={scores.max():.2f} | "
        f"mean={scores.mean():.2f}"
    )

    # ── Add to DataFrame ──────────────────────────────────────
    df = df.copy()
    df[COL_RT_SCORE] = scores

    # Log distribution
    _log_score_distribution(scores, "RT Severity")

    # ── Log correlation with assigned priority ────────────────
    if "Priority_Numeric" in df.columns:
        corr = np.corrcoef(scores, df["Priority_Numeric"].values)[0, 1]
        logger.info(f"RT score correlation with assigned priority: {corr:.3f}")

    log_success(logger, f"Signal 2 complete — {len(scores):,} scores computed")
    return scores, df


# ══════════════════════════════════════════════════════════════
#  ABLATION — SIGNAL 2 ONLY
# ══════════════════════════════════════════════════════════════

def signal2_ablation(
    df: pd.DataFrame,
    method: str = "continuous",
) -> np.ndarray:
    """
    Runs Signal 2 in isolation.
    Used for the ablation table in README.

    Returns raw RT scores only.
    """
    logger.info("Running Signal 2 ablation (RT only)")

    rt_values = df[COL_RT].values.astype(np.float64)

    if method == "continuous":
        scores = score_by_continuous_percentile(rt_values)
    else:
        scores = score_by_priority_comparison(rt_values, df[COL_PRIORITY])

    _log_score_distribution(scores, "Signal 2 Ablation")
    return scores


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _log_score_distribution(scores: np.ndarray, label: str) -> None:
    """Logs how many RT scores fall in each severity bucket."""
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


def get_rt_interpretation(
    resolution_time: float,
    assigned_priority: str,
) -> str:
    """
    Generates a human-readable interpretation of resolution time
    relative to the expected window for the assigned priority.

    Used in Evidence Dossier generation.

    Args:
        resolution_time   : Actual RT in hours
        assigned_priority : Assigned priority label

    Returns:
        Interpretation string for the dossier
    """
    expected_low, expected_high = EXPECTED_RT.get(
        assigned_priority, (8, 24)
    )

    if resolution_time > expected_high * 2:
        return (
            f"Resolved in {resolution_time:.0f}hrs — "
            f"far exceeds the {expected_low}–{expected_high}hr window "
            f"expected for {assigned_priority} priority tickets, "
            f"strongly suggesting under-triage"
        )
    elif resolution_time > expected_high:
        return (
            f"Resolved in {resolution_time:.0f}hrs — "
            f"exceeds the {expected_low}–{expected_high}hr expected window "
            f"for {assigned_priority} priority"
        )
    elif resolution_time < expected_low * 0.5:
        return (
            f"Resolved in {resolution_time:.0f}hrs — "
            f"much faster than the {expected_low}–{expected_high}hr window "
            f"for {assigned_priority} priority, suggesting possible over-triage"
        )
    elif resolution_time < expected_low:
        return (
            f"Resolved in {resolution_time:.0f}hrs — "
            f"slightly below the {expected_low}–{expected_high}hr "
            f"expected window for {assigned_priority} priority"
        )
    else:
        return (
            f"Resolved in {resolution_time:.0f}hrs — "
            f"within the expected {expected_low}–{expected_high}hr "
            f"window for {assigned_priority} priority"
        )