# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/pseudo_labels/generate_labels.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config.constants import (
    COL_TICKET_ID,
    COL_PRIORITY,
    COL_PRIORITY_NUM,
    COL_FUSED_SCORE,
    COL_INFERRED_SEV,
    COL_INFERRED_NUM,
    COL_DELTA,
    COL_DELTA_ABS,
    COL_MISMATCH_LABEL,
    COL_MISMATCH_TYPE,
    MISMATCH_THRESHOLD,
    MISMATCH_HIDDEN_CRISIS,
    MISMATCH_FALSE_ALARM,
    CONSISTENT,
    LABEL_MISMATCH,
    LABEL_CONSISTENT,
    PRIORITY_MAP,
)
from src.utils.helpers import (
    load_config,
    save_json,
    ensure_dir,
    check_mismatch_rate,
    print_label_distribution,
)
from src.utils.logger import (
    get_sia_logger,
    log_step,
    log_success,
    log_warning,
    log_metrics,
)

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  DELTA COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_severity_delta(
    df: pd.DataFrame,
    fused_scores: np.ndarray = None,
) -> pd.DataFrame:
    """
    Computes the signed and absolute severity delta between
    inferred severity and assigned priority.

        delta     = Inferred_Numeric - Priority_Numeric
        delta_abs = |delta|

    Positive delta → ticket under-triaged (Hidden Crisis)
    Negative delta → ticket over-triaged  (False Alarm)
    Zero delta     → consistent assignment

    Args:
        df           : DataFrame with Priority_Numeric + Inferred_Numeric
        fused_scores : Optional raw fused scores (more precise than rounded)

    Returns:
        DataFrame with Severity_Delta and Severity_Delta_Abs columns added
    """
    log_step(logger, "Computing severity delta")

    df = df.copy()

    # Use raw fused scores for delta if available
    # (more precise than rounded integer levels)
    if fused_scores is not None:
        inferred_numeric = fused_scores
    elif COL_INFERRED_NUM in df.columns:
        inferred_numeric = df[COL_INFERRED_NUM].values.astype(float)
    else:
        raise ValueError(
            f"No inferred severity found. "
            f"Run fusion_pipeline() first."
        )

    if COL_PRIORITY_NUM not in df.columns:
        raise ValueError(
            f"Column '{COL_PRIORITY_NUM}' not found. "
            f"Run preprocess_pipeline() first."
        )

    assigned_numeric = df[COL_PRIORITY_NUM].values.astype(float)

    df[COL_DELTA]     = inferred_numeric - assigned_numeric
    df[COL_DELTA_ABS] = np.abs(df[COL_DELTA].values)

    logger.info(
        f"Delta stats — "
        f"mean abs: {df[COL_DELTA_ABS].mean():.3f} | "
        f"max abs: {df[COL_DELTA_ABS].max():.3f}"
    )

    return df


# ══════════════════════════════════════════════════════════════
#  LABEL GENERATION
# ══════════════════════════════════════════════════════════════

def generate_binary_labels(
    df: pd.DataFrame,
    threshold: float = MISMATCH_THRESHOLD,
) -> pd.DataFrame:
    """
    Generates binary pseudo-labels from severity delta.

        |delta| >= threshold → Mismatch  (1)
        |delta| <  threshold → Consistent (0)

    Default threshold: 1.5
    This requires at least a 2-level gap to flag a mismatch,
    avoiding false positives from minor scoring noise.

    Args:
        df        : DataFrame with Severity_Delta_Abs column
        threshold : Mismatch threshold (default 1.5)

    Returns:
        DataFrame with Mismatch_Label column added
    """
    log_step(logger, f"Generating binary labels (threshold={threshold})")

    if COL_DELTA_ABS not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA_ABS}' not found. "
            f"Run compute_severity_delta() first."
        )

    df = df.copy()
    df[COL_MISMATCH_LABEL] = (
        df[COL_DELTA_ABS] >= threshold
    ).astype(int)

    return df


def generate_mismatch_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns mismatch type based on signed delta direction.

        delta >= +threshold → Hidden Crisis  (under-triaged)
        delta <= -threshold → False Alarm    (over-triaged)
        else                → Consistent

    Args:
        df : DataFrame with Severity_Delta and Mismatch_Label columns

    Returns:
        DataFrame with Mismatch_Type column added
    """
    log_step(logger, "Assigning mismatch types")

    if COL_DELTA not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA}' not found. "
            f"Run compute_severity_delta() first."
        )

    df = df.copy()

    def assign_type(row) -> str:
        if row[COL_MISMATCH_LABEL] == LABEL_CONSISTENT:
            return CONSISTENT
        return (
            MISMATCH_HIDDEN_CRISIS
            if row[COL_DELTA] > 0
            else MISMATCH_FALSE_ALARM
        )

    df[COL_MISMATCH_TYPE] = df.apply(assign_type, axis=1)

    # Log type distribution
    type_counts = df[COL_MISMATCH_TYPE].value_counts()
    logger.info("Mismatch Type Distribution:")
    for mtype, count in type_counts.items():
        pct = count / len(df) * 100
        logger.info(f"   {mtype:<20} {count:>6,}  ({pct:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════
#  THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════

def find_optimal_threshold(
    df: pd.DataFrame,
    target_min: float = 0.20,
    target_max: float = 0.40,
    candidates: list  = None,
) -> Tuple[float, Dict]:
    """
    Finds the threshold that produces a mismatch rate
    within the target range (20–40%).

    Tests multiple candidate thresholds and returns the
    one closest to the midpoint of the target range.

    Args:
        df          : DataFrame with Severity_Delta_Abs
        target_min  : Minimum acceptable mismatch rate (default 0.20)
        target_max  : Maximum acceptable mismatch rate (default 0.40)
        candidates  : List of thresholds to test

    Returns:
        (best_threshold, results_dict)
    """
    if candidates is None:
        candidates = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]

    log_step(logger, "Finding optimal mismatch threshold")

    delta_abs = df[COL_DELTA_ABS].values
    results   = {}
    target_mid = (target_min + target_max) / 2

    logger.info(f"Testing {len(candidates)} threshold candidates:")

    for t in candidates:
        rate = (delta_abs >= t).mean()
        in_range = target_min <= rate <= target_max
        results[t] = {
            "mismatch_rate": round(float(rate), 4),
            "in_range":      in_range,
        }
        status = "✔" if in_range else "✘"
        logger.info(
            f"   threshold={t:.1f}  "
            f"mismatch_rate={rate:.1%}  {status}"
        )

    # Find best: prefer in-range, closest to target midpoint
    in_range = {
        t: r for t, r in results.items()
        if r["in_range"]
    }

    if in_range:
        best = min(
            in_range.keys(),
            key=lambda t: abs(results[t]["mismatch_rate"] - target_mid)
        )
        log_success(
            logger,
            f"Best threshold: {best} "
            f"(mismatch rate: {results[best]['mismatch_rate']:.1%})"
        )
    else:
        # No threshold in range — pick closest to target_mid
        best = min(
            candidates,
            key=lambda t: abs(results[t]["mismatch_rate"] - target_mid)
        )
        log_warning(
            logger,
            f"No threshold achieved target range. "
            f"Using closest: {best} "
            f"(rate: {results[best]['mismatch_rate']:.1%})"
        )

    return best, results


# ══════════════════════════════════════════════════════════════
#  SIGNAL SAVING (for ablation)
# ══════════════════════════════════════════════════════════════

def save_signal_scores(
    df: pd.DataFrame,
    save_path: str = "data/processed/signal_scores.csv",
) -> None:
    """
    Saves individual signal scores alongside pseudo-labels
    for ablation analysis.

    Columns saved:
        Ticket_ID, Priority_Level, Priority_Numeric,
        Severity_Semantic, Severity_RT, Fused_Score,
        Inferred_Severity, Severity_Delta, Mismatch_Label,
        Mismatch_Type
    """
    cols_to_save = [
        col for col in [
            COL_TICKET_ID,
            COL_PRIORITY,
            COL_PRIORITY_NUM,
            "Severity_Semantic",
            "Severity_RT",
            COL_FUSED_SCORE,
            COL_INFERRED_SEV,
            COL_INFERRED_NUM,
            COL_DELTA,
            COL_DELTA_ABS,
            COL_MISMATCH_LABEL,
            COL_MISMATCH_TYPE,
        ]
        if col in df.columns
    ]

    ensure_dir(Path(save_path).parent)
    df[cols_to_save].to_csv(save_path, index=False)
    log_success(logger, f"Signal scores saved → {save_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════

def pseudo_label_pipeline(
    df: pd.DataFrame,
    fused_scores: np.ndarray = None,
    config_path: str         = "config/config.yaml",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full pseudo-label generation pipeline.
    Runs after fusion_pipeline().

    Steps:
        1. Compute severity delta (inferred vs assigned)
        2. Auto-tune threshold if needed
        3. Generate binary labels (Mismatch=1 / Consistent=0)
        4. Assign mismatch type (Hidden Crisis / False Alarm)
        5. Validate mismatch rate is in target range
        6. Save signal scores for ablation

    Args:
        df           : DataFrame after fusion_pipeline()
        fused_scores : Raw fused scores (more precise than rounded)
        config_path  : Path to config.yaml

    Returns:
        (df, stats)
        df    : DataFrame with pseudo-label columns added
        stats : Dict with label distribution stats
    """
    cfg       = load_config(config_path)
    threshold = cfg["pseudo_labels"].get("threshold", MISMATCH_THRESHOLD)
    save_sigs = cfg["pseudo_labels"].get("save_signals", True)
    min_rate  = cfg["fusion"]["target_mismatch_rate"]["min"]
    max_rate  = cfg["fusion"]["target_mismatch_rate"]["max"]

    logger.info("Starting pseudo-label pipeline")

    # ── Step 1: Compute delta ─────────────────────────────────
    df = compute_severity_delta(df, fused_scores=fused_scores)

    # ── Step 2: Auto-tune threshold if needed ─────────────────
    preview_rate = (df[COL_DELTA_ABS] >= threshold).mean()
    if not (min_rate <= preview_rate <= max_rate):
        log_warning(
            logger,
            f"Default threshold {threshold} gives rate {preview_rate:.1%} "
            f"outside target {min_rate:.0%}–{max_rate:.0%}. Auto-tuning..."
        )
        threshold, _ = find_optimal_threshold(
            df=df,
            target_min=min_rate,
            target_max=max_rate,
        )

    # ── Step 3: Generate binary labels ───────────────────────
    df = generate_binary_labels(df, threshold=threshold)

    # ── Step 4: Assign mismatch types ────────────────────────
    df = generate_mismatch_type(df)

    # ── Step 5: Validate mismatch rate ───────────────────────
    mismatch_rate, is_ok = check_mismatch_rate(
        df       = df,
        label_col = COL_MISMATCH_LABEL,
        min_rate  = min_rate,
        max_rate  = max_rate,
    )

    # ── Step 6: Log full distribution ────────────────────────
    print_label_distribution(df, COL_MISMATCH_LABEL, "Pseudo-Label Distribution")

    # ── Step 7: Save signal scores ───────────────────────────
    if save_sigs:
        save_signal_scores(
            df=df,
            save_path="data/processed/signal_scores.csv",
        )

    # ── Step 8: Save pseudo-labels CSV ───────────────────────
    pseudo_labels_path = cfg["data"]["pseudo_labels_path"]
    ensure_dir(Path(pseudo_labels_path).parent)
    df.to_csv(pseudo_labels_path, index=False)
    log_success(logger, f"Pseudo-labels saved → {pseudo_labels_path}")

    # ── Build stats dict ──────────────────────────────────────
    n_total     = len(df)
    n_mismatch  = int(df[COL_MISMATCH_LABEL].sum())
    n_consistent = n_total - n_mismatch
    n_hidden    = int((df[COL_MISMATCH_TYPE] == MISMATCH_HIDDEN_CRISIS).sum())
    n_false     = int((df[COL_MISMATCH_TYPE] == MISMATCH_FALSE_ALARM).sum())

    stats = {
        "total_tickets":    n_total,
        "n_mismatch":       n_mismatch,
        "n_consistent":     n_consistent,
        "mismatch_rate":    round(mismatch_rate, 4),
        "n_hidden_crisis":  n_hidden,
        "n_false_alarm":    n_false,
        "threshold_used":   threshold,
        "rate_in_target":   is_ok,
    }

    log_metrics(logger, stats, title="Pseudo-Label Stats")
    log_success(logger, "Pseudo-label pipeline complete")

    return df, stats