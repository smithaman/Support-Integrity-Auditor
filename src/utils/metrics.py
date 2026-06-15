"""
metrics.py
Evaluation metric helpers.
Wraps sklearn metrics with SIA-specific threshold checks.
Returns pass/fail verdict per metric.
"""

# TODO: implement

# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/utils/metrics.py
# ─────────────────────────────────────────

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
)

from config.constants import (
    LABEL_CONSISTENT,
    LABEL_MISMATCH,
    LABEL_NAMES,
    THRESHOLD_ACCURACY,
    THRESHOLD_MACRO_F1,
    THRESHOLD_PER_CLASS_RECALL,
    ADVERSARIAL_PASS_COUNT,
)
from src.utils.helpers import save_json, ensure_dir
from src.utils.logger import (
    get_sia_logger,
    log_step,
    log_success,
    log_warning,
    log_metrics,
)

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict:
    """
    Computes all classification metrics for SIA evaluation.

    Required by project:
        - Binary Classification Accuracy  (>= 83%)
        - Macro F1 Score                  (>= 0.82)
        - Per-Class Recall                (>= 0.78 both)

    Additional:
        - Per-class Precision
        - ROC AUC
        - Cohen's Kappa
        - Full classification report string

    Args:
        y_true : np.ndarray (N,) true labels (0/1)
        y_pred : np.ndarray (N,) predicted labels (0/1)
        y_prob : np.ndarray (N,) mismatch probabilities (optional)

    Returns:
        Dict with all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1  = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    recall_consistent = recall_score(
        y_true, y_pred, pos_label=LABEL_CONSISTENT, zero_division=0
    )
    recall_mismatch = recall_score(
        y_true, y_pred, pos_label=LABEL_MISMATCH, zero_division=0
    )

    precision_consistent = precision_score(
        y_true, y_pred, pos_label=LABEL_CONSISTENT, zero_division=0
    )
    precision_mismatch = precision_score(
        y_true, y_pred, pos_label=LABEL_MISMATCH, zero_division=0
    )

    f1_consistent = f1_score(
        y_true, y_pred, pos_label=LABEL_CONSISTENT,
        average="binary", zero_division=0
    )
    f1_mismatch = f1_score(
        y_true, y_pred, pos_label=LABEL_MISMATCH,
        average="binary", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Cohen's Kappa
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except Exception:
        kappa = None

    # ROC AUC
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None

    # Full classification report
    report = classification_report(
        y_true, y_pred,
        target_names  = [LABEL_NAMES[0], LABEL_NAMES[1]],
        zero_division = 0,
    )

    metrics = {
        # ── Required metrics ──────────────────────────────────
        "accuracy":              round(float(accuracy),           4),
        "macro_f1":              round(float(macro_f1),           4),
        "recall_consistent":     round(float(recall_consistent),  4),
        "recall_mismatch":       round(float(recall_mismatch),    4),

        # ── Additional metrics ────────────────────────────────
        "micro_f1":              round(float(micro_f1),           4),
        "weighted_f1":           round(float(weighted_f1),        4),
        "precision_consistent":  round(float(precision_consistent), 4),
        "precision_mismatch":    round(float(precision_mismatch), 4),
        "f1_consistent":         round(float(f1_consistent),      4),
        "f1_mismatch":           round(float(f1_mismatch),        4),
        "cohen_kappa":           round(float(kappa), 4) if kappa is not None else None,
        "roc_auc":               round(float(roc_auc), 4) if roc_auc is not None else None,

        # ── Counts ────────────────────────────────────────────
        "n_samples":             int(len(y_true)),
        "n_true_consistent":     int((y_true == LABEL_CONSISTENT).sum()),
        "n_true_mismatch":       int((y_true == LABEL_MISMATCH).sum()),
        "n_pred_consistent":     int((y_pred == LABEL_CONSISTENT).sum()),
        "n_pred_mismatch":       int((y_pred == LABEL_MISMATCH).sum()),

        # ── Confusion matrix ──────────────────────────────────
        "confusion_matrix": {
            "tn": int(cm[0, 0]),   # True Consistent
            "fp": int(cm[0, 1]),   # Consistent predicted as Mismatch
            "fn": int(cm[1, 0]),   # Mismatch predicted as Consistent
            "tp": int(cm[1, 1]),   # True Mismatch
        },

        # ── Full report ───────────────────────────────────────
        "classification_report": report,
    }

    return metrics


# ══════════════════════════════════════════════════════════════
#  THRESHOLD VERIFICATION
# ══════════════════════════════════════════════════════════════

def check_verification_thresholds(metrics: Dict) -> Dict:
    """
    Checks all required verification thresholds.

    Thresholds (from project spec):
        Accuracy           >= 83%   (0.83)
        Macro F1           >= 0.82
        Recall Consistent  >= 0.78
        Recall Mismatch    >= 0.78

    Args:
        metrics : Dict from compute_classification_metrics()

    Returns:
        Dict with individual pass/fail + overall verdict:
        {
            "accuracy_pass":           True,
            "macro_f1_pass":           True,
            "recall_consistent_pass":  True,
            "recall_mismatch_pass":    False,
            "all_passed":              False,
            "failing_metrics":         ["recall_mismatch_pass"],
        }
    """
    checks = {
        "accuracy_pass": (
            metrics.get("accuracy", 0) >= THRESHOLD_ACCURACY,
            f"accuracy={metrics.get('accuracy', 0):.4f} "
            f"(required >= {THRESHOLD_ACCURACY})"
        ),
        "macro_f1_pass": (
            metrics.get("macro_f1", 0) >= THRESHOLD_MACRO_F1,
            f"macro_f1={metrics.get('macro_f1', 0):.4f} "
            f"(required >= {THRESHOLD_MACRO_F1})"
        ),
        "recall_consistent_pass": (
            metrics.get("recall_consistent", 0) >= THRESHOLD_PER_CLASS_RECALL,
            f"recall_consistent={metrics.get('recall_consistent', 0):.4f} "
            f"(required >= {THRESHOLD_PER_CLASS_RECALL})"
        ),
        "recall_mismatch_pass": (
            metrics.get("recall_mismatch", 0) >= THRESHOLD_PER_CLASS_RECALL,
            f"recall_mismatch={metrics.get('recall_mismatch', 0):.4f} "
            f"(required >= {THRESHOLD_PER_CLASS_RECALL})"
        ),
    }

    results      = {}
    failing      = []
    all_passed   = True

    logger.info("═" * 55)
    logger.info("  VERIFICATION THRESHOLD RESULTS")
    logger.info("═" * 55)

    for name, (passed, detail) in checks.items():
        results[name] = passed
        status        = "✔  PASS" if passed else "✘  FAIL"
        logger.info(f"  {name:<30} {status}")
        logger.info(f"    └─ {detail}")

        if not passed:
            all_passed = False
            failing.append(name)

    logger.info("═" * 55)

    results["all_passed"]      = all_passed
    results["failing_metrics"] = failing

    if all_passed:
        log_success(logger, "SUBMISSION VERIFIED — All thresholds met!")
    else:
        log_warning(
            logger,
            f"NOT VERIFIED — Failing: {failing}"
        )

    return results


# ══════════════════════════════════════════════════════════════
#  SIGNAL AGREEMENT METRICS
# ══════════════════════════════════════════════════════════════

def compute_signal_agreement_metrics(
    signal1_scores: np.ndarray,
    signal2_scores: np.ndarray,
) -> Dict:
    """
    Computes agreement metrics between Signal 1 and Signal 2.

    Required by project:
        "Pseudo-Label Signal Agreement (pairwise agreement
        between the two chosen signals)"

    Metrics:
        exact_agreement   : % with same rounded severity level
        within_1_level    : % within 1 severity level
        cohen_kappa       : Inter-rater agreement
        pearson_corr      : Linear correlation
        mean_abs_diff     : Mean |S1 - S2|

    Args:
        signal1_scores : np.ndarray (N,) Signal 1 scores (1–4)
        signal2_scores : np.ndarray (N,) Signal 2 scores (1–4)

    Returns:
        Dict with agreement metrics
    """
    s1 = np.round(np.clip(signal1_scores, 1.0, 4.0)).astype(int)
    s2 = np.round(np.clip(signal2_scores, 1.0, 4.0)).astype(int)

    exact    = float((s1 == s2).mean())
    within_1 = float((np.abs(s1 - s2) <= 1).mean())
    mad      = float(np.abs(signal1_scores - signal2_scores).mean())

    try:
        kappa = float(cohen_kappa_score(s1, s2))
    except Exception:
        kappa = None

    try:
        corr = float(np.corrcoef(signal1_scores, signal2_scores)[0, 1])
    except Exception:
        corr = None

    metrics = {
        "exact_agreement":  round(exact,    4),
        "within_1_level":   round(within_1, 4),
        "mean_abs_diff":    round(mad,       4),
        "cohen_kappa":      round(kappa, 4)  if kappa is not None else None,
        "pearson_corr":     round(corr, 4)   if corr  is not None else None,
    }

    logger.info("Signal Agreement Metrics:")
    for k, v in metrics.items():
        logger.info(f"   {k:<22} {v}")

    return metrics


# ══════════════════════════════════════════════════════════════
#  PSEUDO-LABEL QUALITY
# ══════════════════════════════════════════════════════════════

def compute_pseudo_label_stats(df: pd.DataFrame) -> Dict:
    """
    Computes statistics about the generated pseudo-labels.

    Useful for ablation table and README documentation.

    Args:
        df : DataFrame with Mismatch_Label, Mismatch_Type,
             Severity_Delta, Inferred_Severity columns

    Returns:
        Dict with pseudo-label statistics
    """
    from config.constants import (
        COL_MISMATCH_LABEL,
        COL_MISMATCH_TYPE,
        COL_DELTA_ABS,
        COL_INFERRED_SEV,
        MISMATCH_HIDDEN_CRISIS,
        MISMATCH_FALSE_ALARM,
    )

    stats = {}

    if COL_MISMATCH_LABEL in df.columns:
        n_total     = len(df)
        n_mismatch  = int(df[COL_MISMATCH_LABEL].sum())
        n_consistent = n_total - n_mismatch

        stats["n_total"]       = n_total
        stats["n_mismatch"]    = n_mismatch
        stats["n_consistent"]  = n_consistent
        stats["mismatch_rate"] = round(n_mismatch / n_total, 4)

    if COL_MISMATCH_TYPE in df.columns:
        type_counts = df[COL_MISMATCH_TYPE].value_counts().to_dict()
        stats["n_hidden_crisis"] = int(
            type_counts.get(MISMATCH_HIDDEN_CRISIS, 0)
        )
        stats["n_false_alarm"] = int(
            type_counts.get(MISMATCH_FALSE_ALARM, 0)
        )

    if COL_DELTA_ABS in df.columns:
        stats["mean_delta"]   = round(float(df[COL_DELTA_ABS].mean()), 4)
        stats["median_delta"] = round(float(df[COL_DELTA_ABS].median()), 4)
        stats["max_delta"]    = round(float(df[COL_DELTA_ABS].max()), 4)

    if COL_INFERRED_SEV in df.columns:
        inferred_dist = df[COL_INFERRED_SEV].value_counts().to_dict()
        stats["inferred_severity_distribution"] = {
            k: int(v) for k, v in inferred_dist.items()
        }

    return stats


# ══════════════════════════════════════════════════════════════
#  ABLATION TABLE BUILDER
# ══════════════════════════════════════════════════════════════

def build_ablation_table(
    ablation_results: Dict,
    save_path: str = "outputs/metrics/ablation_table.json",
) -> pd.DataFrame:
    """
    Formats ablation results into a clean DataFrame table.

    Expected input format (from fusion_pipeline.run_ablation):
    {
        "Signal 1 only (Semantic)": {
            "mismatch_rate": 0.28,
            "assign_agreement": 0.61,
        },
        ...
    }

    Returns:
        pd.DataFrame formatted for README markdown table
    """
    log_step(logger, "Building ablation table")

    rows = []
    for config_name, result in ablation_results.items():
        rows.append({
            "Configuration":      config_name,
            "Mismatch Rate":      f"{result.get('mismatch_rate', 0):.1%}",
            "Signal Agreement":   f"{result.get('assign_agreement', 0):.1%}",
            "Mean Fused Score":   f"{result.get('mean_fused_score', 0):.3f}",
        })

    table_df = pd.DataFrame(rows)

    # Save to JSON
    save_json(ablation_results, save_path)
    log_success(logger, f"Ablation table saved → {save_path}")

    # Print as markdown
    logger.info("\nAblation Table:")
    logger.info(table_df.to_markdown(index=False))

    return table_df


# ══════════════════════════════════════════════════════════════
#  FULL METRICS REPORT
# ══════════════════════════════════════════════════════════════

def build_full_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray            = None,
    signal1_scores: np.ndarray    = None,
    signal2_scores: np.ndarray    = None,
    pseudo_label_df: pd.DataFrame = None,
    adversarial_score: int        = None,
    save_path: str                = "outputs/metrics/evaluation.json",
) -> Dict:
    """
    Builds the complete metrics report for submission.

    Combines:
        - Classification metrics
        - Verification threshold results
        - Signal agreement metrics
        - Pseudo-label statistics
        - Adversarial test score

    Args:
        y_true            : True test labels
        y_pred            : Predicted test labels
        y_prob            : Mismatch probabilities
        signal1_scores    : Signal 1 scores for agreement
        signal2_scores    : Signal 2 scores for agreement
        pseudo_label_df   : DataFrame with pseudo-labels
        adversarial_score : Score from adversarial testing
        save_path         : Path to save full report

    Returns:
        Complete metrics report dict
    """
    logger.info("Building full metrics report")

    report = {}

    # ── Classification metrics ────────────────────────────────
    log_step(logger, "Computing classification metrics")
    clf_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    report["classification"] = {
        k: v for k, v in clf_metrics.items()
        if k != "classification_report"
    }
    report["classification_report"] = clf_metrics["classification_report"]

    # ── Threshold verification ────────────────────────────────
    log_step(logger, "Checking verification thresholds")
    threshold_results = check_verification_thresholds(clf_metrics)
    report["thresholds"] = threshold_results

    # ── Signal agreement ──────────────────────────────────────
    if signal1_scores is not None and signal2_scores is not None:
        log_step(logger, "Computing signal agreement")
        agreement = compute_signal_agreement_metrics(
            signal1_scores, signal2_scores
        )
        report["signal_agreement"] = agreement

    # ── Pseudo-label stats ────────────────────────────────────
    if pseudo_label_df is not None:
        log_step(logger, "Computing pseudo-label statistics")
        pl_stats = compute_pseudo_label_stats(pseudo_label_df)
        report["pseudo_label_stats"] = pl_stats

    # ── Adversarial score ─────────────────────────────────────
    if adversarial_score is not None:
        report["adversarial"] = {
            "score":         adversarial_score,
            "total":         10,
            "pass_threshold": ADVERSARIAL_PASS_COUNT,
            "bonus_earned":  adversarial_score >= ADVERSARIAL_PASS_COUNT,
        }

    # ── Save report ───────────────────────────────────────────
    ensure_dir("outputs/metrics")
    save_json(report, save_path)
    log_success(logger, f"Full metrics report saved → {save_path}")

    return report
