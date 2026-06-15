# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/classifier/evaluate.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification

from config.constants import (
    LABEL_CONSISTENT,
    LABEL_MISMATCH,
    LABEL_NAMES,
    THRESHOLD_ACCURACY,
    THRESHOLD_MACRO_F1,
    THRESHOLD_PER_CLASS_RECALL,
    ADVERSARIAL_PASS_COUNT,
    COL_TICKET_ID,
    COL_MISMATCH_LABEL,
    COL_PREDICTION,
    COL_CONFIDENCE,
)
from src.classifier.dataset import (
    load_tokenizer,
    build_dataloader,
    SIADataset,
)
from src.utils.helpers import (
    load_config,
    save_json,
    ensure_dir,
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
#  MODEL LOADER FOR EVALUATION
# ══════════════════════════════════════════════════════════════

def load_trained_model(
    model_dir: str   = "outputs/models/deberta_classifier/",
    device: torch.device = None,
) -> Tuple[AutoModelForSequenceClassification, object]:
    """
    Loads the saved DeBERTa checkpoint for evaluation.

    Args:
        model_dir : Directory containing saved model + tokenizer
        device    : Target device

    Returns:
        (model, tokenizer)
    """
    log_step(logger, f"Loading trained model from {model_dir}")

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run train_classifier() first."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    tokenizer = load_tokenizer(model_dir)

    log_success(logger, f"Model loaded on {device}")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════

def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference and collects predictions, labels, confidences.

    Args:
        model  : Trained DeBERTa model
        loader : DataLoader
        device : Target device

    Returns:
        (all_preds, all_labels, all_confidences)
        all_preds       : np.ndarray (N,) predicted labels (0/1)
        all_labels      : np.ndarray (N,) true labels (0/1)
        all_confidences : np.ndarray (N,) mismatch probability (0–1)
    """
    model.eval()

    all_preds   = []
    all_labels  = []
    all_confs   = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_confs.extend(probs[:, LABEL_MISMATCH].tolist())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confs),
    )


# ══════════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray = None,
) -> Dict:
    """
    Computes all evaluation metrics required by the project.

    Required metrics:
        - Binary Classification Accuracy  (>= 83%)
        - Macro F1 Score                  (>= 0.82)
        - Per-Class Recall                (>= 0.78 both classes)

    Additional metrics:
        - Per-class Precision
        - ROC AUC Score
        - Full classification report

    Args:
        y_true : True labels
        y_pred : Predicted labels
        y_conf : Confidence scores (mismatch probability)

    Returns:
        Dict with all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro",    zero_division=0)

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

    # ROC AUC (requires confidence scores)
    roc_auc = None
    if y_conf is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_conf)
        except Exception:
            roc_auc = None

    # Full classification report
    report = classification_report(
        y_true, y_pred,
        target_names = [LABEL_NAMES[0], LABEL_NAMES[1]],
        zero_division = 0,
    )

    metrics = {
        "accuracy":            round(float(accuracy),            4),
        "macro_f1":            round(float(macro_f1),            4),
        "micro_f1":            round(float(micro_f1),            4),
        "recall_consistent":   round(float(recall_consistent),   4),
        "recall_mismatch":     round(float(recall_mismatch),     4),
        "precision_consistent":round(float(precision_consistent),4),
        "precision_mismatch":  round(float(precision_mismatch),  4),
        "roc_auc":             round(float(roc_auc), 4) if roc_auc else None,
        "classification_report": report,
    }

    return metrics


# ══════════════════════════════════════════════════════════════
#  THRESHOLD VERIFICATION
# ══════════════════════════════════════════════════════════════

def check_verification_thresholds(metrics: Dict) -> Dict:
    """
    Checks all required verification thresholds.

    Thresholds:
        Accuracy           >= 83%
        Macro F1           >= 0.82
        Recall Consistent  >= 0.78
        Recall Mismatch    >= 0.78

    Args:
        metrics : Dict from compute_all_metrics()

    Returns:
        Dict with pass/fail per threshold + overall verdict
    """
    checks = {
        "accuracy_pass": (
            metrics["accuracy"] >= THRESHOLD_ACCURACY,
            f"{metrics['accuracy']:.4f} >= {THRESHOLD_ACCURACY}",
        ),
        "macro_f1_pass": (
            metrics["macro_f1"] >= THRESHOLD_MACRO_F1,
            f"{metrics['macro_f1']:.4f} >= {THRESHOLD_MACRO_F1}",
        ),
        "recall_consistent_pass": (
            metrics["recall_consistent"] >= THRESHOLD_PER_CLASS_RECALL,
            f"{metrics['recall_consistent']:.4f} >= {THRESHOLD_PER_CLASS_RECALL}",
        ),
        "recall_mismatch_pass": (
            metrics["recall_mismatch"] >= THRESHOLD_PER_CLASS_RECALL,
            f"{metrics['recall_mismatch']:.4f} >= {THRESHOLD_PER_CLASS_RECALL}",
        ),
    }

    results     = {}
    all_passed  = True

    logger.info("=" * 55)
    logger.info("  VERIFICATION THRESHOLD RESULTS")
    logger.info("=" * 55)

    for check_name, (passed, detail) in checks.items():
        results[check_name] = passed
        status    = "✔  PASS" if passed else "✘  FAIL"
        if not passed:
            all_passed = False
        logger.info(f"  {check_name:<30} {status}  ({detail})")

    logger.info("=" * 55)

    results["all_passed"] = all_passed

    if all_passed:
        log_success(logger, "SUBMISSION VERIFIED — All thresholds met!")
    else:
        log_warning(
            logger,
            "SUBMISSION NOT VERIFIED — Some thresholds not met. "
            "Continue training or adjust hyperparameters."
        )

    return results


# ══════════════════════════════════════════════════════════════
#  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "outputs/metrics/confusion_matrix.png",
) -> None:
    """
    Plots and saves a styled confusion matrix.

    Args:
        y_true    : True labels
        y_pred    : Predicted labels
        save_path : Where to save the figure
    """
    ensure_dir(Path(save_path).parent)

    cm     = confusion_matrix(y_true, y_pred)
    labels = [LABEL_NAMES[0], LABEL_NAMES[1]]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        xticklabels = labels,
        yticklabels = labels,
        ax         = ax,
        linewidths = 0.5,
    )

    ax.set_title(
        "SIA Classifier — Confusion Matrix",
        fontsize = 14,
        pad      = 15,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)

    # Add per-class accuracy annotations
    for i in range(len(labels)):
        row_sum = cm[i].sum()
        if row_sum > 0:
            acc = cm[i, i] / row_sum
            ax.text(
                len(labels) + 0.1, i + 0.5,
                f"Recall: {acc:.1%}",
                va="center", fontsize=10, color="gray"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    log_success(logger, f"Confusion matrix saved → {save_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════

def evaluation_pipeline(
    test_df: pd.DataFrame,
    model_dir: str   = "outputs/models/deberta_classifier/",
    config_path: str = "config/config.yaml",
) -> Dict:
    """
    Full evaluation pipeline on the test set.

    Steps:
        1. Load trained model + tokenizer
        2. Build test DataLoader
        3. Get predictions + confidences
        4. Compute all metrics
        5. Check verification thresholds
        6. Plot confusion matrix
        7. Save results to outputs/metrics/

    Args:
        test_df     : Test split DataFrame
        model_dir   : Saved model directory
        config_path : Path to config.yaml

    Returns:
        Dict with all metrics + threshold results
    """
    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting evaluation pipeline")

    # ── Load model ────────────────────────────────────────────
    model, tokenizer = load_trained_model(model_dir, device)

    # ── Build DataLoader ──────────────────────────────────────
    log_step(logger, "Building test DataLoader")
    test_loader = build_dataloader(
        df         = test_df,
        tokenizer  = tokenizer,
        batch_size = cfg["classifier"]["eval_batch_size"],
        max_length = cfg["classifier"]["max_length"],
        shuffle    = False,
        is_test    = False,
    )

    # ── Get predictions ───────────────────────────────────────
    log_step(logger, "Running inference on test set")
    y_pred, y_true, y_conf = get_predictions(model, test_loader, device)

    # ── Compute metrics ───────────────────────────────────────
    log_step(logger, "Computing evaluation metrics")
    metrics = compute_all_metrics(y_true, y_pred, y_conf)

    log_metrics(
        logger,
        {k: v for k, v in metrics.items() if k != "classification_report"},
        title="Test Set Metrics"
    )

    # Print full classification report
    logger.info("\n" + metrics["classification_report"])

    # ── Threshold verification ────────────────────────────────
    log_step(logger, "Checking verification thresholds")
    threshold_results = check_verification_thresholds(metrics)

    # ── Confusion matrix ──────────────────────────────────────
    log_step(logger, "Plotting confusion matrix")
    plot_confusion_matrix(
        y_true    = y_true,
        y_pred    = y_pred,
        save_path = cfg["evaluation"]["confusion_matrix_path"],
    )

    # ── Add predictions to test_df ────────────────────────────
    test_df = test_df.copy()
    test_df[COL_PREDICTION] = y_pred
    test_df[COL_CONFIDENCE] = y_conf

    # ── Save results ──────────────────────────────────────────
    results = {
        "metrics":            {k: v for k, v in metrics.items()
                               if k != "classification_report"},
        "threshold_results":  threshold_results,
        "classification_report": metrics["classification_report"],
        "n_test_samples":     int(len(y_true)),
        "n_mismatch_true":    int(y_true.sum()),
        "n_mismatch_pred":    int(y_pred.sum()),
    }

    save_json(results, cfg["evaluation"]["metrics_path"])
    log_success(logger, f"Evaluation results saved → {cfg['evaluation']['metrics_path']}")

    log_success(logger, "Evaluation pipeline complete")
    return results