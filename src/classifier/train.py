"""
train.py
DeBERTa-v3-small Fine-Tuning
Trains binary classifier on pseudo-labeled data.
Uses weighted CrossEntropyLoss to handle class imbalance.
Saves best checkpoint based on macro F1 on validation set.
"""

# TODO: implement

# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/classifier/train.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, Optional, Tuple
import torch.nn.functional as nn_functional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)

from config.constants import (
    LABEL_CONSISTENT,
    LABEL_MISMATCH,
    LABEL_NAMES,
    THRESHOLD_ACCURACY,
    THRESHOLD_MACRO_F1,
    THRESHOLD_PER_CLASS_RECALL,
)
from src.classifier.dataset import (
    load_tokenizer,
    build_all_dataloaders,
    compute_class_weights,
    validate_dataset,
)
from src.utils.helpers import (
    load_config,
    set_seed,
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
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════

def load_model(
    model_name: str      = "microsoft/deberta-v3-small",
    num_labels: int      = 2,
    device: torch.device = None,
) -> AutoModelForSequenceClassification:

    log_step(logger, f"Loading model: {model_name}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels  = num_labels,
            ignore_mismatched_sizes = True,
        )

        # Force float32 — prevents NaN from mixed precision
        model = model.float()

        if device:
            model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        n_train  = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
        logger.info(
            f"Model parameters — "
            f"total={n_params:,} | trainable={n_train:,} | "
            f"dtype=float32"
        )

        log_success(logger, f"Model loaded on {device} (float32)")
        return model

    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}': {e}"
        )


# ══════════════════════════════════════════════════════════════
#  WEIGHTED LOSS
# ══════════════════════════════════════════════════════════════

class WeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropyLoss with class weights for imbalanced data.
    Handles mixed precision (float16/float32) automatically.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        device: torch.device = None,
    ):
        super().__init__()
        self.weights = class_weights.to(device) if device else class_weights
        self.device  = device

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # Cast weights to match logits dtype
        # Fixes: "expected scalar type Half but found Float"
        weights = self.weights.to(
            device = logits.device,
            dtype  = logits.dtype,     # ← match whatever the model outputs
        )
        return nn.functional.cross_entropy(logits, labels, weight=weights)


# ══════════════════════════════════════════════════════════════
#  TRAINING STEP
# ══════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    scaler=None,
) -> Dict:
    """
    Runs one full training epoch with NaN protection.
    """
    model.train()

    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = len(loader)
    nan_count  = 0

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass — always in float32 for stability
        with torch.cuda.amp.autocast(enabled=False):
            # Cast inputs to float32 explicitly
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )
            # Force float32
            logits = outputs.logits.float()
            loss   = loss_fn(logits, labels)

        # NaN guard — skip batch if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            logger.warning(
                f"NaN/Inf loss at step {step} — skipping batch"
            )
            optimizer.zero_grad()
            continue

        # Backward
        loss.backward()

        # Gradient clipping — critical for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for NaN gradients
        has_nan_grad = any(
            p.grad is not None and
            (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters()
        )

        if has_nan_grad:
            nan_count += 1
            logger.warning(
                f"NaN gradient at step {step} — skipping update"
            )
            optimizer.zero_grad()
            continue

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % max(1, n_batches // 10) == 0:
            logger.info(
                f"   Epoch {epoch} | "
                f"Step {step+1}/{n_batches} | "
                f"Loss: {loss.item():.4f}"
            )

    if nan_count > 0:
        log_warning(logger, f"{nan_count} NaN batches skipped this epoch")

    valid_steps = n_batches - nan_count
    avg_loss    = total_loss / max(valid_steps, 1)
    accuracy    = accuracy_score(all_labels, all_preds) if all_preds else 0.0
    macro_f1    = f1_score(
        all_labels, all_preds,
        average="macro", zero_division=0
    ) if all_preds else 0.0

    return {
        "loss":     round(avg_loss, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
    }


# ══════════════════════════════════════════════════════════════
#  EVALUATION STEP
# ══════════════════════════════════════════════════════════════

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    split_name: str = "val",
) -> Dict:
    """
    Evaluates the model on a given split.

    Args:
        model      : DeBERTa model
        loader     : Val or Test DataLoader
        loss_fn    : WeightedCrossEntropyLoss
        device     : Target device
        split_name : Name for logging

    Returns:
        Dict with loss, accuracy, macro_f1, per_class_recall
    """
    model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )
            logits = outputs.logits
            loss   = loss_fn(logits, labels)

            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].tolist())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels, all_preds,
        average="macro", zero_division=0
    )
    recall_consistent = recall_score(
        all_labels, all_preds,
        pos_label=LABEL_CONSISTENT, zero_division=0
    )
    recall_mismatch = recall_score(
        all_labels, all_preds,
        pos_label=LABEL_MISMATCH, zero_division=0
    )

    metrics = {
        "loss":              round(avg_loss,          4),
        "accuracy":          round(accuracy,           4),
        "macro_f1":          round(macro_f1,           4),
        "recall_consistent": round(recall_consistent,  4),
        "recall_mismatch":   round(recall_mismatch,    4),
    }

    log_metrics(logger, metrics, title=f"{split_name.upper()} Metrics")
    _check_thresholds(metrics, split_name)

    return metrics


def _check_thresholds(metrics: Dict, split_name: str) -> None:
    """Logs pass/fail for each verification threshold."""
    checks = {
        "Accuracy >= 83%":          metrics["accuracy"]          >= THRESHOLD_ACCURACY,
        "Macro F1 >= 0.82":         metrics["macro_f1"]          >= THRESHOLD_MACRO_F1,
        "Recall Consistent >= 78%": metrics["recall_consistent"] >= THRESHOLD_PER_CLASS_RECALL,
        "Recall Mismatch >= 78%":   metrics["recall_mismatch"]   >= THRESHOLD_PER_CLASS_RECALL,
    }

    all_pass = all(checks.values())
    logger.info(f"Threshold checks ({split_name}):")
    for check, passed in checks.items():
        status = "✔ PASS" if passed else "✘ FAIL"
        logger.info(f"   {check:<35} {status}")

    if all_pass:
        log_success(logger, "All verification thresholds met!")
    else:
        log_warning(logger, "Some thresholds not yet met — continue training")


# ══════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_classifier(
    train_df,
    val_df,
    test_df,
    config_path: str = "config/config.yaml",
) -> Tuple[nn.Module, Dict]:
    """
    Full DeBERTa-v3-small training pipeline.

    Steps:
        1. Set seed for reproducibility
        2. Load tokenizer and build DataLoaders
        3. Compute class weights
        4. Load model
        5. Setup optimizer + scheduler
        6. Train for N epochs — save best on macro_f1
        7. Evaluate best model on test set
        8. Save model + training history

    Args:
        train_df    : Training split DataFrame
        val_df      : Validation split DataFrame
        test_df     : Test split DataFrame
        config_path : Path to config.yaml

    Returns:
        (best_model, test_metrics)
    """
    cfg = load_config(config_path)
    c   = cfg["classifier"]

    # ── Setup ─────────────────────────────────────────────────
    set_seed(c["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Training device: {device}")

    # ── Validate datasets ─────────────────────────────────────
    validate_dataset(train_df, "train")
    validate_dataset(val_df,   "val")
    validate_dataset(test_df,  "test")

    # ── Tokenizer + DataLoaders ───────────────────────────────
    log_step(logger, "Setting up tokenizer and DataLoaders")
    tokenizer = load_tokenizer(c["model_name"])

    train_loader, val_loader, test_loader = build_all_dataloaders(
        train_df    = train_df,
        val_df      = val_df,
        test_df     = test_df,
        tokenizer   = tokenizer,
        config_path = config_path,
    )

    # ── Class weights ─────────────────────────────────────────
    log_step(logger, "Computing class weights")
    class_weights = compute_class_weights(train_df)
    loss_fn       = WeightedCrossEntropyLoss(
        class_weights = class_weights,
        device        = device,
    )

    # ── Model ─────────────────────────────────────────────────
    model = load_model(
        model_name = c["model_name"],
        num_labels = c["num_labels"],
        device     = device,
    )

    # ── Optimizer ─────────────────────────────────────────────
    # ── Optimizer ─────────────────────────────────────────────
    # eps=1e-7 instead of default 1e-8 — prevents NaN in float32
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = c["learning_rate"],
        weight_decay = c["weight_decay"],
        eps          = 1e-7,
    )

    # ── Scheduler ─────────────────────────────────────────────
    total_steps  = len(train_loader) * c["num_epochs"]
    warmup_steps = int(total_steps  * c["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer          = optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    logger.info(
        f"Training setup — "
        f"epochs={c['num_epochs']} | "
        f"total_steps={total_steps} | "
        f"warmup_steps={warmup_steps} | "
        f"lr={c['learning_rate']}"
    )

    # ── Training loop ─────────────────────────────────────────
    best_f1    = 0.0
    best_epoch = 0
    best_state = None
    history    = []

    save_dir = Path(c["save_dir"])
    ensure_dir(save_dir)

    for epoch in range(1, c["num_epochs"] + 1):
        log_step(logger, f"Epoch {epoch}/{c['num_epochs']}")

        # Train
        train_metrics = train_one_epoch(
            model     = model,
            loader    = train_loader,
            optimizer = optimizer,
            scheduler = scheduler,
            loss_fn   = loss_fn,
            device    = device,
            epoch     = epoch,
        )

        # Validate
        val_metrics = evaluate_model(
            model      = model,
            loader     = val_loader,
            loss_fn    = loss_fn,
            device     = device,
            split_name = "val",
        )

        # Record history
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val":   val_metrics,
        })

        logger.info(
            f"Epoch {epoch} summary — "
            f"train_loss={train_metrics['loss']} | "
            f"val_f1={val_metrics['macro_f1']} | "
            f"val_acc={val_metrics['accuracy']}"
        )

        # ── Save best model ───────────────────────────────────
        if val_metrics["macro_f1"] > best_f1:
            best_f1    = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            log_success(
                logger,
                f"New best model saved — "
                f"epoch={epoch} | val_f1={best_f1:.4f}"
            )

    logger.info(
        f"Training complete — "
        f"best epoch={best_epoch} | "
        f"best val_f1={best_f1:.4f}"
    )

    # ── Load best model for test evaluation ───────────────────
    log_step(logger, "Loading best model for test evaluation")
    model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────
    log_step(logger, "Evaluating on test set")
    test_metrics = evaluate_model(
        model      = model,
        loader     = test_loader,
        loss_fn    = loss_fn,
        device     = device,
        split_name = "test",
    )

    # ── Save metadata ─────────────────────────────────────────
    metadata = {
        "model_name":   c["model_name"],
        "num_labels":   c["num_labels"],
        "label_map":    {str(k): v for k, v in LABEL_NAMES.items()},
        "best_epoch":   best_epoch,
        "best_val_f1":  best_f1,
        "test_metrics": test_metrics,
        "history":      history,
        "config":       c,
    }

    save_json(metadata, "outputs/models/metadata.json")
    save_json(history,  "outputs/metrics/training_history.json")

    log_success(logger, "Classifier training complete")
    return model, test_metrics
