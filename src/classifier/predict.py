"""
predict.py — Classifier Inference Module
Loads saved DeBERTa checkpoint.
Runs inference on new tickets.
Returns predicted label + confidence score (softmax probability).
"""

# TODO: implement

# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/classifier/predict.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from config.constants import (
    COL_TICKET_ID,
    COL_PRIORITY,
    COL_INFERRED_SEV,
    COL_MISMATCH_TYPE,
    COL_DELTA_ABS,
    COL_PREDICTION,
    COL_CONFIDENCE,
    COL_MODEL_INPUT,
    LABEL_CONSISTENT,
    LABEL_MISMATCH,
    LABEL_NAMES,
)
from src.classifier.dataset import (
    load_tokenizer,
    build_dataloader,
    SIADataset,
)
from src.classifier.evaluate import load_trained_model
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
)

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  BATCH INFERENCE
# ══════════════════════════════════════════════════════════════

def predict_batch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_confs = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )

            probs          = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
            mismatch_probs = probs[:, 1]

            # Use configurable threshold
            preds = (mismatch_probs >= threshold).astype(int)

            all_preds.extend(preds.tolist())
            all_confs.extend(mismatch_probs.tolist())

    return np.array(all_preds), np.array(all_confs)

# ══════════════════════════════════════════════════════════════
#  SINGLE TICKET INFERENCE
# ══════════════════════════════════════════════════════════════

def predict_single(
    model: nn.Module,
    tokenizer,
    model_input: str,
    device: torch.device,
    max_length: int = 512,
) -> Dict:
    """
    Runs inference on a single model input string.
    Used by the Streamlit app for single-ticket analysis.

    Args:
        model       : Trained DeBERTa model
        tokenizer   : Loaded tokenizer
        model_input : Formatted input string from build_model_input()
        device      : Target device
        max_length  : Max token length

    Returns:
        Dict with prediction, confidence, label
        {
            "prediction":   1,
            "label":        "Mismatch",
            "confidence":   0.923,
            "consistent_prob": 0.077,
            "mismatch_prob":   0.923,
        }
    """
    model.eval()

    encoding = tokenizer(
        model_input,
        max_length     = max_length,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

    probs      = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    prediction = int(outputs.logits.argmax(dim=-1).cpu().numpy()[0])
    confidence = float(probs[LABEL_MISMATCH])

    return {
        "prediction":      prediction,
        "label":           LABEL_NAMES[prediction],
        "confidence":      round(confidence,        4),
        "consistent_prob": round(float(probs[LABEL_CONSISTENT]), 4),
        "mismatch_prob":   round(float(probs[LABEL_MISMATCH]),   4),
    }


# ══════════════════════════════════════════════════════════════
#  FULL INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════

def predict_dataframe(
    df: pd.DataFrame,
    model: nn.Module         = None,
    tokenizer                = None,
    model_dir: str           = "outputs/models/deberta_classifier/",
    config_path: str         = "config/config.yaml",
    device: torch.device     = None,
) -> pd.DataFrame:
    """
    Runs inference on a full DataFrame.
    Adds Prediction and Confidence columns.

    Loads model from disk if not provided.

    Args:
        df          : DataFrame with Model_Input column
        model       : Pre-loaded model (optional)
        tokenizer   : Pre-loaded tokenizer (optional)
        model_dir   : Saved model directory
        config_path : Path to config.yaml
        device      : Target device

    Returns:
        DataFrame with Prediction and Confidence columns added
    """
    cfg = load_config(config_path)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ── Load model if not provided ────────────────────────────
    if model is None or tokenizer is None:
        log_step(logger, f"Loading model from {model_dir}")
        model, tokenizer = load_trained_model(model_dir, device)

    # ── Validate input ────────────────────────────────────────
    if COL_MODEL_INPUT not in df.columns:
        raise ValueError(
            f"Column '{COL_MODEL_INPUT}' not found. "
            f"Run feature_engineering_pipeline() first."
        )

    log_step(logger, f"Running inference on {len(df):,} tickets")

    # ── Build DataLoader ──────────────────────────────────────
    loader = build_dataloader(
        df         = df,
        tokenizer  = tokenizer,
        batch_size = cfg["classifier"]["eval_batch_size"],
        max_length = cfg["classifier"]["max_length"],
        shuffle    = False,
        is_test    = True,   # no labels required
    )

    # ── Run inference ─────────────────────────────────────────
    predictions, confidences = predict_batch(model, loader, device)

    # ── Add to DataFrame ──────────────────────────────────────
    df = df.copy()
    df[COL_PREDICTION] = predictions
    df[COL_CONFIDENCE] = confidences

    # Add human-readable label
    df["Prediction_Label"] = df[COL_PREDICTION].map(LABEL_NAMES)

    # Log summary
    n_mismatch  = int((predictions == LABEL_MISMATCH).sum())
    n_consistent = int((predictions == LABEL_CONSISTENT).sum())
    logger.info(
        f"Inference complete — "
        f"Mismatch={n_mismatch:,} ({n_mismatch/len(df):.1%}) | "
        f"Consistent={n_consistent:,} ({n_consistent/len(df):.1%})"
    )

    log_success(logger, f"Predictions added for {len(df):,} tickets")
    return df


# ══════════════════════════════════════════════════════════════
#  CSV INFERENCE (root predict.py calls this)
# ══════════════════════════════════════════════════════════════

def predict_from_csv(
    input_path: str,
    output_dir: str  = "outputs/dossiers/",
    model_dir: str   = "outputs/models/deberta_classifier/",
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    End-to-end inference from a raw CSV file.

    This is the function called by the root predict.py script.

    Pipeline:
        1. Load and preprocess the input CSV
        2. Build model input strings
        3. Run classifier inference
        4. Generate evidence dossiers for flagged tickets
        5. Save predictions CSV + dossiers JSON

    Args:
        input_path  : Path to input CSV file
        output_dir  : Directory to save outputs
        model_dir   : Saved model directory
        config_path : Path to config.yaml

    Returns:
        (predictions_df, dossiers_list)
    """
    from src.preprocessing.preprocess import preprocess_pipeline
    from src.preprocessing.feature_engineering import feature_engineering_pipeline
    from src.embeddings.generate_embeddings import embedding_pipeline
    from src.signals.signal1_semantic import compute_signal1
    from src.signals.signal2_resolution import compute_signal2
    from src.fusion.severity_fusion import fusion_pipeline
    from src.pseudo_labels.generate_labels import compute_severity_delta
    from src.dossier.generate_dossier import generate_all_dossiers

    logger.info(f"Starting CSV inference pipeline: {input_path}")

    ensure_dir(output_dir)

    # ── Step 1: Preprocess ────────────────────────────────────
    log_step(logger, "Step 1: Preprocessing")
    df = preprocess_pipeline(
        raw_path    = input_path,
        save_path   = None,   # don't overwrite training data
        config_path = config_path,
    )

    # ── Step 2: Feature engineering ───────────────────────────
    log_step(logger, "Step 2: Feature engineering")
    df = feature_engineering_pipeline(df)

    # ── Step 3: Embeddings + Signals ─────────────────────────
    log_step(logger, "Step 3: Generating embeddings and signals")
    ticket_embs, anchor_embs, emb_model = embedding_pipeline(
        df=df, config_path=config_path
    )

    _, df = compute_signal1(
        df=df, ticket_embs=ticket_embs,
        anchor_embs=anchor_embs, model=emb_model,
        config_path=config_path,
    )

    _, df = compute_signal2(df=df, config_path=config_path)

    # ── Step 4: Fusion ────────────────────────────────────────
    log_step(logger, "Step 4: Signal fusion")
    fused_scores, df, _ = fusion_pipeline(df=df, config_path=config_path)
    df = compute_severity_delta(df, fused_scores=fused_scores)

    # ── Step 5: Classifier inference ─────────────────────────
    log_step(logger, "Step 5: Classifier inference")
    df = predict_dataframe(
        df          = df,
        model_dir   = model_dir,
        config_path = config_path,
    )

    # ── Step 6: Generate dossiers ─────────────────────────────
    log_step(logger, "Step 6: Generating evidence dossiers")
    dossiers = generate_all_dossiers(
        df          = df,
        config_path = config_path,
    )

    # ── Save outputs ──────────────────────────────────────────
    pred_path    = str(Path(output_dir) / "predictions.csv")
    dossier_path = str(Path(output_dir) / "dossiers.json")

    df.to_csv(pred_path, index=False)
    save_json(dossiers, dossier_path)

    log_success(logger, f"Predictions saved → {pred_path}")
    log_success(logger, f"Dossiers saved    → {dossier_path}")
    log_success(logger, "CSV inference pipeline complete")

    return df, dossiers
