"""
inference_pipeline.py — Full Inference Pipeline (importable module)
Loads saved model + FAISS index.
Runs end-to-end prediction + dossier generation on new tickets.
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/pipeline/inference_pipeline.py
#
#  BOUNDARY RULE:
#  This file orchestrates inference stages in order.
#  It imports from src/ modules and calls them.
#  NO argument parsing here — that lives in root predict.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils.helpers import (
    load_config,
    set_seed,
    save_json,
    ensure_dir,
)
from src.utils.logger import (
    get_sia_logger,
    log_stage,
    log_step,
    log_success,
    log_warning,
)

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  RESOURCE LOADER
# ══════════════════════════════════════════════════════════════

class InferenceResources:
    """
    Holds all loaded resources for inference.
    Loaded once and reused across multiple inference calls.
    Avoids reloading model/index on every batch.

    Attributes:
        model      : Loaded DeBERTa classifier
        tokenizer  : Loaded tokenizer
        emb_model  : Loaded SentenceTransformer
        anchor_embs: Severity anchor embeddings
        searcher   : Loaded FAISSSearcher
        device     : Target device
    """

    def __init__(self):
        self.model       = None
        self.tokenizer   = None
        self.emb_model   = None
        self.anchor_embs = None
        self.searcher    = None
        self.device      = None
        self._loaded     = False

    def load(
        self,
        config_path: str = "config/config.yaml",
        device: torch.device = None,
    ) -> None:
        """
        Loads all inference resources from disk.

        Args:
            config_path : Path to config.yaml
            device      : Target device (auto-detected if None)
        """
        from src.classifier.evaluate import load_trained_model
        from src.embeddings.sentence_encoder import load_encoder, build_anchor_embeddings
        from src.retrieval.search import load_searcher

        cfg = load_config(config_path)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        log_step(logger, f"Loading inference resources on {self.device}")

        # Load DeBERTa classifier + tokenizer
        log_step(logger, "Loading DeBERTa classifier")
        self.model, self.tokenizer = load_trained_model(
            model_dir = cfg["classifier"]["save_dir"],
            device    = self.device,
        )

        # Load embedding model
        log_step(logger, "Loading sentence encoder")
        self.emb_model = load_encoder(cfg["embeddings"]["model_name"])

        # Build anchor embeddings
        log_step(logger, "Building anchor embeddings")
        self.anchor_embs = build_anchor_embeddings(model=self.emb_model)

        # Load FAISS searcher
        log_step(logger, "Loading FAISS searcher")
        try:
            self.searcher = load_searcher(config_path=config_path)
        except FileNotFoundError:
            log_warning(
                logger,
                "FAISS index not found — "
                "inference will run without semantic search evidence"
            )
            self.searcher = None

        self._loaded = True
        log_success(logger, "All inference resources loaded")

    def check_loaded(self) -> None:
        """Raises error if resources not loaded."""
        if not self._loaded:
            raise RuntimeError(
                "InferenceResources not loaded. "
                "Call resources.load() first."
            )


# ══════════════════════════════════════════════════════════════
#  SINGLE TICKET INFERENCE
# ══════════════════════════════════════════════════════════════

def infer_single_ticket(
    ticket: Dict,
    resources: InferenceResources,
    config_path: str = "config/config.yaml",
) -> Dict:
    """
    Runs full inference pipeline on a single ticket dict.

    Used by Streamlit app Page 1 (Single Ticket Analysis).

    Input ticket dict:
    {
        "Ticket_Subject":      "Cannot login to account",
        "Ticket_Description":  "All users affected since 6am...",
        "Priority_Level":      "Low",
        "Ticket_Channel":      "Email",
        "Issue_Category":      "Technical",
        "Resolution_Time_Hours": 72.0,
        "Customer_Email":      "user@enterprise.org",
        "Ticket_ID":           "TKT-99999",   # optional
    }

    Returns full result dict:
    {
        "ticket_id":          "TKT-99999",
        "prediction":         1,
        "label":              "Mismatch",
        "confidence":         0.934,
        "assigned_priority":  "Low",
        "inferred_severity":  "High",
        "mismatch_type":      "Hidden Crisis",
        "severity_delta":     "2.0 levels",
        "semantic_score":     3.21,
        "rt_score":           2.87,
        "fused_score":        3.10,
        "similar_tickets":    [...],
        "dossier":            {...},
    }

    Args:
        ticket      : Single ticket as dict
        resources   : Loaded InferenceResources
        config_path : Path to config.yaml

    Returns:
        Full inference result dict
    """
    from src.preprocessing.preprocess import (
        clean_text, encode_metadata, handle_missing
    )
    from src.preprocessing.feature_engineering import (
        build_model_input, get_category_bias
    )
    from src.embeddings.sentence_encoder import encode_single, compute_soft_severity_score
    from src.signals.signal2_resolution import score_by_continuous_percentile
    from src.classifier.predict import predict_single
    from src.dossier.generate_dossier import generate_single_dossier
    from src.utils.helpers import (
        derive_customer_tier,
        score_to_label,
        get_mismatch_type,
    )
    from config.constants import (
        COL_TICKET_ID,
        COL_PRIORITY,
        COL_RT,
        COL_INFERRED_SEV,
        COL_INFERRED_NUM,
        COL_DELTA,
        COL_DELTA_ABS,
        COL_MISMATCH_TYPE,
        COL_CONFIDENCE,
        WEIGHT_SEMANTIC,
        WEIGHT_RESOLUTION,
        PRIORITY_MAP,
    )

    resources.check_loaded()
    cfg = load_config(config_path)

    # ── Build Series from ticket dict ─────────────────────────
    row = pd.Series(ticket)

    # ── Derive customer tier ──────────────────────────────────
    email = ticket.get("Customer_Email", "unknown@example.com")
    tier  = derive_customer_tier(email)
    row["Customer_Tier"] = tier

    # ── Clean text ────────────────────────────────────────────
    from src.preprocessing.preprocess import clean_text
    subject = clean_text(str(ticket.get("Ticket_Subject",     "")))
    desc    = clean_text(str(ticket.get("Ticket_Description", "")))

    row["Ticket_Subject"]     = subject
    row["Ticket_Description"] = desc
    row["combined_text"]      = f"{subject} [SEP] {desc}"

    # ── Signal 1: Semantic severity ───────────────────────────
    ticket_emb   = encode_single(
        text      = row["combined_text"],
        model     = resources.emb_model,
        normalize = True,
    )
    sem_score_raw = compute_soft_severity_score(
        ticket_emb  = ticket_emb,
        anchor_embs = resources.anchor_embs,
    )

    # Apply category bias
    category   = ticket.get("Issue_Category", "General Inquiry")
    bias       = get_category_bias(category)
    bias_wt    = cfg["signal1"]["category_bias_weight"]
    sem_score  = (1 - bias_wt) * sem_score_raw + bias_wt * bias

    # ── Signal 2: Resolution time severity ────────────────────
    rt_val = float(ticket.get("Resolution_Time_Hours", 24.0))

    # Use dataset percentile anchors for single ticket scoring
    from config.constants import RT_PERCENTILES
    p25, p50, p75 = (
        RT_PERCENTILES["p25"],
        RT_PERCENTILES["p50"],
        RT_PERCENTILES["p75"],
    )
    if rt_val <= p25:
        rt_score = 1.0 + (rt_val / p25) * 1.0
    elif rt_val <= p50:
        rt_score = 2.0 + ((rt_val - p25) / (p50 - p25)) * 1.0
    elif rt_val <= p75:
        rt_score = 3.0 + ((rt_val - p50) / (p75 - p50)) * 1.0
    else:
        rt_score = 4.0

    rt_score = float(np.clip(rt_score, 1.0, 4.0))

    # ── Fusion ────────────────────────────────────────────────
    w_sem    = cfg["fusion"]["semantic_weight"]
    w_rt     = cfg["fusion"]["resolution_weight"]
    fused    = w_sem * sem_score + w_rt * rt_score
    fused    = float(np.clip(fused, 1.0, 4.0))

    inferred_label = score_to_label(fused)
    inferred_num   = PRIORITY_MAP[inferred_label]

    assigned_label = str(ticket.get("Priority_Level", "Medium"))
    assigned_num   = PRIORITY_MAP.get(assigned_label, 2)

    delta     = float(fused - assigned_num)
    delta_abs = abs(delta)

    mismatch_type = get_mismatch_type(delta)

    # ── Update row for dossier generation ────────────────────
    row[COL_PRIORITY]      = assigned_label
    row[COL_RT]            = rt_val
    row[COL_INFERRED_SEV]  = inferred_label
    row[COL_INFERRED_NUM]  = inferred_num
    row[COL_DELTA]         = delta
    row[COL_DELTA_ABS]     = delta_abs
    row[COL_MISMATCH_TYPE] = mismatch_type
    row["Priority_Numeric"] = assigned_num

    # ── Build model input for DeBERTa ─────────────────────────
    model_input = build_model_input(row)

    # ── Classifier prediction ─────────────────────────────────
    clf_result = predict_single(
        model       = resources.model,
        tokenizer   = resources.tokenizer,
        model_input = model_input,
        device      = resources.device,
        max_length  = cfg["classifier"]["max_length"],
    )

    prediction = clf_result["prediction"]
    confidence = clf_result["confidence"]
    row[COL_CONFIDENCE] = confidence

    # ── FAISS similar tickets ─────────────────────────────────
    similar_tickets = []
    if resources.searcher is not None:
        try:
            similar_tickets = resources.searcher.search(
                query_emb = ticket_emb,
                k         = cfg["faiss"]["top_k"],
            )
        except Exception as e:
            log_warning(logger, f"FAISS search failed: {e}")

    # ── Generate dossier if mismatch ──────────────────────────
    dossier = None
    if prediction == 1:
        dossier = generate_single_dossier(
            row             = row,
            row_idx         = 0,
            similar_tickets = similar_tickets,
            max_keywords    = cfg["dossier"]["max_keywords"],
        )

    # ── Build result ──────────────────────────────────────────
    result = {
        "ticket_id":         str(ticket.get(COL_TICKET_ID, "N/A")),
        "prediction":        prediction,
        "label":             clf_result["label"],
        "confidence":        round(confidence, 4),
        "consistent_prob":   clf_result["consistent_prob"],
        "mismatch_prob":     clf_result["mismatch_prob"],
        "assigned_priority": assigned_label,
        "inferred_severity": inferred_label,
        "mismatch_type":     mismatch_type,
        "severity_delta":    f"{delta_abs:.1f} levels",
        "semantic_score":    round(sem_score, 4),
        "rt_score":          round(rt_score,  4),
        "fused_score":       round(fused,     4),
        "similar_tickets":   similar_tickets,
        "dossier":           dossier,
    }

    return result


# ══════════════════════════════════════════════════════════════
#  BATCH INFERENCE
# ══════════════════════════════════════════════════════════════

def infer_batch(
    df: pd.DataFrame,
    resources: InferenceResources,
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Runs full inference pipeline on a DataFrame of tickets.

    Used by root predict.py for CSV batch inference.

    Steps:
        1. Preprocess DataFrame
        2. Feature engineering
        3. Generate embeddings
        4. Compute signals + fusion
        5. Run classifier
        6. Generate dossiers
        7. Return predictions + dossiers

    Args:
        df          : Raw tickets DataFrame
        resources   : Loaded InferenceResources
        config_path : Path to config.yaml

    Returns:
        (predictions_df, dossiers)
        predictions_df : DataFrame with Prediction + Confidence columns
        dossiers       : List of dossier dicts for flagged tickets
    """
    from src.preprocessing.preprocess import (
        handle_missing, clean_texts, merge_text, encode_metadata, drop_columns
    )
    from src.preprocessing.feature_engineering import feature_engineering_pipeline
    from src.embeddings.sentence_encoder import encode_texts, compute_soft_severity_batch
    from src.signals.signal1_semantic import apply_category_bias
    from src.signals.signal2_resolution import score_by_continuous_percentile
    from src.fusion.severity_fusion import weighted_fusion, scores_to_labels
    from src.pseudo_labels.generate_labels import compute_severity_delta
    from src.classifier.predict import predict_dataframe
    from src.dossier.generate_dossier import generate_all_dossiers, save_dossiers
    from src.dossier.verify_dossier import verify_all_dossiers
    from config.constants import (
        COL_INFERRED_SEV,
        COL_INFERRED_NUM,
        COL_MISMATCH_TYPE,
        PRIORITY_INV,
    )

    resources.check_loaded()
    cfg = load_config(config_path)

    logger.info(f"Running batch inference on {len(df):,} tickets")

    # ── Preprocess ────────────────────────────────────────────
    log_step(logger, "Preprocessing")
    df = handle_missing(df)
    df = clean_texts(df)
    df = merge_text(df)
    df = encode_metadata(df)
    df = drop_columns(df)
    df = feature_engineering_pipeline(df)

    # ── Embeddings ────────────────────────────────────────────
    log_step(logger, "Encoding tickets")
    ticket_embs = encode_texts(
        texts         = df["combined_text"].fillna("").tolist(),
        model         = resources.emb_model,
        batch_size    = cfg["embeddings"]["batch_size"],
        normalize     = True,
        show_progress = True,
    )

    # ── Signal 1 ──────────────────────────────────────────────
    log_step(logger, "Computing Signal 1 (Semantic)")
    from src.embeddings.sentence_encoder import compute_soft_severity_batch
    sem_scores = compute_soft_severity_batch(ticket_embs, resources.anchor_embs)
    sem_scores = apply_category_bias(
        semantic_scores = sem_scores,
        categories      = df["Issue_Category"],
        bias_weight     = cfg["signal1"]["category_bias_weight"],
    )

    # ── Signal 2 ──────────────────────────────────────────────
    log_step(logger, "Computing Signal 2 (Resolution Time)")
    rt_scores = score_by_continuous_percentile(
        rt_values  = df["Resolution_Time_Hours"].values,
        clip_lower = cfg["signal2"]["clip_lower"],
        clip_upper = cfg["signal2"]["clip_upper"],
    )

    # ── Fusion ────────────────────────────────────────────────
    log_step(logger, "Fusing signals")
    fused_scores = weighted_fusion(
        semantic_scores = sem_scores,
        rt_scores       = rt_scores,
        w_semantic      = cfg["fusion"]["semantic_weight"],
        w_resolution    = cfg["fusion"]["resolution_weight"],
    )

    inferred_numeric, inferred_labels = scores_to_labels(fused_scores)
    df[COL_INFERRED_SEV] = inferred_labels
    df[COL_INFERRED_NUM] = inferred_numeric
    df = compute_severity_delta(df, fused_scores=fused_scores)

    # ── Classifier ────────────────────────────────────────────
    log_step(logger, "Running classifier inference")
    df = predict_dataframe(
        df          = df,
        model       = resources.model,
        tokenizer   = resources.tokenizer,
        config_path = config_path,
        device      = resources.device,
    )

    # ── Dossiers ──────────────────────────────────────────────
    log_step(logger, "Generating evidence dossiers")
    dossiers = generate_all_dossiers(
        df          = df,
        searcher    = resources.searcher,
        ticket_embs = ticket_embs,
        config_path = config_path,
    )

    # Verify dossiers
    verified, rejected, report = verify_all_dossiers(
        dossiers    = dossiers,
        df          = df,
        config_path = config_path,
    )

    log_success(
        logger,
        f"Batch inference complete — "
        f"predictions={len(df):,} | "
        f"dossiers={len(verified):,}"
    )

    return df, verified


# ══════════════════════════════════════════════════════════════
#  MAIN INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════

def run_inference_pipeline(
    input_path: str,
    output_dir: str  = "outputs/dossiers/",
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Full inference pipeline entry point.

    Called by root predict.py — never directly by user.

    Steps:
        1. Load all resources (model, FAISS, encoder)
        2. Load input CSV
        3. Run batch inference
        4. Save predictions + dossiers

    Args:
        input_path  : Path to input CSV file
        output_dir  : Directory to save outputs
        config_path : Path to config.yaml

    Returns:
        (predictions_df, verified_dossiers)
    """
    cfg = load_config(config_path)
    set_seed(cfg["preprocessing"]["random_seed"])

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   SIA — Support Integrity Auditor        ║")
    logger.info("║   Inference Pipeline                     ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info(f"Input : {input_path}")
    logger.info(f"Output: {output_dir}")

    ensure_dir(output_dir)

    # ── Load resources ────────────────────────────────────────
    log_stage(logger, 1, "Loading Inference Resources")
    resources = InferenceResources()
    resources.load(config_path=config_path)

    # ── Load input CSV ────────────────────────────────────────
    log_stage(logger, 2, "Loading Input Data")
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} tickets from {input_path}")

    # ── Run batch inference ───────────────────────────────────
    log_stage(logger, 3, "Running Batch Inference")
    predictions_df, dossiers = infer_batch(
        df          = df,
        resources   = resources,
        config_path = config_path,
    )

    # ── Save outputs ──────────────────────────────────────────
    log_stage(logger, 4, "Saving Outputs")

    pred_path    = str(Path(output_dir) / "predictions.csv")
    dossier_path = str(Path(output_dir) / "dossiers.json")

    predictions_df.to_csv(pred_path, index=False)
    save_json(dossiers, dossier_path)

    log_success(logger, f"Predictions → {pred_path}")
    log_success(logger, f"Dossiers    → {dossier_path}")

    # ── Summary ───────────────────────────────────────────────
    n_mismatch = int(
        (predictions_df["Prediction"] == 1).sum()
    ) if "Prediction" in predictions_df.columns else 0

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   INFERENCE PIPELINE COMPLETE            ║")
    logger.info(f"║   Total tickets  : {len(predictions_df):<22}║")
    logger.info(f"║   Mismatches     : {n_mismatch:<22}║")
    logger.info(f"║   Dossiers       : {len(dossiers):<22}║")
    logger.info("╚══════════════════════════════════════════╝")

    return predictions_df, dossiers