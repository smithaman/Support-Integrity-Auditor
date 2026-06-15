"""
train_pipeline.py — Full Training Pipeline (importable module)
Orchestrates: preprocess → embeddings → signals → fusion
→ pseudo_labels → classifier training → FAISS index build.
"""

# TODO: implement

# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/pipeline/train_pipeline.py
#
#  BOUNDARY RULE:
#  This file orchestrates all stages in order.
#  It imports from src/ modules and calls them.
#  NO argument parsing here — that lives in root train_pipeline.py
# ─────────────────────────────────────────

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils.helpers import (
    load_config,
    set_seed,
    save_json,
    ensure_dir,
    check_mismatch_rate,
)
from src.utils.logger import (
    get_sia_logger,
    log_stage,
    log_step,
    log_success,
    log_warning,
    log_metrics,
)
from src.utils.metrics import (
    build_full_metrics_report,
    build_ablation_table,
)

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  STAGE RUNNERS
# ══════════════════════════════════════════════════════════════

def run_preprocessing(cfg: Dict) -> pd.DataFrame:
    """
    Stage 0 — Data Preprocessing
    Loads raw CSV, cleans text, encodes metadata.
    """
    from src.preprocessing.preprocess import preprocess_pipeline

    log_stage(logger, 0, "Data Preprocessing")

    df = preprocess_pipeline(
        raw_path    = cfg["data"]["raw_path"],
        save_path   = cfg["data"]["processed_path"],
        config_path = "config/config.yaml",
    )

    log_success(logger, f"Preprocessing complete — {len(df):,} tickets")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 0b — Feature Engineering
    Builds DeBERTa model inputs and NLP features.
    """
    from src.preprocessing.feature_engineering import feature_engineering_pipeline

    log_stage(logger, 0, "Feature Engineering")

    df = feature_engineering_pipeline(df)

    log_success(logger, "Feature engineering complete")
    return df


def run_embeddings(
    df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[np.ndarray, Dict, object]:
    """
    Stage 0c — Embedding Generation
    Encodes all tickets and builds anchor embeddings.
    """
    from src.embeddings.generate_embeddings import embedding_pipeline

    log_stage(logger, 0, "Embedding Generation")

    ticket_embs, anchor_embs, model = embedding_pipeline(
        df          = df,
        config_path = "config/config.yaml",
    )

    log_success(
        logger,
        f"Embeddings complete — "
        f"shape={ticket_embs.shape}"
    )
    return ticket_embs, anchor_embs, model


def run_signal1(
    df: pd.DataFrame,
    ticket_embs: np.ndarray,
    anchor_embs: Dict,
    emb_model: object,
    cfg: Dict,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Stage 1a — Signal 1: Semantic Severity
    Computes embedding-based severity scores.
    """
    from src.signals.signal1_semantic import compute_signal1

    log_stage(logger, 1, "Signal 1 — Semantic Severity")

    scores, df = compute_signal1(
        df                   = df,
        ticket_embs          = ticket_embs,
        anchor_embs          = anchor_embs,
        model                = emb_model,
        use_category_bias    = cfg["signal1"]["use_category_bias"],
        category_bias_weight = cfg["signal1"]["category_bias_weight"],
        config_path          = "config/config.yaml",
    )

    log_success(
        logger,
        f"Signal 1 complete — "
        f"mean={scores.mean():.3f} | "
        f"std={scores.std():.3f}"
    )
    return scores, df


def run_signal2(
    df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Stage 1b — Signal 2: Resolution Time Severity
    Computes RT-based severity scores.
    """
    from src.signals.signal2_resolution import compute_signal2

    log_stage(logger, 1, "Signal 2 — Resolution Time Severity")

    scores, df = compute_signal2(
        df          = df,
        method      = cfg["signal2"]["method"],
        clip_lower  = cfg["signal2"]["clip_lower"],
        clip_upper  = cfg["signal2"]["clip_upper"],
        config_path = "config/config.yaml",
    )

    log_success(
        logger,
        f"Signal 2 complete — "
        f"mean={scores.mean():.3f} | "
        f"std={scores.std():.3f}"
    )
    return scores, df


def run_ablation(
    sem_scores: np.ndarray,
    rt_scores: np.ndarray,
    df: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    """
    Ablation — Tests all fusion configurations.
    Results go into README ablation table.
    """
    from src.fusion.severity_fusion import run_ablation as _run_ablation

    log_step(logger, "Running fusion ablation experiments")

    assigned_numeric = df["Priority_Numeric"].values.astype(float)

    ablation_results = _run_ablation(
        semantic_scores  = sem_scores,
        rt_scores        = rt_scores,
        assigned_numeric = assigned_numeric,
        threshold        = cfg["fusion"]["mismatch_threshold"],
    )

    build_ablation_table(
        ablation_results = ablation_results,
        save_path        = "outputs/metrics/ablation_table.json",
    )

    return ablation_results


def run_fusion(
    df: pd.DataFrame,
    sem_scores: np.ndarray,
    rt_scores: np.ndarray,
    cfg: Dict,
) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """
    Stage 1c — Signal Fusion
    Combines signals into inferred severity.
    """
    from src.fusion.severity_fusion import fusion_pipeline

    log_stage(logger, 1, "Signal Fusion")

    fused_scores, df, agreement = fusion_pipeline(
        df              = df,
        semantic_scores = sem_scores,
        rt_scores       = rt_scores,
        config_path     = "config/config.yaml",
    )

    log_success(
        logger,
        f"Fusion complete — "
        f"mean={fused_scores.mean():.3f}"
    )
    return fused_scores, df, agreement


def run_pseudo_labels(
    df: pd.DataFrame,
    fused_scores: np.ndarray,
    cfg: Dict,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Stage 1d — Pseudo-Label Generation
    Generates binary mismatch labels.
    """
    from src.pseudo_labels.generate_labels import pseudo_label_pipeline

    log_stage(logger, 1, "Pseudo-Label Generation")

    df, pl_stats = pseudo_label_pipeline(
        df           = df,
        fused_scores = fused_scores,
        config_path  = "config/config.yaml",
    )

    log_success(
        logger,
        f"Pseudo-labels complete — "
        f"mismatch_rate={pl_stats['mismatch_rate']:.1%}"
    )
    return df, pl_stats


def run_split(
    df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits pseudo-labeled data into train/val/test.
    Stratified on Mismatch_Label.
    """
    from src.preprocessing.preprocess import split_data
    from config.constants import COL_MISMATCH_LABEL

    log_step(logger, "Splitting data into train/val/test")

    train_df, val_df, test_df = split_data(
        df            = df,
        test_size     = cfg["preprocessing"]["test_size"],
        val_size      = cfg["preprocessing"]["val_size"],
        seed          = cfg["preprocessing"]["random_seed"],
        stratify_col  = COL_MISMATCH_LABEL,
    )

    # Save splits
    ensure_dir("data/processed")
    train_df.to_csv(cfg["data"]["train_path"], index=False)
    val_df.to_csv(cfg["data"]["val_path"],     index=False)
    test_df.to_csv(cfg["data"]["test_path"],   index=False)

    log_success(
        logger,
        f"Splits saved — "
        f"train={len(train_df):,} | "
        f"val={len(val_df):,} | "
        f"test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def run_classifier_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[object, Dict]:
    """
    Stage 2 — DeBERTa-v3-small Classifier Training
    Applies oversampling to training set before training.
    Val and test sets are NEVER oversampled — real distribution only.
    """
    from src.classifier.train import train_classifier
    from src.classifier.dataset import apply_smote

    log_stage(logger, 2, "Classifier Training")

    # ── Log original distribution ─────────────────────────────
    from config.constants import COL_MISMATCH_LABEL
    orig_rate = train_df[COL_MISMATCH_LABEL].mean()
    logger.info(
        f"Original train distribution — "
        f"total={len(train_df):,} | "
        f"mismatch_rate={orig_rate:.1%}"
    )

    # ── Apply oversampling to training set ONLY ───────────────
    log_step(logger, "Applying oversampling to training set")
    train_df_balanced = apply_smote(
        df        = train_df,
        label_col = COL_MISMATCH_LABEL,
        seed      = cfg["preprocessing"]["random_seed"],
    )

    new_rate = train_df_balanced[COL_MISMATCH_LABEL].mean()
    logger.info(
        f"Balanced train distribution — "
        f"total={len(train_df_balanced):,} | "
        f"mismatch_rate={new_rate:.1%}"
    )

    # ── Train on balanced, evaluate on real distribution ──────
    model, test_metrics = train_classifier(
        train_df    = train_df_balanced,   # ← balanced
        val_df      = val_df,              # ← original (real distribution)
        test_df     = test_df,             # ← original (real distribution)
        config_path = "config/config.yaml",
    )

    log_success(
        logger,
        f"Training complete — "
        f"test_f1={test_metrics.get('macro_f1', 0):.4f} | "
        f"test_acc={test_metrics.get('accuracy', 0):.4f}"
    )
    return model, test_metrics


def run_full_evaluation(
    test_df: pd.DataFrame,
    sem_scores: np.ndarray,
    rt_scores: np.ndarray,
    df_full: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    """
    Stage 5 — Full Evaluation
    Computes all metrics and checks verification thresholds.
    """
    from src.classifier.evaluate import evaluation_pipeline

    log_stage(logger, 5, "Evaluation")

    results = evaluation_pipeline(
        test_df     = test_df,
        model_dir   = cfg["classifier"]["save_dir"],
        config_path = "config/config.yaml",
    )

    return results


def run_faiss_index(
    df: pd.DataFrame,
    ticket_embs: np.ndarray,
    cfg: Dict,
) -> object:
    """
    Stage 3 — FAISS Index Build
    Builds vector index for semantic search.
    """
    from src.retrieval.build_index import build_faiss_pipeline

    log_stage(logger, 3, "FAISS Index Build")

    index = build_faiss_pipeline(
        df          = df,
        embeddings  = ticket_embs,
        config_path = "config/config.yaml",
    )

    log_success(
        logger,
        f"FAISS index built — "
        f"{index.ntotal:,} vectors"
    )
    return index


def run_dossier_generation(
    df: pd.DataFrame,
    ticket_embs: np.ndarray,
    cfg: Dict,
) -> list:
    """
    Stage 4 — Evidence Dossier Generation
    Generates and verifies dossiers for flagged tickets.
    """
    from src.retrieval.search import load_searcher
    from src.dossier.generate_dossier import (
        generate_single_dossier,
        save_dossiers,
    )
    from src.dossier.verify_dossier import (
        verify_all_dossiers,
        save_verification_report,
    )
    from config.constants import COL_PREDICTION, LABEL_MISMATCH

    log_stage(logger, 4, "Evidence Dossier Generation")

    # Load FAISS searcher
    try:
        searcher = load_searcher(config_path="config/config.yaml")
    except Exception as e:
        log_warning(logger, f"FAISS searcher failed to load: {e} — running without FAISS")
        searcher = None

    # Filter mismatch predictions
    pred_col = COL_PREDICTION if COL_PREDICTION in df.columns else "Mismatch_Label"
    flagged  = df[df[pred_col] == LABEL_MISMATCH].copy()

    logger.info(
        f"Generating dossiers for "
        f"{len(flagged):,} flagged tickets"
    )

    top_k = cfg["faiss"]["top_k"]

    dossiers = []
    for i, (df_idx, row) in enumerate(flagged.iterrows()):

        # FAISS search
        similar_tickets = []
        if searcher is not None and ticket_embs is not None:
            try:
                query_emb       = ticket_embs[df_idx]
                similar_tickets = searcher.search(
                    query_emb   = query_emb,
                    k           = top_k,
                    exclude_idx = df_idx,
                )
            except Exception as e:
                log_warning(logger, f"FAISS search failed for {df_idx}: {e}")

        # Generate dossier
        dossier = generate_single_dossier(
            row             = row,
            row_idx         = df_idx,
            similar_tickets = similar_tickets,
            max_keywords    = cfg["dossier"]["max_keywords"],
        )
        dossiers.append(dossier)

        if (i + 1) % 500 == 0:
            logger.info(f"   Generated {i+1:,}/{len(flagged):,} dossiers")

    # Verify dossiers
    verified, rejected, report = verify_all_dossiers(
        dossiers    = dossiers,
        df          = df,
        config_path = "config/config.yaml",
    )

    # Save
    save_dossiers(verified, cfg["dossier"]["output_path"])
    save_verification_report(
        verified   = verified,
        rejected   = rejected,
        report     = report,
        output_dir = "outputs/dossiers/",
    )

    log_success(
        logger,
        f"Dossiers complete — "
        f"verified={len(verified):,} | "
        f"rejected={len(rejected):,}"
    )
    return verified


# ══════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def run_training_pipeline(
    config_path: str         = "config/config.yaml",
    skip_training: bool      = False,
    skip_dossiers: bool      = False,
    run_adversarial: bool    = True,
) -> Dict:
    """
    Full SIA training pipeline orchestrator.

    Called by root train_pipeline.py — never directly by user.

    Stages:
        0  : Preprocessing + Feature Engineering + Embeddings
        1  : Signal 1 + Signal 2 + Fusion + Pseudo-Labels + Split
        2  : DeBERTa Classifier Training
        3  : FAISS Index Build
        4  : Evidence Dossier Generation + Verification
        5  : Evaluation + Metrics Report
        6  : Adversarial Testing (optional)

    Args:
        config_path      : Path to config.yaml
        skip_training    : Skip classifier training (use saved model)
        skip_dossiers    : Skip dossier generation
        run_adversarial  : Run adversarial robustness tests

    Returns:
        Dict with final metrics and pipeline summary
    """
    cfg = load_config(config_path)
    set_seed(cfg["preprocessing"]["random_seed"])

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   SIA — Support Integrity Auditor        ║")
    logger.info("║   Training Pipeline                      ║")
    logger.info("╚══════════════════════════════════════════╝")

    pipeline_results = {}

    # ── Stage 0: Preprocessing ────────────────────────────────
    df = run_preprocessing(cfg)
    df = run_feature_engineering(df)
    ticket_embs, anchor_embs, emb_model = run_embeddings(df, cfg)

    # ── Stage 1: Pseudo-Label Generation ─────────────────────
    sem_scores, df = run_signal1(df, ticket_embs, anchor_embs, emb_model, cfg)
    rt_scores,  df = run_signal2(df, cfg)

    # Ablation table
    ablation = run_ablation(sem_scores, rt_scores, df, cfg)
    pipeline_results["ablation"] = ablation

    # Fusion → Pseudo-labels → Split
    fused_scores, df, agreement = run_fusion(df, sem_scores, rt_scores, cfg)
    df, pl_stats                = run_pseudo_labels(df, fused_scores, cfg)
    train_df, val_df, test_df   = run_split(df, cfg)

    pipeline_results["pseudo_label_stats"]  = pl_stats
    pipeline_results["signal_agreement"]    = agreement

    # ── Stage 2: Classifier Training ─────────────────────────
    if not skip_training:
        model, test_metrics = run_classifier_training(
            train_df, val_df, test_df, cfg
        )
        pipeline_results["test_metrics"] = test_metrics
    else:
        log_warning(logger, "Skipping classifier training (skip_training=True)")
        model = None

    # ── Stage 3: FAISS Index ──────────────────────────────────
    index = run_faiss_index(df, ticket_embs, cfg)

    # ── Stage 4: Dossier Generation ───────────────────────────
    if not skip_dossiers:
        # Run inference on full dataset for dossiers
        from src.classifier.predict import predict_dataframe
        from src.classifier.evaluate import load_trained_model

        trained_model, tokenizer = load_trained_model(
            model_dir = cfg["classifier"]["save_dir"]
        )
        df = predict_dataframe(
            df          = df,
            model       = trained_model,
            tokenizer   = tokenizer,
            config_path = config_path,
        )
        dossiers = run_dossier_generation(df, ticket_embs, cfg)
        pipeline_results["n_dossiers"] = len(dossiers)
    else:
        log_warning(logger, "Skipping dossier generation (skip_dossiers=True)")

    # ── Stage 5: Full Evaluation ──────────────────────────────
    eval_results = run_full_evaluation(
        test_df     = test_df,
        sem_scores  = sem_scores[:len(test_df)],
        rt_scores   = rt_scores[:len(test_df)],
        df_full     = df,
        cfg         = cfg,
    )
    pipeline_results["evaluation"] = eval_results

    # ── Stage 6: Adversarial Testing ─────────────────────────
    if run_adversarial and not skip_training:
        from src.adversarial.adversarial_tests import run_adversarial_tests
        from src.classifier.evaluate import load_trained_model

        log_stage(logger, 6, "Adversarial Robustness Testing")

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        adv_model, adv_tokenizer = load_trained_model(
            model_dir = cfg["classifier"]["save_dir"],
            device    = device,
        )

        adv_score, bonus_earned, adv_report = run_adversarial_tests(
            model       = adv_model,
            tokenizer   = adv_tokenizer,
            config_path = config_path,
            device      = device,
        )

        pipeline_results["adversarial"] = {
            "score":        adv_score,
            "bonus_earned": bonus_earned,
        }

    # ── Save pipeline summary ─────────────────────────────────
    ensure_dir("outputs/metrics")
    save_json(
        pipeline_results,
        "outputs/metrics/pipeline_summary.json"
    )

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   TRAINING PIPELINE COMPLETE             ║")
    logger.info("╚══════════════════════════════════════════╝")

    # ── Get metrics from evaluation results ───────────────────
    # Pull from evaluation pipeline results (most reliable source)
    eval_clf = pipeline_results.get("evaluation", {}).get("metrics", {})
    test_met  = pipeline_results.get("test_metrics", {})

    # Merge — prefer evaluation pipeline results
    final_metrics = {**test_met, **eval_clf}

    log_metrics(
        logger,
        {k: v for k, v in final_metrics.items() if isinstance(v, float)},
        title = "Final Test Metrics"
    )

    pipeline_results["final_metrics"] = final_metrics
    return pipeline_results
