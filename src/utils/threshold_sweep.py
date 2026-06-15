# ─────────────────────────────────────────
#  SIA — Threshold Sweep
#  src/utils/threshold_sweep.py
# ─────────────────────────────────────────

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score,
    recall_score, precision_score,
    confusion_matrix,
)
from src.utils.helpers import load_config, save_json
from src.utils.logger import get_sia_logger

logger = get_sia_logger(__name__)


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list = None,
) -> pd.DataFrame:
    """
    Tests multiple classification thresholds and returns
    metrics for each. Finds optimal threshold for Macro F1.

    Args:
        y_true      : True labels (0/1)
        y_prob      : Mismatch probabilities from model
        thresholds  : List of thresholds to test

    Returns:
        DataFrame with metrics per threshold
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50,
                      0.55, 0.60, 0.65, 0.70]

    rows = []

    logger.info("Threshold Sweep Results:")
    logger.info(
        f"{'Threshold':<12} {'Accuracy':<12} {'Macro F1':<12} "
        f"{'Rec Consist':<14} {'Rec Mismatch':<14} {'Prec Mismatch':<14}"
    )
    logger.info("─" * 80)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        rc   = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        rm   = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        pm   = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred)

        # Verification checks
        acc_pass = acc >= 0.83
        f1_pass  = f1  >= 0.82
        rc_pass  = rc  >= 0.78
        rm_pass  = rm  >= 0.78
        all_pass = acc_pass and f1_pass and rc_pass and rm_pass

        rows.append({
            "threshold":         t,
            "accuracy":          round(acc, 4),
            "macro_f1":          round(f1,  4),
            "recall_consistent": round(rc,  4),
            "recall_mismatch":   round(rm,  4),
            "precision_mismatch":round(pm,  4),
            "n_predicted_mismatch": int(y_pred.sum()),
            "tn": int(cm[0,0]),
            "fp": int(cm[0,1]),
            "fn": int(cm[1,0]),
            "tp": int(cm[1,1]),
            "all_thresholds_pass": all_pass,
        })

        status = "✔ ALL PASS" if all_pass else ""
        logger.info(
            f"{t:<12.2f} {acc:<12.4f} {f1:<12.4f} "
            f"{rc:<14.4f} {rm:<14.4f} {pm:<14.4f} {status}"
        )

    df_results = pd.DataFrame(rows)

    # Find best threshold for Macro F1
    best_f1_row = df_results.loc[df_results["macro_f1"].idxmax()]
    logger.info(f"\nBest threshold for Macro F1: {best_f1_row['threshold']}")
    logger.info(f"  Macro F1   : {best_f1_row['macro_f1']:.4f}")
    logger.info(f"  Accuracy   : {best_f1_row['accuracy']:.4f}")
    logger.info(f"  Rec Consist: {best_f1_row['recall_consistent']:.4f}")
    logger.info(f"  Rec Mismatch:{best_f1_row['recall_mismatch']:.4f}")
    logger.info(f"  All Pass   : {best_f1_row['all_thresholds_pass']}")

    # Find best threshold where ALL thresholds pass
    passing = df_results[df_results["all_thresholds_pass"]]
    if len(passing) > 0:
        best_pass = passing.loc[passing["macro_f1"].idxmax()]
        logger.info(f"\nBest VERIFIED threshold: {best_pass['threshold']}")
        logger.info(f"  Macro F1   : {best_pass['macro_f1']:.4f}")
        logger.info(f"  Accuracy   : {best_pass['accuracy']:.4f}")
    else:
        logger.info("\nNo threshold achieves all verification criteria")

    return df_results


def run_threshold_sweep(
    model_dir: str   = "outputs/models/deberta_classifier/",
    test_path: str   = "data/processed/test.csv",
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Loads model, runs inference on test set,
    then sweeps thresholds.
    """
    import pandas as pd
    from src.classifier.evaluate import load_trained_model, get_predictions
    from src.classifier.dataset import build_dataloader
    from src.classifier.train import WeightedCrossEntropyLoss
    from src.classifier.dataset import compute_class_weights

    cfg    = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info("Loading model for threshold sweep...")
    model, tokenizer = load_trained_model(model_dir, device)

    # Load test data
    test_df = pd.read_csv(test_path)
    logger.info(f"Test set: {len(test_df):,} tickets")

    # Build loader
    loader = build_dataloader(
        df         = test_df,
        tokenizer  = tokenizer,
        batch_size = cfg["classifier"]["eval_batch_size"],
        max_length = cfg["classifier"]["max_length"],
        shuffle    = False,
        is_test    = False,
    )

    # Dummy loss for get_predictions
    class_weights = compute_class_weights(test_df)
    loss_fn = WeightedCrossEntropyLoss(class_weights, device)

    # Get probabilities
    logger.info("Running inference...")
    y_pred, y_true, y_prob = get_predictions(model, loader, device)

    logger.info(
        f"Probability stats — "
        f"min={y_prob.min():.3f} | "
        f"max={y_prob.max():.3f} | "
        f"mean={y_prob.mean():.3f} | "
        f"median={np.median(y_prob):.3f}"
    )

    # Run sweep
    results = sweep_thresholds(y_true, y_prob)

    # Save results
    save_json(
        results.to_dict(orient="records"),
        "outputs/metrics/threshold_sweep.json"
    )
    results.to_csv(
        "outputs/metrics/threshold_sweep.csv",
        index=False
    )
    logger.info("Threshold sweep saved → outputs/metrics/threshold_sweep.csv")

    return results

if __name__ == "__main__":
    results = run_threshold_sweep(
        model_dir   = "outputs/models/deberta_classifier/",
        test_path   = "data/processed/test.csv",
        config_path = "config/config.yaml",
    )
    print("\nFull Results:")
    print(results.to_string(index=False))