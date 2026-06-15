"""
evaluate.py — Standalone Evaluation Script

Runs evaluation on the held-out test set and prints:
  - Binary Classification Accuracy
  - Macro F1 Score
  - Per-Class Recall (Consistent + Mismatch)
  - Confusion Matrix
  - Pass / Fail verdict per verification threshold

Usage:
  python evaluate.py
  python evaluate.py --split test
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  evaluate.py  (ROOT ENTRY POINT)
#
#  BOUNDARY RULE:
#  This file ONLY handles:
#    1. Argument parsing
#    2. Loading config
#    3. Validating paths
#    4. Calling src/utils/metrics.py
#  ALL logic lives in src/
# ─────────────────────────────────────────

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Usage:
        python evaluate.py
        python evaluate.py --split test
        python evaluate.py --split val
        python evaluate.py --model-dir outputs/models/deberta_classifier/
        python evaluate.py --adversarial
    """
    parser = argparse.ArgumentParser(
        prog        = "evaluate.py",
        description = "SIA — Support Integrity Auditor Evaluation",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python evaluate.py
  python evaluate.py --split test
  python evaluate.py --split val --model-dir outputs/models/deberta_classifier/
  python evaluate.py --adversarial
  python evaluate.py --split test --adversarial --save-predictions
        """,
    )

    parser.add_argument(
        "--split",
        type    = str,
        default = "test",
        choices = ["train", "val", "test"],
        help    = "Dataset split to evaluate on (default: test)",
    )

    parser.add_argument(
        "--config",
        type    = str,
        default = "config/config.yaml",
        help    = "Path to config.yaml (default: config/config.yaml)",
    )

    parser.add_argument(
        "--model-dir",
        type    = str,
        default = None,
        help    = (
            "Override saved model directory. "
            "Defaults to value in config.yaml."
        ),
    )

    parser.add_argument(
        "--adversarial",
        action  = "store_true",
        default = False,
        help    = "Run adversarial robustness tests after evaluation",
    )

    parser.add_argument(
        "--save-predictions",
        action  = "store_true",
        default = False,
        help    = (
            "Save predictions CSV alongside metrics. "
            "Saved to outputs/metrics/<split>_predictions.csv"
        ),
    )

    parser.add_argument(
        "--threshold",
        type    = float,
        default = 0.5,
        help    = (
            "Classification threshold for mismatch prediction. "
            "Default: 0.5. Lower = more mismatches flagged."
        ),
    )

    return parser.parse_args()


def validate_config(config_path: str) -> dict:
    """
    Loads and validates config.yaml.

    Args:
        config_path : Path to config.yaml

    Returns:
        Loaded config dict
    """
    from src.utils.helpers import load_config

    if not Path(config_path).exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)

    required_sections = [
        "data", "classifier", "evaluation",
    ]

    missing = [s for s in required_sections if s not in cfg]
    if missing:
        print(f"[ERROR] Missing config sections: {missing}")
        sys.exit(1)

    return cfg


def validate_model_exists(cfg: dict, model_dir_override: str = None) -> str:
    """
    Validates that the trained model directory exists.

    Args:
        cfg                : Loaded config dict
        model_dir_override : CLI override for model directory

    Returns:
        Validated model directory path
    """
    model_dir = model_dir_override or cfg["classifier"]["save_dir"]

    if not Path(model_dir).exists():
        print(
            f"[ERROR] Trained model not found: {model_dir}\n\n"
            f"Run training first:\n"
            f"  python train_pipeline.py\n"
        )
        sys.exit(1)

    return model_dir


def validate_split_exists(cfg: dict, split: str) -> str:
    """
    Validates that the split CSV file exists.

    Args:
        cfg   : Loaded config dict
        split : Split name (train/val/test)

    Returns:
        Path to split CSV
    """
    split_path_map = {
        "train": cfg["data"]["train_path"],
        "val":   cfg["data"]["val_path"],
        "test":  cfg["data"]["test_path"],
    }

    split_path = split_path_map[split]

    if not Path(split_path).exists():
        print(
            f"[ERROR] Split file not found: {split_path}\n\n"
            f"Run training pipeline first to generate splits:\n"
            f"  python train_pipeline.py\n"
        )
        sys.exit(1)

    return split_path


def print_banner(args: argparse.Namespace) -> None:
    """Prints startup banner."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      SIA — Support Integrity Auditor             ║")
    print("║      Evaluation                                  ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Split        : {args.split:<33}║")
    print(f"║  Config       : {args.config:<33}║")
    print(f"║  Threshold    : {args.threshold:<33}║")
    print(f"║  Adversarial  : {str(args.adversarial):<33}║")
    print(f"║  Save preds   : {str(args.save_predictions):<33}║")
    print("╚══════════════════════════════════════════════════╝")
    print()


def print_metrics_table(metrics: dict) -> None:
    """
    Prints a clean metrics summary table.

    Args:
        metrics : Dict from compute_classification_metrics()
    """
    clf = metrics.get("classification", {})

    print()
    print("┌─────────────────────────────────────────────────┐")
    print("│                EVALUATION METRICS               │")
    print("├─────────────────────────────────────────────────┤")
    print(f"│  Accuracy             : {clf.get('accuracy', 0):.4f}                 │")
    print(f"│  Macro F1             : {clf.get('macro_f1', 0):.4f}                 │")
    print(f"│  Recall (Consistent)  : {clf.get('recall_consistent', 0):.4f}                 │")
    print(f"│  Recall (Mismatch)    : {clf.get('recall_mismatch', 0):.4f}                 │")
    print(f"│  Precision (Consistent): {clf.get('precision_consistent', 0):.4f}                │")
    print(f"│  Precision (Mismatch) : {clf.get('precision_mismatch', 0):.4f}                 │")
    print(f"│  ROC AUC              : {clf.get('roc_auc') or 'N/A':<23}  │")
    print(f"│  Cohen Kappa          : {clf.get('cohen_kappa') or 'N/A':<23}  │")
    print("├─────────────────────────────────────────────────┤")

    # Confusion matrix
    cm = clf.get("confusion_matrix", {})
    if cm:
        print(f"│  True Consistent (TN) : {cm.get('tn', 0):<23}  │")
        print(f"│  False Mismatch  (FP) : {cm.get('fp', 0):<23}  │")
        print(f"│  False Consistent(FN) : {cm.get('fn', 0):<23}  │")
        print(f"│  True Mismatch   (TP) : {cm.get('tp', 0):<23}  │")

    print("└─────────────────────────────────────────────────┘")
    print()


def print_threshold_results(thresholds: dict) -> None:
    """
    Prints verification threshold pass/fail results.

    Args:
        thresholds : Dict from check_verification_thresholds()
    """
    checks = {
        "Accuracy >= 83%":          thresholds.get("accuracy_pass",           False),
        "Macro F1 >= 0.82":         thresholds.get("macro_f1_pass",           False),
        "Recall Consistent >= 78%": thresholds.get("recall_consistent_pass",  False),
        "Recall Mismatch >= 78%":   thresholds.get("recall_mismatch_pass",    False),
    }

    all_passed = thresholds.get("all_passed", False)

    print("┌─────────────────────────────────────────────────┐")
    print("│           VERIFICATION THRESHOLDS               │")
    print("├─────────────────────────────────────────────────┤")

    for check, passed in checks.items():
        status = "✔ PASS" if passed else "✘ FAIL"
        print(f"│  {check:<30} {status:<15} │")

    print("├─────────────────────────────────────────────────┤")

    if all_passed:
        print("│  VERDICT: ✔ SUBMISSION VERIFIED               │")
    else:
        failing = thresholds.get("failing_metrics", [])
        print(f"│  VERDICT: ✘ NOT VERIFIED                      │")
        print(f"│  Failing: {str(failing):<39}│")

    print("└─────────────────────────────────────────────────┘")
    print()


def print_adversarial_results(adv_report: dict) -> None:
    """
    Prints adversarial test results.

    Args:
        adv_report : Dict from run_adversarial_tests()
    """
    score   = adv_report.get("score",        0)
    total   = adv_report.get("total",        10)
    bonus   = adv_report.get("bonus_earned", False)
    results = adv_report.get("per_ticket_results", [])

    print("┌─────────────────────────────────────────────────┐")
    print("│           ADVERSARIAL TEST RESULTS              │")
    print("├─────────────────────────────────────────────────┤")
    print(f"│  Score   : {score}/{total}                               │")
    print(f"│  Bonus   : {'✔ EARNED (+10%)' if bonus else '✘ NOT EARNED':<35} │")
    print("├─────────────────────────────────────────────────┤")

    for r in results:
        status = "✔" if r.get("correct") else "✘"
        tid    = r.get("ticket_id", "")
        exp    = r.get("expected_label",   "")[:10]
        pred   = r.get("predicted_label",  "")[:10]
        conf   = r.get("confidence",        0)
        print(
            f"│  {status} {tid:<8} "
            f"exp={exp:<12} "
            f"pred={pred:<12} "
            f"conf={conf:.3f}  │"
        )

    print("└─────────────────────────────────────────────────┘")
    print()


def main() -> None:
    """
    Root entry point for SIA evaluation.

    Flow:
        1. Parse CLI arguments
        2. Load + validate config
        3. Validate model + split paths
        4. Load split DataFrame
        5. Call evaluation pipeline
        6. Optionally run adversarial tests
        7. Print full results
    """
    import pandas as pd
    import numpy as np

    # ── Step 1: Parse arguments ───────────────────────────────
    args = parse_args()

    # ── Step 2: Load + validate config ───────────────────────
    cfg = validate_config(args.config)

    # Override model dir if provided
    if args.model_dir:
        cfg["classifier"]["save_dir"] = args.model_dir

    # ── Step 3: Validate paths ────────────────────────────────
    model_dir  = validate_model_exists(cfg, args.model_dir)
    split_path = validate_split_exists(cfg, args.split)

    # ── Step 4: Print banner ──────────────────────────────────
    print_banner(args)

    # ── Step 5: Load split DataFrame ─────────────────────────
    print(f"Loading {args.split} split from {split_path}...")
    df = pd.read_csv(split_path)
    print(f"Loaded {len(df):,} tickets\n")

    # ── Step 6: Run evaluation ────────────────────────────────
    from src.classifier.evaluate import evaluation_pipeline
    from src.utils.metrics import build_full_metrics_report

    results = evaluation_pipeline(
        test_df     = df,
        model_dir   = model_dir,
        config_path = args.config,
    )

    # ── Step 7: Print metrics ─────────────────────────────────
    print_metrics_table(results)
    print_threshold_results(
        results.get("threshold_results", {})
    )

    # Print classification report
    report = results.get("classification_report", "")
    if report:
        print("Classification Report:")
        print(report)

    # ── Step 8: Save predictions if requested ────────────────
    if args.save_predictions:
        from src.classifier.evaluate import (
            load_trained_model, get_predictions, build_dataloader
        )
        from src.classifier.dataset import load_tokenizer, build_dataloader
        import torch

        device    = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model, tokenizer = load_trained_model(model_dir, device)
        loader = build_dataloader(
            df         = df,
            tokenizer  = tokenizer,
            batch_size = cfg["classifier"]["eval_batch_size"],
            max_length = cfg["classifier"]["max_length"],
            shuffle    = False,
            is_test    = False,
        )

        y_pred, y_true, y_conf = get_predictions(model, loader, device)

        df["Prediction"]  = y_pred
        df["Confidence"]  = y_conf
        df["True_Label"]  = y_true

        pred_path = f"outputs/metrics/{args.split}_predictions.csv"
        df.to_csv(pred_path, index=False)
        print(f"Predictions saved → {pred_path}\n")

    # ── Step 9: Adversarial testing ───────────────────────────
    if args.adversarial:
        import torch
        from src.adversarial.adversarial_tests import run_adversarial_tests
        from src.classifier.evaluate import load_trained_model

        print("Running adversarial robustness tests...")

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        adv_model, adv_tokenizer = load_trained_model(
            model_dir = model_dir,
            device    = device,
        )

        adv_score, bonus_earned, adv_report = run_adversarial_tests(
            model       = adv_model,
            tokenizer   = adv_tokenizer,
            config_path = args.config,
            device      = device,
        )

        print_adversarial_results(adv_report)

    # ── Final verdict ─────────────────────────────────────────
    all_passed = results.get(
        "threshold_results", {}
    ).get("all_passed", False)

    if all_passed:
        print("✔  Submission VERIFIED — all thresholds met!")
    else:
        print("✘  Submission NOT VERIFIED — see failing metrics above.")

    print()
    print("Outputs saved to:")
    print(f"  {cfg['evaluation']['metrics_path']}")
    print(f"  {cfg['evaluation']['confusion_matrix_path']}")
    print()


if __name__ == "__main__":
    main()