"""
train_pipeline.py — Standalone Training Script (Root Entry Point)

Run this to execute the full SIA training pipeline:
  Step 1 : Data Preprocessing
  Step 2 : Generate Embeddings
  Step 3 : Compute Signal 1 (Semantic Severity)
  Step 4 : Compute Signal 2 (Resolution Time Severity)
  Step 5 : Fuse Signals → Inferred Severity
  Step 6 : Generate Pseudo Labels
  Step 7 : Train DeBERTa-v3-small Classifier
  Step 8 : Build FAISS Index

Usage:
  python train_pipeline.py
  python train_pipeline.py --config config/config.yaml
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  train_pipeline.py  (ROOT ENTRY POINT)
#
#  BOUNDARY RULE:
#  This file ONLY handles:
#    1. Argument parsing
#    2. Loading config
#    3. Validating paths
#    4. Calling src/pipeline/train_pipeline.py
#  ALL logic lives in src/
# ─────────────────────────────────────────

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Usage:
        python train_pipeline.py
        python train_pipeline.py --config config/config.yaml
        python train_pipeline.py --skip-training
        python train_pipeline.py --skip-dossiers
        python train_pipeline.py --no-adversarial
    """
    parser = argparse.ArgumentParser(
        prog        = "train_pipeline.py",
        description = "SIA — Support Integrity Auditor Training Pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python train_pipeline.py
  python train_pipeline.py --config config/config.yaml
  python train_pipeline.py --skip-training
  python train_pipeline.py --skip-dossiers --no-adversarial
        """,
    )

    parser.add_argument(
        "--config",
        type    = str,
        default = "config/config.yaml",
        help    = "Path to config.yaml (default: config/config.yaml)",
    )

    parser.add_argument(
        "--skip-training",
        action  = "store_true",
        default = False,
        help    = (
            "Skip classifier training. "
            "Uses saved model from outputs/models/deberta_classifier/. "
            "Useful for re-running dossiers or evaluation only."
        ),
    )

    parser.add_argument(
        "--skip-dossiers",
        action  = "store_true",
        default = False,
        help    = (
            "Skip evidence dossier generation. "
            "Useful for quick training + evaluation runs."
        ),
    )

    parser.add_argument(
        "--no-adversarial",
        action  = "store_true",
        default = False,
        help    = (
            "Skip adversarial robustness testing. "
            "Use when iterating quickly on model performance."
        ),
    )

    parser.add_argument(
        "--data",
        type    = str,
        default = None,
        help    = (
            "Override raw data path from config. "
            "e.g. --data data/raw/crm_tickets.csv"
        ),
    )

    return parser.parse_args()


def validate_config(config_path: str) -> dict:
    """
    Loads and validates config.yaml.
    Checks that critical paths and values are present.

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

    # Validate required sections
    required_sections = [
        "data", "preprocessing", "embeddings",
        "signal1", "signal2", "fusion",
        "pseudo_labels", "classifier",
        "faiss", "dossier", "evaluation",
    ]

    missing = [s for s in required_sections if s not in cfg]
    if missing:
        print(f"[ERROR] Missing config sections: {missing}")
        sys.exit(1)

    return cfg


def validate_data_path(cfg: dict, override_path: str = None) -> str:
    """
    Validates that the raw data file exists.

    Args:
        cfg           : Loaded config dict
        override_path : CLI override for data path

    Returns:
        Validated data path string
    """
    data_path = override_path or cfg["data"]["raw_path"]

    if not Path(data_path).exists():
        print(
            f"[ERROR] Dataset not found: {data_path}\n"
            f"Please download from:\n"
            f"  https://www.kaggle.com/datasets/ajverse/"
            f"customersupport-tickets-crm-dataset\n"
            f"And place at: {data_path}"
        )
        sys.exit(1)

    return data_path


def validate_output_dirs(cfg: dict) -> None:
    """
    Creates all required output directories if they don't exist.

    Args:
        cfg : Loaded config dict
    """
    from src.utils.helpers import ensure_dir

    dirs_to_create = [
        Path(cfg["data"]["processed_path"]).parent,
        Path(cfg["classifier"]["save_dir"]),
        Path(cfg["faiss"]["index_path"]).parent,
        Path(cfg["dossier"]["output_path"]).parent,
        Path(cfg["evaluation"]["metrics_path"]).parent,
        Path("outputs/logs"),
        Path("outputs/embeddings"),
        Path("outputs/metrics"),
    ]

    for d in dirs_to_create:
        ensure_dir(d)


def print_banner(cfg: dict, args: argparse.Namespace) -> None:
    """Prints startup banner with run configuration."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      SIA — Support Integrity Auditor             ║")
    print("║      Training Pipeline                           ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Config       : {args.config:<33}║")
    print(f"║  Data         : {cfg['data']['raw_path']:<33}║")
    print(f"║  Model        : {cfg['classifier']['model_name']:<33}║")
    print(f"║  Embeddings   : {cfg['embeddings']['model_name']:<33}║")
    print(f"║  Skip training: {str(args.skip_training):<33}║")
    print(f"║  Skip dossiers: {str(args.skip_dossiers):<33}║")
    print(f"║  Adversarial  : {str(not args.no_adversarial):<33}║")
    print("╚══════════════════════════════════════════════════╝")
    print()


def print_final_summary(results: dict) -> None:
    """
    Prints final pipeline summary with key metrics.
    """
    # Try multiple sources for metrics
    test_metrics = (
        results.get("evaluation", {}).get("metrics") or
        results.get("test_metrics") or
        results.get("final_metrics") or
        {}
    )

    adv      = results.get("adversarial", {})
    pl_stats = results.get("pseudo_label_stats", {})

    acc = test_metrics.get("accuracy",          0)
    f1  = test_metrics.get("macro_f1",          0)
    rc  = test_metrics.get("recall_consistent", 0)
    rm  = test_metrics.get("recall_mismatch",   0)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      PIPELINE COMPLETE — FINAL SUMMARY          ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Accuracy          : {acc:.4f}                       ║")
    print(f"║  Macro F1          : {f1:.4f}                       ║")
    print(f"║  Recall Consistent : {rc:.4f}                       ║")
    print(f"║  Recall Mismatch   : {rm:.4f}                       ║")

    all_pass = (
        acc >= 0.83 and
        f1  >= 0.82 and
        rc  >= 0.78 and
        rm  >= 0.78
    )
    verdict = "✔ VERIFIED" if all_pass else "✘ NOT VERIFIED"
    print(f"║  Submission        : {verdict:<29}║")

    mrate = pl_stats.get("mismatch_rate", 0)
    print(f"║  Mismatch Rate     : {mrate:.1%}                       ║")

    if adv:
        adv_score = adv.get("score", 0)
        bonus     = "YES (+10%)" if adv.get("bonus_earned") else "NO"
        print(f"║  Adversarial Score : {adv_score}/10                        ║")
        print(f"║  Bonus             : {bonus:<29}║")

    print("╚══════════════════════════════════════════════════╝")
    print()

    if all_pass:
        print("✔  All verification thresholds met!")
        print("✔  Submission is VERIFIED.")
    else:
        print("✘  Some thresholds not met.")
        print("   Adjust hyperparameters and retrain.")

    print()
    print("Outputs saved to:")
    print("  outputs/models/deberta_classifier/")
    print("  outputs/dossiers/evidence_dossiers_verified.json")
    print("  outputs/metrics/evaluation.json")
    print("  outputs/metrics/adversarial_results.json")
    print()


def main() -> None:
    """
    Root entry point for SIA training pipeline.

    Flow:
        1. Parse CLI arguments
        2. Load + validate config
        3. Validate data path
        4. Create output directories
        5. Call src/pipeline/train_pipeline.py
        6. Print final summary
    """
    # ── Step 1: Parse arguments ───────────────────────────────
    args = parse_args()

    # ── Step 2: Load + validate config ───────────────────────
    cfg = validate_config(args.config)

    # Override data path if provided
    if args.data:
        cfg["data"]["raw_path"] = args.data

    # ── Step 3: Validate data path ────────────────────────────
    validate_data_path(cfg, override_path=args.data)

    # ── Step 4: Create output directories ────────────────────
    validate_output_dirs(cfg)

    # ── Step 5: Print banner ──────────────────────────────────
    print_banner(cfg, args)

    # ── Step 6: Call training pipeline ───────────────────────
    from src.pipeline.train_pipeline import run_training_pipeline

    results = run_training_pipeline(
        config_path     = args.config,
        skip_training   = args.skip_training,
        skip_dossiers   = args.skip_dossiers,
        run_adversarial = not args.no_adversarial,
    )

    # ── Step 7: Print final summary ───────────────────────────
    print_final_summary(results)


if __name__ == "__main__":
    main()