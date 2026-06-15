"""
predict.py — Inference Script (Root Entry Point)

Accepts a CSV of new tickets and outputs:
  - predictions.csv  : ticket_id, prediction, confidence
  - dossiers.json    : full evidence dossier per flagged ticket

Usage:
  python predict.py --input data/raw/new_tickets.csv
  python predict.py --input data/raw/new_tickets.csv --output outputs/dossiers/
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  predict.py  (ROOT ENTRY POINT)
#
#  BOUNDARY RULE:
#  This file ONLY handles:
#    1. Argument parsing
#    2. Loading config
#    3. Validating paths
#    4. Calling src/pipeline/inference_pipeline.py
#  ALL logic lives in src/
# ─────────────────────────────────────────

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Usage:
        python predict.py --input data/raw/new_tickets.csv
        python predict.py --input data/raw/new_tickets.csv --output outputs/dossiers/
        python predict.py --input demo_data/sample_batch.csv --output outputs/dossiers/
    """
    parser = argparse.ArgumentParser(
        prog        = "predict.py",
        description = "SIA — Support Integrity Auditor Inference",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python predict.py --input demo_data/sample_batch.csv
  python predict.py --input data/raw/new_tickets.csv --output outputs/dossiers/
  python predict.py --input tickets.csv --output results/ --config config/config.yaml
        """,
    )

    parser.add_argument(
        "--input",
        type     = str,
        required = True,
        help     = "Path to input CSV file containing tickets to analyze",
    )

    parser.add_argument(
        "--output",
        type    = str,
        default = "outputs/dossiers/",
        help    = (
            "Directory to save predictions.csv and dossiers.json "
            "(default: outputs/dossiers/)"
        ),
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
        "--no-dossiers",
        action  = "store_true",
        default = False,
        help    = (
            "Skip dossier generation. "
            "Only outputs predictions.csv."
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
        "data", "embeddings", "signal1",
        "signal2", "fusion", "classifier",
        "faiss", "dossier",
    ]

    missing = [s for s in required_sections if s not in cfg]
    if missing:
        print(f"[ERROR] Missing config sections: {missing}")
        sys.exit(1)

    return cfg


def validate_input_path(input_path: str) -> None:
    """
    Validates that the input CSV exists and has required columns.

    Args:
        input_path : Path to input CSV file
    """
    import pandas as pd

    if not Path(input_path).exists():
        print(f"[ERROR] Input file not found: {input_path}")
        print(
            f"\nExpected a CSV file with columns:\n"
            f"  Ticket_ID, Ticket_Subject, Ticket_Description,\n"
            f"  Priority_Level, Ticket_Channel, Issue_Category,\n"
            f"  Resolution_Time_Hours, Customer_Email"
        )
        sys.exit(1)

    # Check it's a readable CSV
    try:
        df = pd.read_csv(input_path, nrows=2)
    except Exception as e:
        print(f"[ERROR] Cannot read CSV file: {e}")
        sys.exit(1)

    # Check required columns
    required_cols = [
        "Ticket_Subject",
        "Ticket_Description",
        "Priority_Level",
        "Resolution_Time_Hours",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(
            f"[ERROR] Missing required columns in input CSV:\n"
            f"  {missing_cols}\n\n"
            f"Found columns: {df.columns.tolist()}"
        )
        sys.exit(1)


def validate_model_exists(cfg: dict, model_dir_override: str = None) -> str:
    """
    Validates that the trained model exists.

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
            f"  python train_pipeline.py\n\n"
            f"Or specify a different model directory:\n"
            f"  python predict.py --input tickets.csv --model-dir /path/to/model"
        )
        sys.exit(1)

    # Check for model files
    required_files = ["config.json", "tokenizer_config.json"]
    missing_files  = [
        f for f in required_files
        if not (Path(model_dir) / f).exists()
    ]

    if missing_files:
        print(
            f"[ERROR] Model directory incomplete: {model_dir}\n"
            f"Missing files: {missing_files}\n"
            f"Re-run training to regenerate the model."
        )
        sys.exit(1)

    return model_dir


def validate_output_dir(output_dir: str) -> None:
    """
    Creates output directory if it doesn't exist.

    Args:
        output_dir : Path to output directory
    """
    from src.utils.helpers import ensure_dir
    ensure_dir(output_dir)


def print_banner(args: argparse.Namespace) -> None:
    """Prints startup banner."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      SIA — Support Integrity Auditor             ║")
    print("║      Inference Pipeline                          ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Input        : {args.input:<33}║")
    print(f"║  Output       : {args.output:<33}║")
    print(f"║  Config       : {args.config:<33}║")
    print(f"║  Dossiers     : {str(not args.no_dossiers):<33}║")
    print("╚══════════════════════════════════════════════════╝")
    print()


def print_final_summary(
    predictions_df,
    dossiers: list,
    output_dir: str,
) -> None:
    """
    Prints inference summary after pipeline completes.

    Args:
        predictions_df : DataFrame with predictions
        dossiers       : List of generated dossiers
        output_dir     : Output directory path
    """
    n_total    = len(predictions_df)
    n_mismatch = int(
        (predictions_df["Prediction"] == 1).sum()
    ) if "Prediction" in predictions_df.columns else 0
    n_consistent = n_total - n_mismatch
    mismatch_rate = n_mismatch / n_total if n_total > 0 else 0

    # Priority breakdown of mismatches
    mismatch_df = predictions_df[
        predictions_df.get("Prediction", 0) == 1
    ] if "Prediction" in predictions_df.columns else predictions_df

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      INFERENCE COMPLETE — SUMMARY               ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Total tickets    : {n_total:<29}║")
    print(f"║  Mismatches       : {n_mismatch:<29}║")
    print(f"║  Consistent       : {n_consistent:<29}║")
    print(f"║  Mismatch rate    : {mismatch_rate:.1%:<29}║")
    print(f"║  Dossiers         : {len(dossiers):<29}║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  predictions.csv  → {output_dir:<28}║")
    print(f"║  dossiers.json    → {output_dir:<28}║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Mismatch type breakdown
    if "Mismatch_Type" in predictions_df.columns:
        type_counts = (
            predictions_df[predictions_df["Prediction"] == 1]["Mismatch_Type"]
            .value_counts()
            .to_dict()
        )
        if type_counts:
            print("Mismatch Type Breakdown:")
            for mtype, count in type_counts.items():
                print(f"  {mtype:<20} {count:>5}")
            print()

    # High confidence mismatches
    if "Confidence" in predictions_df.columns:
        high_conf = predictions_df[
            (predictions_df.get("Prediction", 0) == 1) &
            (predictions_df["Confidence"] >= 0.9)
        ]
        print(f"High-confidence mismatches (conf >= 0.90): {len(high_conf)}")
        print()

    print("Done. Open the output files to review results.")
    print()


def main() -> None:
    """
    Root entry point for SIA inference pipeline.

    Flow:
        1. Parse CLI arguments
        2. Load + validate config
        3. Validate input CSV
        4. Validate model exists
        5. Create output directory
        6. Call src/pipeline/inference_pipeline.py
        7. Print final summary
    """
    # ── Step 1: Parse arguments ───────────────────────────────
    args = parse_args()

    # ── Step 2: Load + validate config ───────────────────────
    cfg = validate_config(args.config)

    # Override model dir if provided
    if args.model_dir:
        cfg["classifier"]["save_dir"] = args.model_dir

    # ── Step 3: Validate input CSV ────────────────────────────
    validate_input_path(args.input)

    # ── Step 4: Validate model exists ────────────────────────
    model_dir = validate_model_exists(cfg, args.model_dir)
    cfg["classifier"]["save_dir"] = model_dir

    # ── Step 5: Create output directory ──────────────────────
    validate_output_dir(args.output)

    # ── Step 6: Print banner ──────────────────────────────────
    print_banner(args)

    # ── Step 7: Call inference pipeline ──────────────────────
    from src.pipeline.inference_pipeline import run_inference_pipeline

    predictions_df, dossiers = run_inference_pipeline(
        input_path  = args.input,
        output_dir  = args.output,
        config_path = args.config,
    )

    # ── Step 8: Print final summary ───────────────────────────
    print_final_summary(
        predictions_df = predictions_df,
        dossiers       = dossiers,
        output_dir     = args.output,
    )


if __name__ == "__main__":
    main()