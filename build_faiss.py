"""
build_faiss.py — Standalone FAISS Index Builder

Loads ticket embeddings and builds the FAISS index.
Run this separately if you update the dataset without retraining.

Usage:
  python build_faiss.py
"""

# TODO: implement


# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  build_faiss.py  (ROOT ENTRY POINT)
#
#  BOUNDARY RULE:
#  This file ONLY handles:
#    1. Argument parsing
#    2. Loading config
#    3. Validating paths
#    4. Calling src/retrieval/build_index.py
#  ALL logic lives in src/
# ─────────────────────────────────────────

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Usage:
        python build_faiss.py
        python build_faiss.py --embeddings outputs/embeddings/ticket_embeddings.npy
        python build_faiss.py --data data/processed/pseudo_labels.csv
        python build_faiss.py --rebuild
    """
    parser = argparse.ArgumentParser(
        prog        = "build_faiss.py",
        description = "SIA — Build FAISS Vector Index for Semantic Search",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Description:
  Builds a FAISS vector index from ticket embeddings.
  Run this if you update the dataset without retraining,
  or if the index files are missing/corrupted.

Examples:
  python build_faiss.py
  python build_faiss.py --rebuild
  python build_faiss.py --embeddings outputs/embeddings/ticket_embeddings.npy
  python build_faiss.py --data data/processed/pseudo_labels.csv
        """,
    )

    parser.add_argument(
        "--config",
        type    = str,
        default = "config/config.yaml",
        help    = "Path to config.yaml (default: config/config.yaml)",
    )

    parser.add_argument(
        "--embeddings",
        type    = str,
        default = None,
        help    = (
            "Path to pre-computed ticket embeddings (.npy). "
            "Defaults to value in config.yaml. "
            "If not found, re-encodes from processed data."
        ),
    )

    parser.add_argument(
        "--data",
        type    = str,
        default = None,
        help    = (
            "Path to processed tickets CSV. "
            "Used if embeddings need to be regenerated. "
            "Defaults to pseudo_labels_path in config.yaml."
        ),
    )

    parser.add_argument(
        "--rebuild",
        action  = "store_true",
        default = False,
        help    = (
            "Force rebuild even if index already exists. "
            "Use after updating the dataset or embeddings."
        ),
    )

    parser.add_argument(
        "--validate",
        action  = "store_true",
        default = True,
        help    = "Validate index after building (default: True)",
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

    required_sections = ["data", "embeddings", "faiss"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        print(f"[ERROR] Missing config sections: {missing}")
        sys.exit(1)

    return cfg


def check_existing_index(cfg: dict, rebuild: bool) -> bool:
    """
    Checks if FAISS index already exists.

    Args:
        cfg     : Loaded config dict
        rebuild : Force rebuild flag

    Returns:
        True if should proceed with build, False if skip
    """
    index_path = cfg["faiss"]["index_path"]

    if Path(index_path).exists() and not rebuild:
        print(
            f"FAISS index already exists: {index_path}\n"
            f"Use --rebuild to force regeneration.\n"
            f"Skipping build."
        )
        return False

    if Path(index_path).exists() and rebuild:
        print(f"Rebuilding FAISS index (--rebuild flag set): {index_path}")

    return True


def resolve_embeddings(
    cfg: dict,
    embeddings_override: str = None,
    data_override: str       = None,
) -> tuple:
    """
    Resolves ticket embeddings — loads from disk or generates fresh.

    Priority:
        1. CLI --embeddings path (if provided and exists)
        2. Config embeddings save_path (if exists)
        3. Re-encode from processed data

    Args:
        cfg                  : Loaded config dict
        embeddings_override  : CLI override for embeddings path
        data_override        : CLI override for data path

    Returns:
        (embeddings, df)
        embeddings : np.ndarray (N, dim)
        df         : Processed DataFrame
    """
    import numpy as np
    import pandas as pd

    from src.utils.helpers import load_numpy

    emb_path  = embeddings_override or cfg["embeddings"]["save_path"]
    data_path = data_override or cfg["data"]["pseudo_labels_path"]

    # ── Try loading pre-computed embeddings ───────────────────
    if Path(emb_path).exists():
        print(f"Loading pre-computed embeddings from {emb_path}...")
        embeddings = load_numpy(emb_path).astype("float32")
        print(f"Embeddings loaded — shape={embeddings.shape}")

        # Load corresponding DataFrame
        if Path(data_path).exists():
            df = pd.read_csv(data_path)
            print(f"DataFrame loaded — {len(df):,} rows")

            if len(df) != len(embeddings):
                print(
                    f"[WARNING] Embeddings ({len(embeddings)}) and "
                    f"DataFrame ({len(df)}) size mismatch.\n"
                    f"Re-encoding from data..."
                )
                return _encode_fresh(cfg, data_path)

            return embeddings, df
        else:
            print(
                f"[WARNING] Data file not found: {data_path}\n"
                f"Using embeddings without metadata."
            )
            return embeddings, pd.DataFrame()

    # ── Re-encode from processed data ─────────────────────────
    print(
        f"Embeddings not found at {emb_path}.\n"
        f"Re-encoding from data..."
    )
    return _encode_fresh(cfg, data_path)


def _encode_fresh(cfg: dict, data_path: str) -> tuple:
    """
    Encodes tickets fresh from a processed CSV.

    Args:
        cfg       : Loaded config dict
        data_path : Path to processed tickets CSV

    Returns:
        (embeddings, df)
    """
    import pandas as pd
    from src.embeddings.sentence_encoder import load_encoder, encode_texts
    from src.utils.helpers import save_numpy

    if not Path(data_path).exists():
        print(
            f"[ERROR] Data file not found: {data_path}\n"
            f"Run train_pipeline.py first to generate processed data."
        )
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} tickets")

    if "combined_text" not in df.columns:
        print(
            "[ERROR] 'combined_text' column not found in data.\n"
            "Run preprocessing pipeline first."
        )
        sys.exit(1)

    print(f"Loading encoder: {cfg['embeddings']['model_name']}...")
    model = load_encoder(cfg["embeddings"]["model_name"])

    print(f"Encoding {len(df):,} tickets...")
    texts      = df["combined_text"].fillna("").tolist()
    embeddings = encode_texts(
        texts         = texts,
        model         = model,
        batch_size    = cfg["embeddings"]["batch_size"],
        normalize     = True,
        show_progress = True,
    )

    # Save for reuse
    save_path = cfg["embeddings"]["save_path"]
    save_numpy(embeddings, save_path)
    print(f"Embeddings saved → {save_path}")

    return embeddings, df


def print_banner(args: argparse.Namespace, cfg: dict) -> None:
    """Prints startup banner."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      SIA — Support Integrity Auditor             ║")
    print("║      FAISS Index Builder                         ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Config       : {args.config:<33}║")
    print(f"║  Index type   : {cfg['faiss']['index_type']:<33}║")
    print(f"║  Index path   : {cfg['faiss']['index_path']:<33}║")
    print(f"║  Rebuild      : {str(args.rebuild):<33}║")
    print(f"║  Validate     : {str(args.validate):<33}║")
    print("╚══════════════════════════════════════════════════╝")
    print()


def print_final_summary(
    index,
    cfg: dict,
) -> None:
    """
    Prints build summary after index is created.

    Args:
        index : Built FAISS index
        cfg   : Loaded config dict
    """
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║      FAISS INDEX BUILD COMPLETE                  ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Total vectors : {index.ntotal:<32}║")
    print(f"║  Dimension     : {index.d:<32}║")
    print(f"║  Index type    : {cfg['faiss']['index_type']:<32}║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Index saved → {cfg['faiss']['index_path']:<33}║")
    print(f"║  Embeddings  → {cfg['faiss']['embeddings_path']:<33}║")
    print(f"║  Metadata    → {cfg['faiss']['metadata_path']:<33}║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("FAISS index ready for semantic search.")
    print(
        "Run inference with:\n"
        "  python predict.py --input <your_tickets.csv>"
    )
    print()


def main() -> None:
    """
    Root entry point for FAISS index building.

    Flow:
        1. Parse CLI arguments
        2. Load + validate config
        3. Check if index already exists
        4. Resolve embeddings (load or regenerate)
        5. Call src/retrieval/build_index.py
        6. Print final summary
    """
    from src.utils.helpers import ensure_dir

    # ── Step 1: Parse arguments ───────────────────────────────
    args = parse_args()

    # ── Step 2: Load + validate config ───────────────────────
    cfg = validate_config(args.config)

    # ── Step 3: Check existing index ─────────────────────────
    should_build = check_existing_index(cfg, args.rebuild)
    if not should_build:
        sys.exit(0)

    # ── Step 4: Print banner ──────────────────────────────────
    print_banner(args, cfg)

    # ── Step 5: Create output directories ────────────────────
    ensure_dir(Path(cfg["faiss"]["index_path"]).parent)
    ensure_dir(Path(cfg["faiss"]["metadata_path"]).parent)
    ensure_dir(Path(cfg["embeddings"]["save_path"]).parent)

    # ── Step 6: Resolve embeddings ────────────────────────────
    print("Resolving ticket embeddings...")
    embeddings, df = resolve_embeddings(
        cfg                 = cfg,
        embeddings_override = args.embeddings,
        data_override       = args.data,
    )

    print(
        f"\nReady to build index:\n"
        f"  Embeddings shape : {embeddings.shape}\n"
        f"  DataFrame rows   : {len(df):,}\n"
    )

    # ── Step 7: Build FAISS index ─────────────────────────────
    from src.retrieval.build_index import build_faiss_pipeline

    print("Building FAISS index...")
    index = build_faiss_pipeline(
        df          = df,
        embeddings  = embeddings,
        config_path = args.config,
    )

    # ── Step 8: Validate index ────────────────────────────────
    if args.validate:
        from src.retrieval.build_index import validate_index
        print("\nValidating index...")
        validate_index(
            index      = index,
            embeddings = embeddings,
            n_test     = 10,
        )

    # ── Step 9: Print final summary ───────────────────────────
    print_final_summary(index, cfg)


if __name__ == "__main__":
    main()