# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/classifier/dataset.py
# ─────────────────────────────────────────

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config.constants import (
    COL_MODEL_INPUT,
    COL_MISMATCH_LABEL,
    COL_TICKET_ID,
    COL_PRIORITY,
    COL_INFERRED_SEV,
    COL_MISMATCH_TYPE,
    COL_CONFIDENCE,
    LABEL_CONSISTENT,
    LABEL_MISMATCH,
    LABEL_NAMES,
)
from src.utils.helpers import load_config, set_seed
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  DATASET CLASS
# ══════════════════════════════════════════════════════════════

class SIADataset(Dataset):
    """
    PyTorch Dataset for DeBERTa-v3-small binary classifier.

    Each sample contains:
        input_ids      : Tokenized model input
        attention_mask : Attention mask
        label          : 0 (Consistent) or 1 (Mismatch)

    Input format (from feature_engineering.py):
        [CHANNEL: Email] [TIER: enterprise] [RT: 48hrs]
        [CATEGORY: Technical] [PRIORITY: Low]
        <ticket subject> [SEP] <ticket description>

    Args:
        df         : DataFrame with Model_Input and Mismatch_Label columns
        tokenizer  : Loaded HuggingFace tokenizer
        max_length : Maximum token length (default 512)
        is_test    : If True, labels are optional (inference mode)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        is_test: bool   = False,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.is_test    = is_test

        # Validate required columns
        if COL_MODEL_INPUT not in df.columns:
            raise ValueError(
                f"Column '{COL_MODEL_INPUT}' not found. "
                f"Run feature_engineering_pipeline() first."
            )

        if not is_test and COL_MISMATCH_LABEL not in df.columns:
            raise ValueError(
                f"Column '{COL_MISMATCH_LABEL}' not found. "
                f"Run pseudo_label_pipeline() first."
            )

        self.texts  = df[COL_MODEL_INPUT].fillna("").tolist()
        self.labels = (
            df[COL_MISMATCH_LABEL].values.astype(int).tolist()
            if not is_test and COL_MISMATCH_LABEL in df.columns
            else [0] * len(df)   # dummy labels for inference
        )

        # Store ticket IDs for result tracking
        self.ticket_ids = (
            df[COL_TICKET_ID].astype(str).tolist()
            if COL_TICKET_ID in df.columns
            else [str(i) for i in range(len(df))]
        )

        logger.debug(
            f"SIADataset created — "
            f"{len(self.texts)} samples | "
            f"max_length={max_length} | "
            f"is_test={is_test}"
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text  = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }

    def get_ticket_id(self, idx: int) -> str:
        """Returns the Ticket_ID for a given dataset index."""
        return self.ticket_ids[idx]


# ══════════════════════════════════════════════════════════════
#  CLASS WEIGHT COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_class_weights(
    df: pd.DataFrame,
    label_col: str    = COL_MISMATCH_LABEL,
    manual_weights: dict = None,
) -> torch.Tensor:
    """
    Computes class weights for weighted CrossEntropyLoss.

    Uses stronger weighting for 4:1 imbalance ratio.
    Formula: sqrt(total / (num_classes × count))
    Sqrt dampens extreme weights while still correcting imbalance.
    """
    counts    = df[label_col].value_counts().sort_index()
    total     = len(df)
    n_classes = 2

    if manual_weights:
        weights = [
            manual_weights.get("consistent", 1.0),
            manual_weights.get("mismatch",   1.0),
        ]
        logger.info(f"Using manual class weights: {weights}")
    else:
        weights = []
        for cls in range(n_classes):
            count = counts.get(cls, 1)
            # sqrt formula — stronger than linear for severe imbalance
            w = np.sqrt(total / (n_classes * count))
            weights.append(w)
            logger.info(
                f"Class {cls} ({LABEL_NAMES[cls]:<12}) — "
                f"count={count:>6,} | weight={w:.4f}"
            )

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights: {weight_tensor.tolist()}")
    return weight_tensor


# ══════════════════════════════════════════════════════════════
#  TOKENIZER LOADER
# ══════════════════════════════════════════════════════════════

def load_tokenizer(
    model_name: str = "microsoft/deberta-v3-small",
) -> AutoTokenizer:
    """
    Loads the DeBERTa-v3-small tokenizer.

    Args:
        model_name : HuggingFace model identifier

    Returns:
        Loaded AutoTokenizer
    """
    log_step(logger, f"Loading tokenizer: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log_success(logger, f"Tokenizer loaded — vocab_size={tokenizer.vocab_size:,}")
        return tokenizer

    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{model_name}': {e}\n"
            f"Run: pip install transformers"
        )


# ══════════════════════════════════════════════════════════════
#  DATALOADER BUILDERS
# ══════════════════════════════════════════════════════════════

def build_dataloader(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int  = 16,
    max_length: int  = 512,
    shuffle: bool    = True,
    is_test: bool    = False,
    num_workers: int = 0,
    seed: int        = 42,
) -> DataLoader:
    """
    Builds a PyTorch DataLoader for a given split.

    Args:
        df          : DataFrame split (train/val/test)
        tokenizer   : Loaded tokenizer
        batch_size  : Batch size
        max_length  : Max token length
        shuffle     : Shuffle data (True for train, False for val/test)
        is_test     : Inference mode (no labels required)
        num_workers : DataLoader workers (0 = main process)
        seed        : Random seed for reproducibility

    Returns:
        PyTorch DataLoader
    """
    dataset = SIADataset(
        df         = df,
        tokenizer  = tokenizer,
        max_length = max_length,
        is_test    = is_test,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        generator   = generator if shuffle else None,
    )

    logger.debug(
        f"DataLoader built — "
        f"{len(dataset)} samples | "
        f"batch_size={batch_size} | "
        f"batches={len(loader)} | "
        f"shuffle={shuffle}"
    )

    return loader


def build_all_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    config_path: str = "config/config.yaml",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train, val, and test DataLoaders in one call.

    Args:
        train_df    : Training split DataFrame
        val_df      : Validation split DataFrame
        test_df     : Test split DataFrame
        tokenizer   : Loaded tokenizer
        config_path : Path to config.yaml

    Returns:
        (train_loader, val_loader, test_loader)
    """
    cfg        = load_config(config_path)
    batch_size = cfg["classifier"]["train_batch_size"]
    eval_batch = cfg["classifier"]["eval_batch_size"]
    max_length = cfg["classifier"]["max_length"]
    seed       = cfg["classifier"]["seed"]

    log_step(logger, "Building DataLoaders")

    train_loader = build_dataloader(
        df         = train_df,
        tokenizer  = tokenizer,
        batch_size = batch_size,
        max_length = max_length,
        shuffle    = True,
        is_test    = False,
        seed       = seed,
    )

    val_loader = build_dataloader(
        df         = val_df,
        tokenizer  = tokenizer,
        batch_size = eval_batch,
        max_length = max_length,
        shuffle    = False,
        is_test    = False,
        seed       = seed,
    )

    test_loader = build_dataloader(
        df         = test_df,
        tokenizer  = tokenizer,
        batch_size = eval_batch,
        max_length = max_length,
        shuffle    = False,
        is_test    = False,
        seed       = seed,
    )

    logger.info(
        f"DataLoaders ready — "
        f"train={len(train_loader)} batches | "
        f"val={len(val_loader)} batches | "
        f"test={len(test_loader)} batches"
    )

    log_success(logger, "All DataLoaders built")
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════
#  DATASET VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame, split_name: str = "dataset") -> None:
    """
    Validates a DataFrame split before training.

    Checks:
        - Required columns present
        - No null model inputs
        - Label distribution is reasonable
        - No all-zero or all-one labels

    Args:
        df         : DataFrame split to validate
        split_name : Name for logging ("train" / "val" / "test")
    """
    log_step(logger, f"Validating {split_name} dataset")

    # Check required columns
    required = [COL_MODEL_INPUT, COL_MISMATCH_LABEL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {split_name}: {missing}")

    # Check for null inputs
    n_null = df[COL_MODEL_INPUT].isnull().sum()
    if n_null > 0:
        log_warning(logger, f"{n_null} null model inputs in {split_name}")

    # Check label distribution
    counts = df[COL_MISMATCH_LABEL].value_counts()
    n_consistent = counts.get(LABEL_CONSISTENT, 0)
    n_mismatch   = counts.get(LABEL_MISMATCH,   0)

    if n_consistent == 0:
        raise ValueError(f"No Consistent samples in {split_name}")
    if n_mismatch == 0:
        raise ValueError(f"No Mismatch samples in {split_name}")

    mismatch_rate = n_mismatch / len(df)

    logger.info(
        f"{split_name} — "
        f"total={len(df):,} | "
        f"consistent={n_consistent:,} | "
        f"mismatch={n_mismatch:,} | "
        f"mismatch_rate={mismatch_rate:.1%}"
    )

    log_success(logger, f"{split_name} dataset validated")


def apply_smote(
    df: pd.DataFrame,
    label_col: str = COL_MISMATCH_LABEL,
    seed: int      = 42,
) -> pd.DataFrame:
    """
    Applies SMOTE-style oversampling to balance classes.
    Oversample minority class (Mismatch) to 40% of dataset.

    Strategy: Random oversample (safer than SMOTE for text data)
    SMOTE on embeddings can distort text semantics.
    """
    consistent = df[df[label_col] == LABEL_CONSISTENT]
    mismatch   = df[df[label_col] == LABEL_MISMATCH]

    n_consistent = len(consistent)
    n_mismatch   = len(mismatch)

    # Target: mismatch = 40% of total
    # So: n_mismatch_new / (n_consistent + n_mismatch_new) = 0.40
    # → n_mismatch_new = 0.40 * n_consistent / 0.60
    # target_mismatch = int(0.40 * n_consistent / 0.60)
    target_mismatch = int(0.30 * n_consistent / 0.70)

    if target_mismatch <= n_mismatch:
        logger.info(
            f"No oversampling needed — "
            f"mismatch already {n_mismatch/len(df):.1%}"
        )
        return df

    # Oversample mismatch class
    n_to_add = target_mismatch - n_mismatch
    oversampled = mismatch.sample(
        n           = n_to_add,
        replace     = True,
        random_state = seed,
    )

    df_balanced = pd.concat(
        [consistent, mismatch, oversampled],
        ignore_index = True
    ).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Verify
    final_consistent = (df_balanced[label_col] == LABEL_CONSISTENT).sum()
    final_mismatch   = (df_balanced[label_col] == LABEL_MISMATCH).sum()
    new_rate         = final_mismatch / len(df_balanced)

    logger.info(
        f"Oversampling complete:\n"
        f"   Before → consistent={n_consistent:,} | mismatch={n_mismatch:,} | rate={n_mismatch/len(df):.1%}\n"
        f"   After  → consistent={final_consistent:,} | mismatch={final_mismatch:,} | rate={new_rate:.1%}\n"
        f"   Added  → {n_to_add:,} synthetic mismatch samples"
    )

    return df_balanced