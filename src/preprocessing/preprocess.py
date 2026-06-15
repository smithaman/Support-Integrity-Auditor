# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/preprocessing/preprocess.py
# ─────────────────────────────────────────

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config.constants import (
    COL_TICKET_ID,
    COL_EMAIL,
    COL_SUBJECT,
    COL_DESCRIPTION,
    COL_CATEGORY,
    COL_PRIORITY,
    COL_CHANNEL,
    COL_RT,
    COL_SATISFACTION,
    COL_COMBINED_TEXT,
    COL_PRIORITY_NUM,
    COL_CUSTOMER_TIER,
    COLS_TO_DROP,
    PRIORITY_MAP,
    PRIORITY_LEVELS,
    TIER_DEFAULT,
    NOISE_PATTERN,
)
from src.utils.helpers import (
    derive_customer_tier,
    print_label_distribution,
    ensure_dir,
    load_config,
)
from src.utils.logger import get_sia_logger, log_step, log_success, log_warning

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the raw CRM CSV file.
    Validates that all required columns are present.
    """
    log_step(logger, f"Loading dataset from {path}")

    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} tickets | {df.shape[1]} columns")

    # Validate required columns
    required = [
        COL_TICKET_ID, COL_SUBJECT, COL_DESCRIPTION,
        COL_PRIORITY, COL_CHANNEL, COL_RT, COL_EMAIL, COL_CATEGORY
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log_success(logger, "Dataset loaded and validated")
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 2 — MISSING VALUES
# ══════════════════════════════════════════════════════════════

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    log_step(logger, "Handling missing values")

    before = len(df)
    df = df.dropna(subset=[
        COL_SUBJECT, COL_DESCRIPTION, COL_PRIORITY
    ]).copy()

    # Fix FutureWarning — use assignment instead of inplace
    df[COL_CHANNEL]      = df[COL_CHANNEL].fillna("unknown")
    df[COL_CATEGORY]     = df[COL_CATEGORY].fillna("General Inquiry")
    df[COL_RT]           = df[COL_RT].fillna(df[COL_RT].median())
    df[COL_EMAIL]        = df[COL_EMAIL].fillna("unknown@example.com")
    df[COL_SATISFACTION] = df[COL_SATISFACTION].fillna(
        df[COL_SATISFACTION].median()
    )

    after = len(df)
    if before != after:
        log_warning(logger, f"Dropped {before - after} rows")
    else:
        log_success(logger, "No missing values found")

    return df


# ══════════════════════════════════════════════════════════════
#  STEP 3 — TEXT CLEANING
# ══════════════════════════════════════════════════════════════

def strip_noise(text: str) -> str:
    """
    Removes the filler sentences appended to descriptions.
    Dataset-specific: e.g. "Lay soon message show know main."
    Pattern: ends with 3–6 random capitalized words then period.
    """
    return re.sub(NOISE_PATTERN, "", text).strip()


def clean_text(text: str) -> str:
    """
    Cleans a single text string:
    - Strips noise filler sentences
    - Lowercases
    - Removes URLs
    - Removes excessive punctuation
    - Collapses whitespace
    """
    if not isinstance(text, str):
        return ""

    # Remove noise pattern first (before lowercasing)
    text = strip_noise(text)

    # Remove "Hi Support," greeting (very common in this dataset)
    text = re.sub(r"^hi\s+support[\s,]+", "", text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Lowercase
    text = text.lower()

    # Remove special characters — keep letters, digits, basic punctuation
    text = re.sub(r"[^a-z0-9\s\.,!?'\-]", " ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Applies text cleaning to Subject and Description columns."""
    log_step(logger, "Cleaning text fields")

    df = df.copy()
    df[COL_SUBJECT]     = df[COL_SUBJECT].apply(clean_text)
    df[COL_DESCRIPTION] = df[COL_DESCRIPTION].apply(clean_text)

    # Log average lengths after cleaning
    avg_subj = df[COL_SUBJECT].str.len().mean()
    avg_desc = df[COL_DESCRIPTION].str.len().mean()
    logger.info(f"Avg subject length after cleaning : {avg_subj:.0f} chars")
    logger.info(f"Avg description length after cleaning : {avg_desc:.0f} chars")

    log_success(logger, "Text cleaning complete")
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 4 — MERGE TEXT
# ══════════════════════════════════════════════════════════════

def merge_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Subject and Description into a single combined_text field.
    Format: "<subject> [SEP] <description>"
    The [SEP] token helps the embedding model distinguish the two parts.
    """
    log_step(logger, "Merging subject + description")

    df = df.copy()
    df[COL_COMBINED_TEXT] = (
        df[COL_SUBJECT].fillna("") +
        " [SEP] " +
        df[COL_DESCRIPTION].fillna("")
    )

    avg_len = df[COL_COMBINED_TEXT].str.len().mean()
    max_len = df[COL_COMBINED_TEXT].str.len().max()
    logger.info(f"Combined text — avg: {avg_len:.0f} chars | max: {max_len} chars")

    log_success(logger, "Text merge complete")
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 5 — ENCODE METADATA
# ══════════════════════════════════════════════════════════════

def encode_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes structured metadata:
    - Priority_Level → Priority_Numeric (1–4)
    - Customer_Email → Customer_Tier (enterprise/business/tech/standard)

    Validates Priority_Level values are known.
    """
    log_step(logger, "Encoding metadata")

    df = df.copy()

    # Validate priority values
    unknown_priorities = df[~df[COL_PRIORITY].isin(PRIORITY_LEVELS)][COL_PRIORITY].unique()
    if len(unknown_priorities) > 0:
        log_warning(logger, f"Unknown priority values found: {unknown_priorities}")

    # Priority → numeric
    df[COL_PRIORITY_NUM] = df[COL_PRIORITY].map(PRIORITY_MAP)

    # Email → customer tier
    df[COL_CUSTOMER_TIER] = df[COL_EMAIL].apply(derive_customer_tier)

    # Log tier distribution
    print_label_distribution(df, COL_CUSTOMER_TIER, "Customer Tier Distribution")
    print_label_distribution(df, COL_PRIORITY,      "Priority Distribution")

    log_success(logger, "Metadata encoding complete")
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 6 — DROP UNUSED COLUMNS
# ══════════════════════════════════════════════════════════════

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns not useful for modeling:
    Customer_Name, Assigned_Agent, Submission_Date
    """
    log_step(logger, f"Dropping unused columns: {COLS_TO_DROP}")

    existing = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing)

    log_success(logger, f"Dropped {len(existing)} columns")
    return df


# ══════════════════════════════════════════════════════════════
#  STEP 7 — TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════

def split_data(
    df: pd.DataFrame,
    test_size: float  = 0.15,
    val_size: float   = 0.15,
    seed: int         = 42,
    stratify_col: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train / val / test sets.
    Stratifies on stratify_col if provided (use Mismatch_Label
    after pseudo-labels are generated).

    Returns:
        train_df, val_df, test_df
    """
    log_step(logger, f"Splitting data — test={test_size} | val={val_size} | seed={seed}")

    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    # First split off test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=seed,
    )

    # Then split val from train
    stratify_tv = train_val[stratify_col] if stratify_col and stratify_col in train_val.columns else None
    val_ratio   = val_size / (1 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=stratify_tv,
        random_state=seed,
    )

    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    logger.info(f"Train : {len(train):>6,} ({len(train)/len(df):.0%})")
    logger.info(f"Val   : {len(val):>6,} ({len(val)/len(df):.0%})")
    logger.info(f"Test  : {len(test):>6,} ({len(test)/len(df):.0%})")

    log_success(logger, "Data split complete")
    return train, val, test


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════

def preprocess_pipeline(
    raw_path: str  = None,
    save_path: str = None,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline:
        load → missing → clean → merge → encode → drop columns

    Does NOT split the data here — splitting happens after
    pseudo-labels are generated so we can stratify on Mismatch_Label.

    Args:
        raw_path    : Path to raw CSV. Defaults to config value.
        save_path   : Path to save processed CSV. Defaults to config value.
        config_path : Path to config.yaml.

    Returns:
        Fully preprocessed DataFrame (unsplit).
    """
    cfg = load_config(config_path)

    raw_path  = raw_path  or cfg["data"]["raw_path"]
    save_path = save_path or cfg["data"]["processed_path"]

    logger.info("Starting preprocessing pipeline")

    df = load_data(raw_path)
    df = handle_missing(df)
    df = clean_texts(df)
    df = merge_text(df)
    df = encode_metadata(df)
    df = drop_columns(df)

    # Save processed file
    ensure_dir(Path(save_path).parent)
    df.to_csv(save_path, index=False)
    log_success(logger, f"Processed dataset saved → {save_path} ({len(df):,} rows)")

    return df