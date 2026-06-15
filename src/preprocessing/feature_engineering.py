# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/preprocessing/feature_engineering.py
# ─────────────────────────────────────────

import re
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from config.constants import (
    COL_SUBJECT,
    COL_DESCRIPTION,
    COL_CATEGORY,
    COL_PRIORITY,
    COL_CHANNEL,
    COL_RT,
    COL_CUSTOMER_TIER,
    COL_COMBINED_TEXT,
    COL_MODEL_INPUT,
    URGENCY_KEYWORDS,
    CATEGORY_SEVERITY_BIAS,
    TICKET_CHANNELS,
    PRIORITY_LEVELS,
)
from src.utils.logger import get_sia_logger, log_step, log_success

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  DEBERTA INPUT STRING BUILDER
# ══════════════════════════════════════════════════════════════

def build_model_input(row: pd.Series) -> str:
    """
    Builds the structured input string for DeBERTa.

    Format:
        [CHANNEL: Email] [TIER: enterprise]
        [CATEGORY: Technical] [PRIORITY: Low]
        <combined_text>

    NOTE: Resolution Time is deliberately excluded.
    It is used in Signal 2 (pseudo-label generation)
    and including it here would cause leakage —
    the classifier would re-learn the labeling rule
    rather than genuine semantic mismatch patterns.

    Architecture:
        Pseudo Labels  → Semantic Signal + Resolution Signal
        Classifier     → Text + Channel + Tier + Category + Priority
    """
    channel  = str(row.get(COL_CHANNEL,       "unknown")).strip()
    tier     = str(row.get(COL_CUSTOMER_TIER,  "standard")).strip()
    category = str(row.get(COL_CATEGORY,       "General Inquiry")).strip()
    priority = str(row.get(COL_PRIORITY,       "Medium")).strip()
    text     = str(row.get(COL_COMBINED_TEXT,  "")).strip()

    model_input = (
        f"[CHANNEL: {channel}] "
        f"[TIER: {tier}] "
        f"[CATEGORY: {category}] "
        f"[PRIORITY: {priority}] "
        f"{text}"
    )

    return model_input


def build_all_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies build_model_input to every row in the DataFrame.
    Adds a new column: Model_Input

    Usage:
        df = build_all_model_inputs(df)
    """
    log_step(logger, "Building DeBERTa model input strings")

    df = df.copy()
    df[COL_MODEL_INPUT] = df.apply(build_model_input, axis=1)

    avg_len = df[COL_MODEL_INPUT].str.len().mean()
    max_len = df[COL_MODEL_INPUT].str.len().max()
    logger.info(f"Model input — avg: {avg_len:.0f} chars | max: {max_len} chars")

    log_success(logger, f"Model inputs built for {len(df):,} tickets")
    return df


# ══════════════════════════════════════════════════════════════
#  RULE-BASED NLP FEATURES
# ══════════════════════════════════════════════════════════════

def count_urgency_keywords(text: str) -> Dict[str, int]:
    """
    Counts urgency keywords per category in a text string.

    Returns:
        Dict with counts per category:
        {
            "critical": 2,
            "high": 1,
            "escalation": 0,
            "negation": 1,
            "total": 4
        }
    """
    text_lower = text.lower()
    counts = {}
    total  = 0

    for category, keywords in URGENCY_KEYWORDS.items():
        cat_count = sum(1 for kw in keywords if kw in text_lower)
        counts[category] = cat_count
        total += cat_count

    counts["total"] = total
    return counts


def detect_negation(text: str) -> bool:
    """
    Detects strong negation patterns indicating a broken/failing service.

    Patterns: "not working", "cannot access", "doesn't work", etc.
    """
    negation_patterns = [
        r"\bnot\s+working\b",
        r"\bcannot\s+\w+\b",
        r"\bcan't\s+\w+\b",
        r"\bunable\s+to\b",
        r"\bfailed\s+to\b",
        r"\bdoesn't\s+work\b",
        r"\bwon't\s+\w+\b",
        r"\bnever\s+\w+\b",
        r"\bstill\s+broken\b",
        r"\bstill\s+not\s+\w+\b",
        r"\bno\s+response\b",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in negation_patterns)


def detect_escalation_phrases(text: str) -> bool:
    """
    Detects phrases that indicate the customer is escalating.
    e.g. "I want to speak to a manager", "legal action", "cancel my account"
    """
    escalation_patterns = [
        r"\bmanager\b",
        r"\bescalat\w+\b",
        r"\blegal\s+action\b",
        r"\blawsuit\b",
        r"\bcancel\s+(my\s+)?(account|subscription)\b",
        r"\brefund\b",
        r"\bunacceptable\b",
        r"\bthis\s+is\s+ridiculous\b",
        r"\bdemand\b",
        r"\bcompensation\b",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in escalation_patterns)


def detect_fraud_indicators(text: str) -> bool:
    """
    Detects fraud/security indicators in ticket text.
    Dataset insight: Fraud category = always Critical/High.
    Even if miscategorized, these words signal high severity.
    """
    fraud_patterns = [
        r"\bfraud\b",
        r"\bunauthorized\s+(transaction|access|charge)\b",
        r"\bhacked\b",
        r"\bstolen\b",
        r"\bsecurity\s+breach\b",
        r"\bdata\s+breach\b",
        r"\bidentity\s+theft\b",
        r"\bsuspicious\s+activity\b",
        r"\bmy\s+account\s+was\s+(compromised|hacked|accessed)\b",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in fraud_patterns)


def compute_urgency_score(text: str) -> float:
    """
    Computes a rule-based urgency score (0.0–1.0) from text.

    Combines:
    - Keyword density (weighted by category severity)
    - Negation detection
    - Escalation detection
    - Fraud detection

    Returns:
        float between 0.0 (no urgency) and 1.0 (maximum urgency)
    """
    text_lower = text.lower()
    score      = 0.0

    # Keyword contribution (weighted)
    kw_counts = count_urgency_keywords(text_lower)
    score += min(kw_counts["critical"]  * 0.30, 0.30)  # max 0.30
    score += min(kw_counts["high"]      * 0.15, 0.20)  # max 0.20
    score += min(kw_counts["escalation"]* 0.10, 0.15)  # max 0.15
    score += min(kw_counts["negation"]  * 0.08, 0.15)  # max 0.15

    # Boolean features
    if detect_negation(text_lower):
        score += 0.10

    if detect_escalation_phrases(text_lower):
        score += 0.10

    if detect_fraud_indicators(text_lower):
        score += 0.20   # Strong boost for fraud indicators

    return round(min(score, 1.0), 4)


# ══════════════════════════════════════════════════════════════
#  CATEGORY SEVERITY BIAS
# ══════════════════════════════════════════════════════════════

def get_category_bias(category: str) -> float:
    """
    Returns a severity bias score (1–4) based on Issue_Category.

    Dataset insight:
        Fraud          → 4.0 (always Critical/High)
        Technical      → 3.0 (usually High)
        Billing        → 2.0 (usually Medium)
        General Inquiry→ 1.0 (usually Low)
        Account        → 1.0 (usually Low)
    """
    return float(CATEGORY_SEVERITY_BIAS.get(category, 2.0))


# ══════════════════════════════════════════════════════════════
#  FULL FEATURE ENGINEERING PIPELINE
# ══════════════════════════════════════════════════════════════

def compute_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all rule-based NLP features and adds them as columns.

    New columns added:
        urgency_score       — float 0.0–1.0
        has_negation        — bool
        has_escalation      — bool
        has_fraud           — bool
        keyword_count_total — int
        category_bias       — float 1.0–4.0

    These features are used in:
        - Signal 1 (optional category bias boost)
        - Evidence dossier (keyword evidence grounding)
        - Adversarial testing analysis
    """
    log_step(logger, "Computing rule-based NLP features")

    df = df.copy()
    text_col = COL_COMBINED_TEXT

    df["urgency_score"]       = df[text_col].apply(compute_urgency_score)
    df["has_negation"]        = df[text_col].apply(detect_negation)
    df["has_escalation"]      = df[text_col].apply(detect_escalation_phrases)
    df["has_fraud"]           = df[text_col].apply(detect_fraud_indicators)
    df["keyword_count_total"] = df[text_col].apply(
        lambda t: count_urgency_keywords(t)["total"]
    )
    df["category_bias"]       = df[COL_CATEGORY].apply(get_category_bias)

    # Log summary stats
    logger.info(f"Avg urgency score    : {df['urgency_score'].mean():.3f}")
    logger.info(f"Has negation         : {df['has_negation'].sum():,} ({df['has_negation'].mean():.1%})")
    logger.info(f"Has escalation       : {df['has_escalation'].sum():,} ({df['has_escalation'].mean():.1%})")
    logger.info(f"Has fraud indicators : {df['has_fraud'].sum():,} ({df['has_fraud'].mean():.1%})")

    log_success(logger, "NLP feature engineering complete")
    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Runs after preprocess_pipeline().

    Steps:
        1. Build DeBERTa model input strings
        2. Compute rule-based NLP features

    Args:
        df : Preprocessed DataFrame from preprocess_pipeline()

    Returns:
        DataFrame with model inputs and NLP features added
    """
    logger.info("Starting feature engineering pipeline")

    df = build_all_model_inputs(df)
    df = compute_nlp_features(df)

    log_success(logger, "Feature engineering complete")
    return df