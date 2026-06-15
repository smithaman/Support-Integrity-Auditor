# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/utils/helpers.py
# ─────────────────────────────────────────

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from config.constants import (
    PRIORITY_MAP,
    PRIORITY_INV,
    PRIORITY_LEVELS,
    MISMATCH_HIDDEN_CRISIS,
    MISMATCH_FALSE_ALARM,
    CONSISTENT,
    TIER_MAP,
    TIER_DEFAULT,
    MISMATCH_THRESHOLD,
)
from src.utils.logger import get_sia_logger

logger = get_sia_logger(__name__)


# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Loads and returns the YAML config as a dictionary.

    Usage:
        cfg = load_config()
        lr  = cfg["classifier"]["learning_rate"]
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Config loaded from {config_path}")
    return config


# ══════════════════════════════════════════════════════════════
#  JSON UTILITIES
# ══════════════════════════════════════════════════════════════

def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Saves any JSON-serializable object to a file."""
    import numpy as np

    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, cls=SafeEncoder)
    logger.debug(f"Saved JSON → {path}")


def load_json(path: str) -> Any:
    """Loads a JSON file and returns the parsed object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  PRIORITY / SEVERITY CONVERTERS
# ══════════════════════════════════════════════════════════════

def priority_to_num(priority: str) -> int:
    """
    Converts priority label to numeric.
        "Low" → 1 | "Medium" → 2 | "High" → 3 | "Critical" → 4
    """
    val = PRIORITY_MAP.get(priority)
    if val is None:
        raise ValueError(
            f"Unknown priority '{priority}'. "
            f"Expected one of {PRIORITY_LEVELS}"
        )
    return val


def num_to_priority(num: Union[int, float]) -> str:
    """
    Converts numeric score to priority label.
        1 → "Low" | 2 → "Medium" | 3 → "High" | 4 → "Critical"
    Clips and rounds input to valid range.
    """
    clipped = int(round(max(1, min(4, num))))
    return PRIORITY_INV[clipped]


def score_to_label(score: float) -> str:
    """
    Converts a continuous fused score (1.0–4.0) to a priority label.
    Uses midpoint boundaries: <1.5=Low, <2.5=Medium, <3.5=High, else Critical
    """
    if score < 1.5:
        return "Low"
    elif score < 2.5:
        return "Medium"
    elif score < 3.5:
        return "High"
    else:
        return "Critical"


def get_mismatch_type(delta: float) -> str:
    """
    Returns mismatch type from signed severity delta.
        delta >= +1.5 → Hidden Crisis  (under-triaged)
        delta <= -1.5 → False Alarm    (over-triaged)
        else          → Consistent
    """
    if delta >= MISMATCH_THRESHOLD:
        return MISMATCH_HIDDEN_CRISIS
    elif delta <= -MISMATCH_THRESHOLD:
        return MISMATCH_FALSE_ALARM
    else:
        return CONSISTENT


# ══════════════════════════════════════════════════════════════
#  CUSTOMER TIER
# ══════════════════════════════════════════════════════════════

def derive_customer_tier(email: str) -> str:
    """
    Derives customer tier from email domain.
        enterprise.org → enterprise
        company.com    → business
        tech.io        → tech
        example.*      → standard
    """
    if not isinstance(email, str) or "@" not in email:
        return TIER_DEFAULT
    domain = email.split("@")[-1].lower().strip()
    return TIER_MAP.get(domain, TIER_DEFAULT)


# ══════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch
    to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass

    logger.debug(f"Random seed set to {seed}")


# ══════════════════════════════════════════════════════════════
#  PATH UTILITIES
# ══════════════════════════════════════════════════════════════

def ensure_dir(path: Union[str, Path]) -> Path:
    """Creates directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_file_exists(path: str, label: str = "File") -> None:
    """Raises FileNotFoundError with a clear message if path doesn't exist."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ══════════════════════════════════════════════════════════════
#  DATAFRAME UTILITIES
# ══════════════════════════════════════════════════════════════

def print_label_distribution(
    df: pd.DataFrame,
    col: str,
    title: str = "Label Distribution"
) -> None:
    """Prints value counts and percentages for a column."""
    counts = df[col].value_counts()
    total  = len(df)
    logger.info(f"{title}:")
    for label, count in counts.items():
        pct = count / total * 100
        logger.info(f"   {str(label):<15} {count:>6}  ({pct:.1f}%)")


def check_mismatch_rate(
    df: pd.DataFrame,
    label_col: str,
    min_rate: float = 0.20,
    max_rate: float = 0.40
) -> Tuple[float, bool]:
    """
    Checks whether the mismatch pseudo-label rate falls within
    the acceptable range (20–40%).

    Returns:
        (mismatch_rate, is_acceptable)
    """
    rate = df[label_col].mean()
    ok   = min_rate <= rate <= max_rate

    if ok:
        logger.info(f"Mismatch rate: {rate:.1%} ✔ (within {min_rate:.0%}–{max_rate:.0%})")
    else:
        logger.warning(
            f"Mismatch rate: {rate:.1%} ⚠ "
            f"(expected {min_rate:.0%}–{max_rate:.0%}) — "
            f"consider adjusting fusion threshold"
        )
    return rate, ok


# ══════════════════════════════════════════════════════════════
#  TIMING UTILITY
# ══════════════════════════════════════════════════════════════

class Timer:
    """
    Simple context manager for timing code blocks.

    Usage:
        with Timer("Signal 1 computation"):
            scores = compute_semantic_severity(texts)
    Output:
        Signal 1 computation completed in 12.34s
    """

    def __init__(self, label: str = "Block"):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        logger.info(f"{self.label} completed in {elapsed:.2f}s")


# ══════════════════════════════════════════════════════════════
#  NUMPY UTILITIES
# ══════════════════════════════════════════════════════════════

def save_numpy(arr: np.ndarray, path: str) -> None:
    """Saves a numpy array to disk."""
    ensure_dir(Path(path).parent)
    np.save(path, arr)
    logger.debug(f"Saved numpy array {arr.shape} → {path}")


def load_numpy(path: str) -> np.ndarray:
    """Loads a numpy array from disk."""
    check_file_exists(path, "Numpy array")
    return np.load(path, allow_pickle=False)


# ══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def format_dossier_summary(dossier: Dict) -> str:
    """
    Returns a compact single-line summary of a dossier.
    Used in logs and CLI output.

    Example output:
        [TKT-100003] Low → High | Hidden Crisis | Δ2.0 | conf=0.91
    """
    return (
        f"[{dossier['ticket_id']}] "
        f"{dossier['assigned_priority']} → {dossier['inferred_severity']} | "
        f"{dossier['mismatch_type']} | "
        f"Δ{dossier['severity_delta']} | "
        f"conf={dossier['confidence']}"
    )