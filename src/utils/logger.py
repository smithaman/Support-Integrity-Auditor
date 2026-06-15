# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  src/utils/logger.py
# ─────────────────────────────────────────

import logging
import os
from pathlib import Path


# ── ANSI color codes for console output ──────────────────────
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"


# ── Colored console formatter ─────────────────────────────────
class ColoredFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG:    Colors.CYAN,
        logging.INFO:     Colors.GREEN,
        logging.WARNING:  Colors.YELLOW,
        logging.ERROR:    Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        record.levelname = (
            f"{color}{Colors.BOLD}{record.levelname:<8}{Colors.RESET}"
        )
        record.name = f"{Colors.BLUE}{record.name}{Colors.RESET}"
        return super().format(record)


# ── Stage banner helper ───────────────────────────────────────
STAGE_COLORS = {
    1: Colors.CYAN,
    2: Colors.GREEN,
    3: Colors.YELLOW,
    4: Colors.MAGENTA,
    5: Colors.BLUE,
    6: Colors.RED,
}


def get_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Returns a configured logger with colored console output
    and optional file output.

    Args:
        name     : Logger name — use __name__ in each module
        log_file : Path to log file (optional). If None, console only.
        level    : Logging level string — DEBUG / INFO / WARNING / ERROR

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    # ── Console handler (colored) ─────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)

    # ── File handler (plain text) ─────────────────────────────
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt=datefmt)
        )
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def log_stage(logger: logging.Logger, stage_num: int, stage_name: str) -> None:
    """
    Prints a clearly visible stage banner to the console.

    Usage:
        log_stage(logger, 1, "Pseudo Label Generation")

    Output:
        ══════════════════════════════════════
         STAGE 1 : Pseudo Label Generation
        ══════════════════════════════════════
    """
    color  = STAGE_COLORS.get(stage_num, Colors.WHITE)
    line   = "═" * 50
    banner = (
        f"\n{color}{line}{Colors.RESET}\n"
        f"{color} STAGE {stage_num} : {stage_name.upper()}{Colors.RESET}\n"
        f"{color}{line}{Colors.RESET}"
    )
    logger.info(banner)


def log_step(logger: logging.Logger, step: str) -> None:
    """
    Prints a step marker inside a stage.

    Usage:
        log_step(logger, "Loading dataset")

    Output:
        ── Loading dataset ──
    """
    logger.info(f"{Colors.CYAN}── {step} ──{Colors.RESET}")


def log_metrics(logger: logging.Logger, metrics: dict, title: str = "Metrics") -> None:
    """
    Prints a metrics summary block.

    Usage:
        log_metrics(logger, {"accuracy": 0.85, "macro_f1": 0.83})
    """
    logger.info(f"{Colors.BOLD}{title}{Colors.RESET}")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key:<25} {Colors.YELLOW}{value:.4f}{Colors.RESET}")
        else:
            logger.info(f"   {key:<25} {Colors.YELLOW}{value}{Colors.RESET}")


def log_success(logger: logging.Logger, message: str) -> None:
    """Logs a green success message."""
    logger.info(f"{Colors.GREEN}✔  {message}{Colors.RESET}")


def log_warning(logger: logging.Logger, message: str) -> None:
    """Logs a yellow warning message."""
    logger.warning(f"{Colors.YELLOW}⚠  {message}{Colors.RESET}")


def log_error(logger: logging.Logger, message: str) -> None:
    """Logs a red error message."""
    logger.error(f"{Colors.RED}✘  {message}{Colors.RESET}")


# ── Default project-level logger ─────────────────────────────
def get_sia_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper that always writes to outputs/logs/sia.log.
    Use this in all src/ modules.

    Usage:
        from src.utils.logger import get_sia_logger
        logger = get_sia_logger(__name__)
    """
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return get_logger(
        name=name,
        log_file=str(log_dir / "sia.log"),
        level="INFO"
    )