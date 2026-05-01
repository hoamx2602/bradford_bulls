"""Loguru-based logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str | Path = "logs",
    run_name: str | None = None,
) -> None:
    """
    Configure loguru logger globally.

    Parameters
    ----------
    level : str
        Log level (DEBUG | INFO | WARNING | ERROR).
    log_to_file : bool
        Whether to additionally log to a file in log_dir.
    log_dir : str | Path
        Directory for log files.
    run_name : str | None
        Optional name for the log file (otherwise timestamp is used).
    """
    logger.remove()

    # Console
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File
    if log_to_file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{run_name}.log" if run_name else "{time:YYYY-MM-DD_HH-mm-ss}.log"
        logger.add(
            log_dir / filename,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="30 days",
        )


def get_logger(name: str | None = None):
    """Get a logger instance. Re-exports loguru's logger for consistency."""
    if name:
        return logger.bind(name=name)
    return logger
