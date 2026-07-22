"""Logger configuration."""

import os
import sys
from pathlib import Path

from loguru import logger as loguru_logger


_configured = False


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure logging.

    Args:
        level: Log level
        log_file: Optional log file path
        format_string: Optional custom format
    """
    global _configured

    if _configured:
        return

    # Remove default handler
    loguru_logger.remove()

    # Default format
    if format_string is None:
        format_string = "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Console handler
    loguru_logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
    )

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        loguru_logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="500 MB",
            retention="7 days",
        )

    _configured = True


def get_logger(name: str) -> loguru_logger:
    """Get logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        loguru Logger instance
    """
    # Configure on first use
    if not _configured:
        level = os.getenv("LOG_LEVEL", "INFO")
        log_file = os.getenv("LOG_FILE")
        configure_logging(level=level, log_file=log_file)

    return loguru_logger.bind(name=name)
