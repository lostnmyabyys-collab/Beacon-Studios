"""Logging configuration for Beacon platform."""

from beacon.logging.logger import configure_logging, get_logger
from beacon.logging.metrics import MetricsCollector

__all__ = [
    "configure_logging",
    "get_logger",
    "MetricsCollector",
]
