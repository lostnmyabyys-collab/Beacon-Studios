"""Logging and metrics collection."""

import logging
from typing import Any


class MetricsCollector:
    """Collects and manages metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: dict[str, list[float]] = {}

    def record(self, key: str, value: float) -> None:
        """Record metric.

        Args:
            key: Metric key
            value: Metric value
        """
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def export(self) -> dict[str, Any]:
        """Export metrics.

        Returns:
            dict: Metrics
        """
        return self.metrics

    def clear(self) -> None:
        """Clear metrics."""
        self.metrics.clear()


def get_logger(name: str) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
