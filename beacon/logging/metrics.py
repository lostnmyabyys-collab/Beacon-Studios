"""Metrics collection and reporting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Metric:
    """Single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.metrics: dict[str, list[Metric]] = {}

    def record(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
            metadata: Optional metadata
        """
        metric = Metric(name=name, value=value, tags=tags or {}, metadata=metadata or {})
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metric)

    def get_metrics(self, name: str) -> list[Metric]:
        """Get metrics by name.

        Args:
            name: Metric name

        Returns:
            List of metrics
        """
        return self.metrics.get(name, [])

    def clear(self, name: str | None = None) -> None:
        """Clear metrics.

        Args:
            name: Optional specific metric to clear (all if None)
        """
        if name:
            self.metrics.pop(name, None)
        else:
            self.metrics.clear()

    def export(self) -> dict[str, list[dict[str, Any]]]:
        """Export metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            name: [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                    "metadata": m.metadata,
                }
                for m in metrics
            ]
            for name, metrics in self.metrics.items()
        }
