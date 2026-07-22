"""Model metadata."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class ModelMetadata(BaseModel):
    """Model metadata."""

    model_id: str
    model_name: str
    model_type: str  # vx, neo, cz
    version: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    context_length: int = 4096
    tokenizer: str = ""
    checkpoint_path: str = ""
    training_date: datetime | None = None
    benchmark_scores: dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    supported_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
