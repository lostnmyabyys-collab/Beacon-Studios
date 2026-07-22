"""Batch processing for inference."""

from dataclasses import dataclass
from typing import Optional

import torch

from beacon.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationRequest:
    """Generation request."""

    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    stop_sequences: Optional[list[str]] = None


class BatchManager:
    """Manages batching for inference."""

    def __init__(self, max_batch_size: int = 32, max_tokens_per_batch: int = 8192):
        """Initialize batch manager.

        Args:
            max_batch_size: Maximum batch size
            max_tokens_per_batch: Maximum tokens per batch
        """
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.queue: list[GenerationRequest] = []
        logger.info(
            f"Initialized batch manager (max_batch_size={max_batch_size}, "
            f"max_tokens_per_batch={max_tokens_per_batch})"
        )

    def add_request(self, request: GenerationRequest) -> None:
        """Add request to queue.

        Args:
            request: Generation request
        """
        self.queue.append(request)

    def get_batch(self) -> Optional[list[GenerationRequest]]:
        """Get next batch.

        Returns:
            list: Batch of requests or None if queue empty
        """
        if not self.queue:
            return None

        batch = []
        total_tokens = 0

        while (
            self.queue
            and len(batch) < self.max_batch_size
            and total_tokens < self.max_tokens_per_batch
        ):
            request = self.queue.pop(0)
            batch.append(request)
            total_tokens += request.max_tokens

        return batch if batch else None

    def get_queue_size(self) -> int:
        """Get queue size.

        Returns:
            int: Number of pending requests
        """
        return len(self.queue)
