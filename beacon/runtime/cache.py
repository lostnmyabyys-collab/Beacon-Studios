"""KV Cache for efficient inference."""

from typing import Optional

import torch

from beacon.logging import get_logger

logger = get_logger(__name__)


class KVCache:
    """Key-Value cache for efficient inference."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        """Initialize KV cache.

        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Head dimension
            device: Device to store cache on
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.position = 0
        
        # Initialize cache
        self.k_cache = torch.zeros(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            device=device,
        )
        self.v_cache = torch.zeros(
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            device=device,
        )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs.

        Args:
            k: New key tensor (batch_size, seq_len, num_heads, head_dim)
            v: New value tensor (batch_size, seq_len, num_heads, head_dim)

        Returns:
            tuple: (cached_k, cached_v) - full cached tensors
        """
        seq_len = k.shape[1]
        
        # Update cache
        self.k_cache[:, self.position:self.position + seq_len] = k
        self.v_cache[:, self.position:self.position + seq_len] = v
        
        # Increment position
        self.position += seq_len
        
        # Return cached tensors
        return (
            self.k_cache[:, :self.position],
            self.v_cache[:, :self.position],
        )

    def reset(self) -> None:
        """Reset cache."""
        self.position = 0

    def get_memory_usage(self) -> int:
        """Get memory usage in bytes.

        Returns:
            int: Memory usage
        """
        return (
            self.k_cache.element_size() * self.k_cache.nelement() +
            self.v_cache.element_size() * self.v_cache.nelement()
        )
