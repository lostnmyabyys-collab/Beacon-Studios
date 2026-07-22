"""Rotary position embeddings (RoPE)."""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """Initialize RoPE.

        Args:
            dim: Embedding dimension
            max_seq_len: Maximum sequence length
            base: Base for frequency calculation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache
        self.cached_cos = None
        self.cached_sin = None
        self.cached_seq_len = 0

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get rotation matrices.

        Args:
            x: Input tensor (batch_size, seq_len, dim)
            seq_len: Optional sequence length override

        Returns:
            tuple: (cos, sin) rotation matrices
        """
        seq_len = seq_len or x.shape[1]
        device = x.device
        
        # Use cache if available
        if (
            self.cached_seq_len >= seq_len
            and self.cached_cos.device == device
        ):
            return self.cached_cos[:seq_len], self.cached_sin[:seq_len]
        
        # Compute positions
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Expand to full dimension
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        
        # Cache
        self.cached_cos = cos
        self.cached_sin = sin
        self.cached_seq_len = seq_len
        
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings.

    Args:
        q: Query tensor (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor (batch_size, num_heads, seq_len, head_dim)
        cos: Cosine component
        sin: Sine component

    Returns:
        tuple: (q_rot, k_rot) rotated tensors
    """
    cos = cos[:, :, :, : cos.shape[-1] // 2 * 2]
    sin = sin[:, :, :, : sin.shape[-1] // 2 * 2]
    
    q_rot = (
        q[..., : cos.shape[-1]] * cos +
        q[..., cos.shape[-1] :] * (-sin)
    )
    k_rot = (
        k[..., : cos.shape[-1]] * cos +
        k[..., cos.shape[-1] :] * (-sin)
    )
    
    return q_rot, k_rot
