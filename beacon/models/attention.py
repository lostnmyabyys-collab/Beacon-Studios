"""Attention mechanisms."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from beacon.logging import get_logger

logger = get_logger(__name__)

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """Initialize attention.

        Args:
            hidden_size: Hidden size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        output = self.out_proj(output)
        
        return output


class GroupedQueryAttention(nn.Module):
    """Grouped query attention (GQA)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        dropout: float = 0.1,
    ):
        """Initialize GQA.

        Args:
            hidden_size: Hidden size
            num_heads: Number of query heads
            num_key_value_heads: Number of key-value heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask

        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Project
        q = self.q_proj(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        # Expand k, v to match number of query heads
        repeat_factor = self.num_heads // self.num_key_value_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        output = self.out_proj(output)
        
        return output


class FlashAttention(nn.Module):
    """Flash Attention (if available)."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """Initialize Flash Attention.

        Args:
            hidden_size: Hidden size
            num_heads: Number of heads
            dropout: Dropout rate
        """
        super().__init__()
        if not FLASH_ATTN_AVAILABLE:
            logger.warning("Flash Attention not available, falling back to standard attention")
            self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
            self.use_flash = False
        else:
            self.attention = None
            self.use_flash = True
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
            attention_mask: Optional attention mask (unused in Flash Attention)

        Returns:
            torch.Tensor: Output tensor
        """
        if not self.use_flash:
            return self.attention(x, attention_mask)
        
        batch_size, seq_len, hidden_size = x.shape
        
        q = self.q_proj(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v = self.v_proj(x).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        
        output = flash_attn_func(q, k, v)
        output = output.reshape(batch_size, seq_len, hidden_size)
        output = self.out_proj(output)
        
        return output
