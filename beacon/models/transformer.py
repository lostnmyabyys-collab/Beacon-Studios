"""Transformer decoder model."""

from typing import Optional

import torch
from torch import nn

from beacon.logging import get_logger
from beacon.models.attention import FlashAttention, MultiHeadAttention
from beacon.models.config import ModelConfig

logger = get_logger(__name__)


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: ModelConfig):
        """Initialize layer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Attention
        if config.use_flash_attention:
            self.self_attn = FlashAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )
        else:
            self.self_attn = MultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )
        
        # Feed-forward
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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
        # Self-attention with residual
        attn_output = self.self_attn(x, attention_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder model."""

    def __init__(self, config: ModelConfig):
        """Initialize decoder.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        
        # Layers
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Output layer
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        logger.info(
            f"Initialized TransformerDecoder "
            f"(layers={config.num_hidden_layers}, "
            f"heads={config.num_attention_heads}, "
            f"hidden={config.hidden_size})"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            output_hidden_states: Return hidden states

        Returns:
            dict: Model outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids)
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Process through layers
        hidden_states = [x] if output_hidden_states else None
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            if output_hidden_states:
                hidden_states.append(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Logits
        logits = self.lm_head(x)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text.

        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            eos_token_id: End of sequence token ID

        Returns:
            torch.Tensor: Generated token IDs
        """
        device = input_ids.device
        current_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get next token predictions
            with torch.no_grad():
                outputs = self.forward(current_ids)
                next_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return current_ids
