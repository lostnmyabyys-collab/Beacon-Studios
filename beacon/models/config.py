"""Model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    vocab_size: int = 50257
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Advanced features
    use_flash_attention: bool = True
    use_grouped_query_attention: bool = False
    num_key_value_heads: int | None = None
    rope: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        if self.use_grouped_query_attention and self.num_key_value_heads is None:
            self.num_key_value_heads = max(1, self.num_attention_heads // 8)
