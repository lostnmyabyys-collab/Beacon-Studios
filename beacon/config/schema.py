"""Configuration schema definitions."""

from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Storage configuration."""

    backend: str = Field(default="local", description="Storage backend (local, s3)")
    local_path: str = Field(default="./data", description="Local storage path")
    s3_endpoint: str | None = Field(default=None, description="S3 endpoint URL")
    s3_bucket: str = Field(default="beacon-studios", description="S3 bucket name")
    s3_region: str = Field(default="us-east-1", description="S3 region")
    s3_prefix: str = Field(default="", description="S3 key prefix")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "backend": "local",
                "local_path": "./data",
            }
        }


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    device: str = Field(default="cuda", description="Device (cuda, cpu, mps)")
    max_batch_size: int = Field(default=32, description="Maximum batch size")
    max_context_length: int = Field(default=4096, description="Maximum context length")
    streaming_enabled: bool = Field(default=True, description="Enable streaming")
    cache_size_mb: int = Field(default=2048, description="KV cache size in MB")
    quantization: str | None = Field(default=None, description="Quantization method")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "device": "cuda",
                "max_batch_size": 32,
                "max_context_length": 4096,
            }
        }


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(default=32, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")
    learning_rate: float = Field(default=5e-4, description="Learning rate")
    weight_decay: float = Field(default=0.01, description="Weight decay")
    warmup_steps: int = Field(default=500, description="Warmup steps")
    max_steps: int = Field(default=100000, description="Maximum training steps")
    eval_steps: int = Field(default=1000, description="Evaluation steps")
    save_steps: int = Field(default=1000, description="Checkpoint save steps")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm")
    seed: int = Field(default=42, description="Random seed")
    distributed: bool = Field(default=False, description="Enable distributed training")
    mixed_precision: str = Field(default="no", description="Mixed precision (no, fp16, bf16)")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "batch_size": 32,
                "learning_rate": 5e-4,
                "max_steps": 100000,
            }
        }


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    model_name: str = Field(description="Model name")
    model_type: str = Field(description="Model type (vx, neo, cz)")
    hidden_size: int = Field(default=768, description="Hidden size")
    num_attention_heads: int = Field(default=12, description="Number of attention heads")
    num_hidden_layers: int = Field(default=12, description="Number of hidden layers")
    intermediate_size: int = Field(default=3072, description="Intermediate size")
    vocab_size: int = Field(default=50257, description="Vocabulary size")
    max_position_embeddings: int = Field(default=2048, description="Max position embeddings")
    dropout_rate: float = Field(default=0.1, description="Dropout rate")
    use_cache: bool = Field(default=True, description="Use KV cache")
    use_flash_attention: bool = Field(default=True, description="Use Flash Attention")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "model_name": "vx-1",
                "model_type": "vx",
                "hidden_size": 768,
                "num_attention_heads": 12,
            }
        }
