"""Application settings and environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Environment
    environment: str = "development"
    debug: bool = True

    # Device
    device: str = "cuda"  # cuda, cpu, mps
    cuda_visible_devices: str | None = None
    flash_attention: bool = True
    mixed_precision: bool = True

    # Storage
    storage_backend: str = "local"  # local, s3
    local_storage_path: str = "./data"
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_bucket_name: str = "beacon-studios"
    s3_region_name: str = "us-east-1"

    # Database
    database_url: str = "postgresql://localhost:5432/beacon"

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_log_level: str = "info"
    api_reload: bool = False

    # WebSocket
    ws_host: str = "0.0.0.0"
    ws_port: int = 8001

    # Authentication
    auth_enabled: bool = True
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Training
    training_batch_size: int = 32
    training_gradient_accumulation_steps: int = 1
    training_learning_rate: float = 5e-4
    training_weight_decay: float = 0.01
    training_warmup_steps: int = 500
    training_max_steps: int = 100000
    training_eval_steps: int = 1000
    training_save_steps: int = 1000
    training_log_steps: int = 100

    # Distributed Training
    distributed_enabled: bool = False
    distributed_backend: str = "nccl"
    distributed_master_addr: str = "localhost"
    distributed_master_port: int = 29500

    # Inference
    inference_max_batch_size: int = 32
    inference_max_context_length: int = 4096
    inference_streaming_enabled: bool = True
    inference_cache_size_mb: int = 2048

    # Model Registry
    model_registry_path: str = "./models"
    model_cache_dir: str = "./model_cache"

    # Checkpoints
    checkpoint_directory: str = "./checkpoints"
    checkpoint_keep_last_n: int = 3

    # Logging
    log_level: str = "info"
    log_file: str | None = None
    telemetry_enabled: bool = True

    # Desktop Integration
    desktop_integration_enabled: bool = False
    desktop_ipc_port: int = 19999

    # Evaluation
    evaluation_benchmark_models: str = "vx-1,neo-1,cz-1"
    evaluation_save_results: bool = True

    # Feature Flags
    feature_quantization: bool = True
    feature_onnx_export: bool = True
    feature_gguf_export: bool = False
    feature_model_merging: bool = False

    # Timeouts
    request_timeout_seconds: int = 300
    training_timeout_seconds: int = 3600

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (singleton).

    Returns:
        Settings: Application settings
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
