"""Configuration loader."""

from pathlib import Path
from typing import Any

from beacon.config.schema import ModelConfig, RuntimeConfig, StorageConfig, TrainingConfig
from beacon.utils.serialization import load_yaml


def load_config(path: str | Path, config_type: type) -> Any:
    """Load and validate configuration from YAML file.

    Args:
        path: Path to configuration file
        config_type: Configuration schema class

    Returns:
        Validated configuration object

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If configuration invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    config_data = load_yaml(path)
    return config_type(**config_data)


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load training configuration.

    Args:
        path: Path to training config file

    Returns:
        TrainingConfig: Training configuration
    """
    return load_config(path, TrainingConfig)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load runtime configuration.

    Args:
        path: Path to runtime config file

    Returns:
        RuntimeConfig: Runtime configuration
    """
    return load_config(path, RuntimeConfig)


def load_storage_config(path: str | Path) -> StorageConfig:
    """Load storage configuration.

    Args:
        path: Path to storage config file

    Returns:
        StorageConfig: Storage configuration
    """
    return load_config(path, StorageConfig)


def load_model_config(path: str | Path) -> ModelConfig:
    """Load model configuration.

    Args:
        path: Path to model config file

    Returns:
        ModelConfig: Model configuration
    """
    return load_config(path, ModelConfig)
