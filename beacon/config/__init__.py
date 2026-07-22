"""Configuration system for Beacon platform."""

from beacon.config.loader import load_config
from beacon.config.schema import RuntimeConfig, StorageConfig, TrainingConfig
from beacon.config.settings import get_settings

__all__ = [
    "load_config",
    "get_settings",
    "RuntimeConfig",
    "StorageConfig",
    "TrainingConfig",
]
