"""Beacon Studios AI Platform.

A production-grade, modular AI model ecosystem supporting multiple model families
with advanced training, inference, and evaluation capabilities.
"""

__version__ = "0.1.0"
__author__ = "Beacon Studios"
__email__ = "info@beacon.studios"

from beacon.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "__version__",
    "logger",
]
