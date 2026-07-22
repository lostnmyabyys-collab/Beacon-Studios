"""Main package."""

__version__ = "1.0.0"
__author__ = "Beacon Studios"
__description__ = "High-performance LLM platform"

from beacon.api.server import app

__all__ = ["app"]
