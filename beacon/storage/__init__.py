"""Storage abstraction layer."""

from beacon.storage.base import StorageBackend
from beacon.storage.local import LocalStorage
from beacon.storage.manager import StorageManager
from beacon.storage.s3 import S3Storage

__all__ = [
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "StorageManager",
]
