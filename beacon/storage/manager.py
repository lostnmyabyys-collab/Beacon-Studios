"""Storage manager for switching between backends."""

from beacon.config.schema import StorageConfig
from beacon.logging import get_logger
from beacon.storage.base import StorageBackend
from beacon.storage.local import LocalStorage
from beacon.storage.s3 import S3Storage

logger = get_logger(__name__)


class StorageManager:
    """Manages storage backend selection and access."""

    def __init__(self, config: StorageConfig) -> None:
        """Initialize storage manager.

        Args:
            config: Storage configuration
        """
        self.config = config
        self.backend = self._create_backend()
        logger.info(f"Storage backend: {config.backend}")

    def _create_backend(self) -> StorageBackend:
        """Create appropriate storage backend.

        Returns:
            StorageBackend: Storage backend instance

        Raises:
            ValueError: If backend type unknown
        """
        if self.config.backend == "local":
            return LocalStorage(self.config.local_path)
        elif self.config.backend == "s3":
            return S3Storage(
                bucket=self.config.s3_bucket,
                endpoint_url=self.config.s3_endpoint,
                region=self.config.s3_region,
                prefix=self.config.s3_prefix,
            )
        else:
            raise ValueError(f"Unknown storage backend: {self.config.backend}")

    def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Path to check

        Returns:
            bool: True if path exists
        """
        return self.backend.exists(path)

    def read(self, path: str) -> bytes:
        """Read file.

        Args:
            path: Path to file

        Returns:
            bytes: File contents
        """
        return self.backend.read(path)

    def write(self, path: str, data: bytes) -> None:
        """Write file.

        Args:
            path: Path to file
            data: Data to write
        """
        self.backend.write(path, data)

    def delete(self, path: str) -> None:
        """Delete file.

        Args:
            path: Path to file
        """
        self.backend.delete(path)

    def list_files(self, path: str = "") -> list[str]:
        """List files in directory.

        Args:
            path: Path to directory

        Returns:
            list: File paths
        """
        return self.backend.list_files(path)

    def makedirs(self, path: str) -> None:
        """Create directory.

        Args:
            path: Path to directory
        """
        self.backend.makedirs(path)
