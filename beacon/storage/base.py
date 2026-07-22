"""Base storage interface."""

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Path to check

        Returns:
            bool: True if path exists
        """

    @abstractmethod
    def read(self, path: str) -> bytes:
        """Read file.

        Args:
            path: Path to file

        Returns:
            bytes: File contents
        """

    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write file.

        Args:
            path: Path to file
            data: Data to write
        """

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete file.

        Args:
            path: Path to file
        """

    @abstractmethod
    def list_files(self, path: str) -> list[str]:
        """List files in directory.

        Args:
            path: Path to directory

        Returns:
            list: File paths
        """

    @abstractmethod
    def makedirs(self, path: str) -> None:
        """Create directory.

        Args:
            path: Path to directory
        """
