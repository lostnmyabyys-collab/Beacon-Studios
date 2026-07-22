"""Local filesystem storage backend."""

from pathlib import Path

from beacon.logging import get_logger
from beacon.storage.base import StorageBackend

logger = get_logger(__name__)


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str = "./data") -> None:
        """Initialize local storage.

        Args:
            base_path: Base path for storage
        """
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized local storage at {self.base_path}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve full path.

        Args:
            path: Relative path

        Returns:
            Path: Resolved absolute path
        """
        full_path = (self.base_path / path).resolve()
        # Security: ensure path is within base_path
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError(f"Path outside base path: {path}")
        return full_path

    def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Path to check

        Returns:
            bool: True if path exists
        """
        return self._resolve_path(path).exists()

    def read(self, path: str) -> bytes:
        """Read file.

        Args:
            path: Path to file

        Returns:
            bytes: File contents

        Raises:
            FileNotFoundError: If file not found
        """
        full_path = self._resolve_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return full_path.read_bytes()

    def write(self, path: str, data: bytes) -> None:
        """Write file.

        Args:
            path: Path to file
            data: Data to write
        """
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)
        logger.debug(f"Wrote file: {path}")

    def delete(self, path: str) -> None:
        """Delete file.

        Args:
            path: Path to file
        """
        full_path = self._resolve_path(path)
        if full_path.exists():
            if full_path.is_dir():
                import shutil
                shutil.rmtree(full_path)
            else:
                full_path.unlink()
            logger.debug(f"Deleted: {path}")

    def list_files(self, path: str = "") -> list[str]:
        """List files in directory.

        Args:
            path: Path to directory

        Returns:
            list: File paths (relative)
        """
        full_path = self._resolve_path(path) if path else self.base_path
        if not full_path.is_dir():
            return []
        
        files = []
        for item in full_path.iterdir():
            rel_path = item.relative_to(self.base_path)
            files.append(str(rel_path))
        return files

    def makedirs(self, path: str) -> None:
        """Create directory.

        Args:
            path: Path to directory
        """
        full_path = self._resolve_path(path)
        full_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")
