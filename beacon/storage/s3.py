"""S3 storage backend."""

import os

from beacon.logging import get_logger
from beacon.storage.base import StorageBackend

logger = get_logger(__name__)

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3Storage(StorageBackend):
    """S3-compatible storage backend."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
        prefix: str = "",
    ) -> None:
        """Initialize S3 storage.

        Args:
            bucket: S3 bucket name
            endpoint_url: Optional S3 endpoint URL
            access_key: Optional access key (uses env if not provided)
            secret_key: Optional secret key (uses env if not provided)
            region: AWS region
            prefix: Key prefix

        Raises:
            ImportError: If boto3 not installed
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 storage")

        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region

        # Get credentials
        access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        # Create S3 client
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        logger.info(f"Initialized S3 storage at s3://{bucket}/{prefix}")

    def _get_key(self, path: str) -> str:
        """Get full S3 key.

        Args:
            path: Relative path

        Returns:
            str: Full S3 key
        """
        if self.prefix:
            return f"{self.prefix}/{path}".lstrip("/")
        return path.lstrip("/")

    def exists(self, path: str) -> bool:
        """Check if path exists.

        Args:
            path: Path to check

        Returns:
            bool: True if path exists
        """
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._get_key(path))
            return True
        except self.client.exceptions.NoSuchKey:
            return False
        except Exception as e:
            logger.error(f"Error checking S3 path: {e}")
            return False

    def read(self, path: str) -> bytes:
        """Read file.

        Args:
            path: Path to file

        Returns:
            bytes: File contents
        """
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._get_key(path))
            return response["Body"].read()
        except self.client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"File not found: {path}")

    def write(self, path: str, data: bytes) -> None:
        """Write file.

        Args:
            path: Path to file
            data: Data to write
        """
        self.client.put_object(Bucket=self.bucket, Key=self._get_key(path), Body=data)
        logger.debug(f"Wrote S3 file: {path}")

    def delete(self, path: str) -> None:
        """Delete file.

        Args:
            path: Path to file
        """
        key = self._get_key(path)
        try:
            # Try to delete as object first
            self.client.delete_object(Bucket=self.bucket, Key=key)
            logger.debug(f"Deleted S3 file: {path}")
        except Exception as e:
            logger.error(f"Error deleting S3 file: {e}")

    def list_files(self, path: str = "") -> list[str]:
        """List files in directory.

        Args:
            path: Path to directory

        Returns:
            list: File paths (relative)
        """
        prefix = self._get_key(path)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        try:
            response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            files = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                # Remove prefix
                if prefix:
                    key = key[len(prefix):]
                if key:  # Skip empty keys
                    files.append(key)
            return files
        except Exception as e:
            logger.error(f"Error listing S3 files: {e}")
            return []

    def makedirs(self, path: str) -> None:
        """Create directory.

        Args:
            path: Path to directory
        """
        # S3 doesn't require explicit directory creation
        # Just log for consistency
        logger.debug(f"S3 directory created (virtual): {path}")
