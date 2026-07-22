"""Version information."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class VersionInfo:
    """Version information."""

    major: int
    minor: int
    patch: int
    pre_release: str = ""
    build_metadata: str = ""
    created_at: datetime | None = None

    @property
    def version_string(self) -> str:
        """Get version as string.

        Returns:
            str: Version string (e.g., "1.0.0")
        """
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build_metadata:
            version += f"+{self.build_metadata}"
        return version

    @classmethod
    def from_string(cls, version_str: str) -> "VersionInfo":
        """Parse version string.

        Args:
            version_str: Version string (e.g., "1.0.0")

        Returns:
            VersionInfo: Parsed version

        Raises:
            ValueError: If version string invalid
        """
        parts = version_str.split("+", 1)
        build_metadata = parts[1] if len(parts) > 1 else ""

        parts = parts[0].split("-", 1)
        pre_release = parts[1] if len(parts) > 1 else ""

        version_parts = parts[0].split(".")
        if len(version_parts) < 3:
            raise ValueError(f"Invalid version string: {version_str}")

        try:
            major = int(version_parts[0])
            minor = int(version_parts[1])
            patch = int(version_parts[2])
        except ValueError as e:
            raise ValueError(f"Invalid version string: {version_str}") from e

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            pre_release=pre_release,
            build_metadata=build_metadata,
        )

    def __lt__(self, other: "VersionInfo") -> bool:
        """Compare versions.

        Args:
            other: Other version

        Returns:
            bool: True if self < other
        """
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch
