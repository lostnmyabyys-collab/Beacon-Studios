"""Access control policies."""

from enum import Enum


class AccessLevel(str, Enum):
    """Access levels."""

    PUBLIC = "public"
    USER = "user"
    ADMIN = "admin"


class AccessPolicy:
    """Manages access policies."""

    def __init__(self) -> None:
        """Initialize access policy."""
        self.policies: dict[str, AccessLevel] = {}

    def register_resource(self, resource_id: str, access_level: AccessLevel) -> None:
        """Register resource access level.

        Args:
            resource_id: Resource ID
            access_level: Required access level
        """
        self.policies[resource_id] = access_level

    def can_access(self, user_scopes: list[str], resource_id: str) -> bool:
        """Check if user can access resource.

        Args:
            user_scopes: User scopes
            resource_id: Resource ID

        Returns:
            bool: True if user can access
        """
        required_level = self.policies.get(resource_id, AccessLevel.PUBLIC)

        if required_level == AccessLevel.PUBLIC:
            return True
        if required_level == AccessLevel.USER:
            return len(user_scopes) > 0
        if required_level == AccessLevel.ADMIN:
            return "admin" in user_scopes

        return False
