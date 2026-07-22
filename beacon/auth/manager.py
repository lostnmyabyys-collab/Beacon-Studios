"""Authentication manager."""

import secrets
from typing import Any

from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from beacon.auth.tokens import TokenManager
from beacon.logging import get_logger

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(BaseModel):
    """User model."""

    user_id: str
    email: EmailStr
    username: str
    is_active: bool = True
    is_admin: bool = False
    metadata: dict[str, Any] = {}


class AuthManager:
    """Manages user authentication."""

    def __init__(self) -> None:
        """Initialize auth manager."""
        self.token_manager = TokenManager()
        # In-memory user store (replace with database in production)
        self.users: dict[str, dict[str, Any]] = {}
        logger.info("Initialized authentication manager")

    def hash_password(self, password: str) -> str:
        """Hash password.

        Args:
            password: Plain text password

        Returns:
            str: Hashed password
        """
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            bool: True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)

    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        is_admin: bool = False,
    ) -> User:
        """Create user.

        Args:
            email: User email
            username: Username
            password: Password
            is_admin: Is admin user

        Returns:
            User: Created user

        Raises:
            ValueError: If email already exists
        """
        # Check if email exists
        for user in self.users.values():
            if user["email"] == email:
                raise ValueError(f"Email already exists: {email}")

        user_id = secrets.token_urlsafe(16)
        hashed_password = self.hash_password(password)

        self.users[user_id] = {
            "user_id": user_id,
            "email": email,
            "username": username,
            "password": hashed_password,
            "is_active": True,
            "is_admin": is_admin,
            "metadata": {},
        }

        logger.info(f"Created user: {email}")
        return User(
            user_id=user_id,
            email=email,
            username=username,
            is_admin=is_admin,
        )

    def authenticate(self, email: str, password: str) -> tuple[User, str]:
        """Authenticate user and get token.

        Args:
            email: User email
            password: Password

        Returns:
            tuple: (User, token)

        Raises:
            ValueError: If authentication failed
        """
        # Find user by email
        user_data = None
        for user in self.users.values():
            if user["email"] == email:
                user_data = user
                break

        if not user_data:
            raise ValueError(f"User not found: {email}")

        # Verify password
        if not self.verify_password(password, user_data["password"]):
            raise ValueError("Invalid password")

        # Create token
        token = self.token_manager.create_token(
            user_id=user_data["user_id"],
            email=user_data["email"],
            scopes=["admin"] if user_data["is_admin"] else [],
        )

        logger.info(f"User authenticated: {email}")
        return (
            User(
                user_id=user_data["user_id"],
                email=user_data["email"],
                username=user_data["username"],
                is_admin=user_data["is_admin"],
            ),
            token,
        )

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User: User object or None
        """
        user_data = self.users.get(user_id)
        if not user_data:
            return None

        return User(
            user_id=user_data["user_id"],
            email=user_data["email"],
            username=user_data["username"],
            is_admin=user_data["is_admin"],
        )

    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email.

        Args:
            email: User email

        Returns:
            User: User object or None
        """
        for user in self.users.values():
            if user["email"] == email:
                return User(
                    user_id=user["user_id"],
                    email=user["email"],
                    username=user["username"],
                    is_admin=user["is_admin"],
                )
        return None
