"""JWT token management."""

from datetime import datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from pydantic import BaseModel

from beacon.config.settings import get_settings
from beacon.logging import get_logger

logger = get_logger(__name__)


class TokenPayload(BaseModel):
    """JWT token payload."""

    user_id: str
    email: str
    scopes: list[str] = []
    exp: datetime | None = None


class TokenManager:
    """Manages JWT tokens."""

    def __init__(self) -> None:
        """Initialize token manager."""
        self.settings = get_settings()
        self.secret_key = self.settings.jwt_secret_key
        self.algorithm = self.settings.jwt_algorithm
        self.expiration_hours = self.settings.jwt_expiration_hours

    def create_token(self, user_id: str, email: str, scopes: list[str] | None = None) -> str:
        """Create JWT token.

        Args:
            user_id: User ID
            email: User email
            scopes: Optional list of scopes

        Returns:
            str: JWT token
        """
        scopes = scopes or []
        expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)

        payload: dict[str, Any] = {
            "user_id": user_id,
            "email": email,
            "scopes": scopes,
            "exp": expire,
        }

        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Created token for user: {user_id}")
        return encoded_jwt

    def verify_token(self, token: str) -> TokenPayload:
        """Verify JWT token.

        Args:
            token: JWT token

        Returns:
            TokenPayload: Token payload

        Raises:
            JWTError: If token invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("user_id")
            email: str = payload.get("email")
            scopes: list[str] = payload.get("scopes", [])

            if not user_id:
                raise JWTError("Invalid token: missing user_id")

            logger.debug(f"Token verified for user: {user_id}")
            return TokenPayload(user_id=user_id, email=email, scopes=scopes)
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise

    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired.

        Args:
            token: JWT token

        Returns:
            bool: True if expired
        """
        try:
            self.verify_token(token)
            return False
        except JWTError:
            return True
