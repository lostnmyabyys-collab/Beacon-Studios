"""Validation utilities."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError


def validate_path(path: str | Path, must_exist: bool = False) -> Path:
    """Validate and normalize path.

    Args:
        path: Path to validate
        must_exist: If True, path must exist

    Returns:
        Path: Normalized path

    Raises:
        ValueError: If validation fails
    """
    p = Path(path).expanduser().resolve()
    if must_exist and not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    return p


def validate_config(config: dict[str, Any], schema: type) -> Any:
    """Validate configuration against schema.

    Args:
        config: Configuration dictionary
        schema: Pydantic model class

    Returns:
        Validated configuration object

    Raises:
        ValidationError: If validation fails
    """
    try:
        return schema(**config)
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
