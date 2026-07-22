"""Serialization utilities."""

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file.

    Args:
        path: Path to YAML file

    Returns:
        dict: Loaded data
    """
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save YAML file.

    Args:
        data: Data to save
        path: Path to YAML file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON file.

    Args:
        path: Path to JSON file

    Returns:
        dict: Loaded data
    """
    with open(path) as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save JSON file.

    Args:
        data: Data to save
        path: Path to JSON file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
