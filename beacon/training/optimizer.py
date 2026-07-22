"""Optimizer factory."""

from typing import Any

import torch


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create optimizer.

    Args:
        model: Model to optimize
        optimizer_name: Optimizer name (adamw, sgd, adam)
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional arguments

    Returns:
        torch.optim.Optimizer: Optimizer instance
    """
    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
