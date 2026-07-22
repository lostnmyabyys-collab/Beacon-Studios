"""Learning rate schedulers."""

import math

import torch


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "linear",
    total_steps: int = 100000,
    warmup_steps: int = 500,
    **kwargs,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Scheduler name (linear, cosine, warmup_linear)
        total_steps: Total training steps
        warmup_steps: Warmup steps
        **kwargs: Additional arguments

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Scheduler instance
    """
    if scheduler_name.lower() == "linear":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_name.lower() == "cosine":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_name.lower() == "constant_with_warmup":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
