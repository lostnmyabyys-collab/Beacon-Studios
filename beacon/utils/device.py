"""Device utility functions."""

import torch

from beacon.logging import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """Get optimal device for computation.

    Returns:
        torch.device: Device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("Using Metal Performance Shaders (MPS)")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")
