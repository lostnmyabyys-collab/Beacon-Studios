"""Distributed training support."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from beacon.logging import get_logger

logger = get_logger(__name__)


class DistributedTrainer:
    """Handles distributed training setup."""

    @staticmethod
    def setup() -> None:
        """Setup distributed training."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "29500")
            
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size,
            )
            logger.info(
                f"Initialized distributed training "
                f"(rank={rank}, world_size={world_size})"
            )

    @staticmethod
    def cleanup() -> None:
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed process group")

    @staticmethod
    def wrap_model(model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training.

        Args:
            model: Model to wrap

        Returns:
            torch.nn.Module: Wrapped model
        """
        if dist.is_initialized():
            model = DDP(model)
            logger.info("Wrapped model with DistributedDataParallel")
        return model
