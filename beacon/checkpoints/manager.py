"""Checkpoint manager."""

from typing import Any

import torch

from beacon.checkpoints.loader import CheckpointLoader
from beacon.checkpoints.saver import CheckpointSaver
from beacon.logging import get_logger
from beacon.storage.manager import StorageManager

logger = get_logger(__name__)


class CheckpointManager:
    """Manages training checkpoints."""

    def __init__(
        self,
        storage: StorageManager,
        checkpoint_dir: str = "checkpoints",
        keep_last_n: int = 3,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            storage: Storage manager
            checkpoint_dir: Checkpoint directory
            keep_last_n: Number of checkpoints to keep
        """
        self.storage = storage
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.saver = CheckpointSaver(storage, checkpoint_dir)
        self.loader = CheckpointLoader(storage)
        logger.info(f"Initialized checkpoint manager (keep_last_n={keep_last_n})")

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> str:
        """Save checkpoint and cleanup old ones.

        Args:
            model: Model to save
            optimizer: Optimizer state
            step: Training step
            metrics: Optional metrics
            name: Optional checkpoint name

        Returns:
            str: Checkpoint path
        """
        checkpoint_path = self.saver.save(model, optimizer, step, metrics, name)
        self.saver.cleanup_old_checkpoints(self.keep_last_n)
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optional optimizer to load state

        Returns:
            dict: Checkpoint data
        """
        return self.loader.load(checkpoint_path, model, optimizer)

    def get_checkpoint_metadata(self, checkpoint_path: str) -> dict[str, Any]:
        """Get checkpoint metadata.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            dict: Checkpoint metadata
        """
        return self.loader.get_checkpoint_metadata(checkpoint_path)

    def list_checkpoints(self) -> list[str]:
        """List all checkpoints.

        Returns:
            list: Checkpoint paths
        """
        try:
            return self.storage.list_files(self.checkpoint_dir)
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def delete_checkpoint(self, checkpoint_path: str) -> None:
        """Delete checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        try:
            self.storage.delete(checkpoint_path)
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            raise
