"""Checkpoint loader."""

import json
from typing import Any

import torch

from beacon.logging import get_logger
from beacon.storage.manager import StorageManager

logger = get_logger(__name__)


class CheckpointLoader:
    """Loads training checkpoints."""

    def __init__(self, storage: StorageManager) -> None:
        """Initialize checkpoint loader.

        Args:
            storage: Storage manager
        """
        self.storage = storage
        logger.info("Initialized checkpoint loader")

    def load(
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
            dict: Checkpoint data including metadata
        """
        try:
            # Load checkpoint
            checkpoint_data = self.storage.read(f"{checkpoint_path}/checkpoint.pt")
            checkpoint = torch.load(
                checkpoint_data,
                map_location="cpu",
                weights_only=False,
            )

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model state from checkpoint")

            # Load optimizer state if provided
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info(f"Loaded optimizer state from checkpoint")

            # Load metadata
            metadata_data = self.storage.read(f"{checkpoint_path}/metadata.json")
            metadata = json.loads(metadata_data.decode())

            return {
                "step": checkpoint.get("step", 0),
                "metrics": checkpoint.get("metrics", {}),
                "metadata": metadata,
            }
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def get_checkpoint_metadata(self, checkpoint_path: str) -> dict[str, Any]:
        """Get checkpoint metadata without loading full state.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            dict: Checkpoint metadata
        """
        try:
            metadata_data = self.storage.read(f"{checkpoint_path}/metadata.json")
            return json.loads(metadata_data.decode())
        except Exception as e:
            logger.error(f"Failed to load checkpoint metadata: {e}")
            raise
