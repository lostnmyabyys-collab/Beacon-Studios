"""Checkpoint saver."""

import json
from pathlib import Path
from typing import Any

import torch

from beacon.logging import get_logger
from beacon.storage.manager import StorageManager

logger = get_logger(__name__)


class CheckpointSaver:
    """Saves training checkpoints."""

    def __init__(self, storage: StorageManager, checkpoint_dir: str = "checkpoints") -> None:
        """Initialize checkpoint saver.

        Args:
            storage: Storage manager
            checkpoint_dir: Checkpoint directory
        """
        self.storage = storage
        self.checkpoint_dir = checkpoint_dir
        self.storage.makedirs(checkpoint_dir)
        logger.info(f"Initialized checkpoint saver at {checkpoint_dir}")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> str:
        """Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            step: Training step
            metrics: Optional metrics
            name: Optional checkpoint name

        Returns:
            str: Checkpoint path
        """
        checkpoint_name = name or f"checkpoint-{step}"
        checkpoint_path = f"{self.checkpoint_dir}/{checkpoint_name}"

        try:
            # Prepare checkpoint data
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "metrics": metrics or {},
            }

            # Save to bytes
            checkpoint_bytes = json.dumps(
                {
                    k: v if not isinstance(v, (dict, torch.Tensor)) else "<tensor>"
                    for k, v in checkpoint.items()
                },
                default=str,
            ).encode()

            # Also save actual checkpoint using torch
            import io
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            checkpoint_bytes = buffer.getvalue()

            self.storage.write(f"{checkpoint_path}/checkpoint.pt", checkpoint_bytes)

            # Save metadata
            metadata = {
                "step": step,
                "metrics": metrics or {},
            }
            self.storage.write(
                f"{checkpoint_path}/metadata.json",
                json.dumps(metadata).encode(),
            )

            logger.info(f"Saved checkpoint: {checkpoint_path} (step {step})")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """Clean up old checkpoints.

        Args:
            keep_last_n: Number of checkpoints to keep
        """
        try:
            checkpoints = self.storage.list_files(self.checkpoint_dir)
            # Sort by name (assuming step-based naming)
            checkpoints.sort()

            # Keep only last N
            if len(checkpoints) > keep_last_n:
                for checkpoint in checkpoints[:-keep_last_n]:
                    self.storage.delete(f"{self.checkpoint_dir}/{checkpoint}")
                    logger.info(f"Deleted old checkpoint: {checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoints: {e}")
