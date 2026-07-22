"""Validation utilities."""

from typing import Any

import torch

from beacon.logging import get_logger

logger = get_logger(__name__)


class Validator:
    """Validation utilities."""

    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """Compute perplexity from loss.

        Args:
            loss: Cross-entropy loss

        Returns:
            float: Perplexity
        """
        return 2.718281828 ** loss

    @staticmethod
    def validate(
        model: torch.nn.Module,
        dataloader,
        device: torch.device,
    ) -> dict[str, Any]:
        """Run validation.

        Args:
            model: Model to validate
            dataloader: Validation dataloader
            device: Device

        Returns:
            dict: Validation metrics
        """
        model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)
                logits = outputs["logits"]

                # Compute loss
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                )

                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        perplexity = Validator.compute_perplexity(avg_loss)

        logger.info(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
