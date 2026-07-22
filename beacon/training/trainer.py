"""Main training loop."""

from typing import Any, Optional

import torch

from beacon.checkpoints.manager import CheckpointManager
from beacon.config.schema import TrainingConfig
from beacon.logging import get_logger, MetricsCollector
from beacon.models.transformer import TransformerDecoder
from beacon.training.optimizer import create_optimizer
from beacon.training.scheduler import create_scheduler
from beacon.training.validation import Validator
from beacon.utils.device import get_device

logger = get_logger(__name__)


class Trainer:
    """Model trainer."""

    def __init__(
        self,
        model: TransformerDecoder,
        config: TrainingConfig,
        checkpoint_manager: CheckpointManager,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            checkpoint_manager: Checkpoint manager
            device: Device (auto-detected if None)
        """
        self.model = model
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.device = device or get_device()
        
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            total_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
        )
        
        # Metrics
        self.metrics_collector = MetricsCollector()
        
        logger.info(
            f"Initialized trainer "
            f"(lr={config.learning_rate}, "
            f"batch_size={config.batch_size}, "
            f"max_steps={config.max_steps})"
        )

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> float:
        """Single training step.

        Args:
            batch: Training batch

        Returns:
            float: Loss value
        """
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids)
        logits = outputs["logits"]
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def train(
        self,
        train_dataloader,
        val_dataloader: Optional[Any] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> dict[str, Any]:
        """Training loop.

        Args:
            train_dataloader: Training dataloader
            val_dataloader: Optional validation dataloader
            resume_from_checkpoint: Optional checkpoint to resume from

        Returns:
            dict: Training results
        """
        start_step = 0
        
        # Resume from checkpoint
        if resume_from_checkpoint:
            try:
                checkpoint_data = self.checkpoint_manager.load_checkpoint(
                    resume_from_checkpoint,
                    self.model,
                    self.optimizer,
                )
                start_step = checkpoint_data["step"]
                logger.info(f"Resumed from checkpoint at step {start_step}")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
        
        # Training loop
        total_loss = 0.0
        best_val_loss = float("inf")
        
        for step in range(start_step, self.config.max_steps):
            for batch in train_dataloader:
                # Training step
                loss = self.train_step(batch)
                total_loss += loss
                
                # Log
                if (step + 1) % self.config.log_steps == 0:
                    avg_loss = total_loss / self.config.log_steps
                    logger.info(
                        f"Step {step + 1}/{self.config.max_steps} - "
                        f"Loss: {avg_loss:.4f}"
                    )
                    self.metrics_collector.record("train/loss", avg_loss)
                    total_loss = 0.0
                
                # Validate
                if (step + 1) % self.config.eval_steps == 0 and val_dataloader:
                    val_metrics = Validator.validate(
                        self.model,
                        val_dataloader,
                        self.device,
                    )
                    
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        self.metrics_collector.record(
                            "val/loss",
                            val_metrics["loss"],
                        )
                
                # Save checkpoint
                if (step + 1) % self.config.save_steps == 0:
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        self.model,
                        self.optimizer,
                        step + 1,
                        metrics={"loss": loss},
                    )
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                step += 1
        
        logger.info("Training completed")
        return {
            "metrics": self.metrics_collector.export(),
            "best_val_loss": best_val_loss,
        }
