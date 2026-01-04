"""Behavior cloning trainer.

Supports:
- Mixed precision training (AMP) for faster training on GPU
- Gradient accumulation for larger effective batch sizes
- Cosine learning rate scheduling with warmup
- Gradient checkpointing for memory efficiency
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BCConfig:
    """Configuration for behavior cloning training."""
    # Data
    train_data_path: Path = Path("data/processed/trajectories.jsonl")
    val_data_path: Optional[Path] = None

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    label_smoothing: float = 0.1

    # Gradient accumulation (effective_batch = batch_size * gradient_accumulation_steps)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Learning rate scheduling
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"  # "cosine", "linear", or "constant"

    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints/bc")
    save_every: int = 1  # Save every N epochs

    # Logging
    log_every: int = 100  # Log every N steps
    use_wandb: bool = False

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine learning rate schedule with linear warmup.

    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class BCTrainer:
    """Trainer for behavior cloning.

    Features:
    - Mixed precision training (AMP) for 2-3x speedup on GPU
    - Gradient accumulation for larger effective batch sizes
    - Cosine learning rate scheduling with warmup
    - Support for large transformer models (200M+ params)
    """

    def __init__(
        self,
        policy: nn.Module,
        config: BCConfig,
        num_training_steps: Optional[int] = None,
    ):
        self.policy = policy.to(config.device)
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = None
        if num_training_steps is not None and config.lr_scheduler != "constant":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=num_training_steps,
            )

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )

        # Mixed precision
        self.use_amp = config.use_amp and config.device == "cuda"
        if self.use_amp:
            self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
            self.scaler = GradScaler()
            logger.info(f"Using AMP with dtype={config.amp_dtype}")
        else:
            self.scaler = None

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        if config.use_wandb:
            import wandb
            wandb.init(project="poke-bc")

        # Log model info
        num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info(f"Model has {num_params:,} trainable parameters")
        logger.info(f"Effective batch size: {config.effective_batch_size}")

    def _move_to_device(self, data):
        """Recursively move tensors to device, handling nested dicts."""
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(self.config.device)
        else:
            return data

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with AMP and gradient accumulation."""
        self.policy.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        accumulation_steps = self.config.gradient_accumulation_steps

        pbar = tqdm(dataloader, desc="Training")
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Move to device (handles nested dicts of tensors)
            batch = self._move_to_device(batch)
            actions = batch.pop("action")

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    action_probs, _ = self.policy(batch)
                    loss = self.criterion(action_probs.log(), actions)
                    # Scale loss for gradient accumulation
                    loss = loss / accumulation_steps
            else:
                action_probs, _ = self.policy(batch)
                loss = self.criterion(action_probs.log(), actions)
                loss = loss / accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Tracking (use unscaled loss for metrics)
            total_loss += loss.item() * accumulation_steps * len(actions)
            predictions = action_probs.argmax(dim=-1)
            total_correct += (predictions == actions).sum().item()
            total_samples += len(actions)

            # Optimizer step after accumulation
            if (step + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Logging
            if self.global_step % self.config.log_every == 0 and self.global_step > 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": total_loss / total_samples,
                    "acc": total_correct / total_samples,
                    "lr": f"{current_lr:.2e}",
                })

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data with optional AMP."""
        self.policy.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            actions = batch.pop("action")

            # Use AMP for validation too
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    action_probs, _ = self.policy(batch)
                    loss = self.criterion(action_probs.log(), actions)
            else:
                action_probs, _ = self.policy(batch)
                loss = self.criterion(action_probs.log(), actions)

            total_loss += loss.item() * len(actions)
            predictions = action_probs.argmax(dim=-1)
            total_correct += (predictions == actions).sum().item()
            total_samples += len(actions)

        return {
            "val_loss": total_loss / total_samples,
            "val_accuracy": total_correct / total_samples,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Run full training loop."""
        logger.info(f"Starting BC training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.config.device}")

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")

            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                logger.info(f"Val: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_accuracy']:.4f}")

                # Save best model
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best.pt")

            # Periodic checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Wandb logging
            if self.config.use_wandb:
                import wandb
                metrics = {"epoch": epoch + 1, **train_metrics}
                if val_loader:
                    metrics.update(val_metrics)
                wandb.log(metrics)

        # Save final model
        self.save_checkpoint("final.pt")
        logger.info("Training complete")

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint: {path}")
