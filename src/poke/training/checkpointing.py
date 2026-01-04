"""Checkpoint management utilities."""
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import torch

logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    epoch: int
    step: int
    timestamp: str
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        return cls(**data)


class CheckpointManager:
    """Manages model checkpoints with versioning and metadata."""

    def __init__(
        self,
        checkpoint_dir: Path,
        max_to_keep: int = 5,
        keep_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.keep_best = keep_best

        self.metadata_file = self.checkpoint_dir / "checkpoints.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load checkpoint metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                data = json.load(f)
                self.checkpoints = [
                    CheckpointMetadata.from_dict(c)
                    for c in data.get("checkpoints", [])
                ]
                self.best_checkpoint = data.get("best_checkpoint")
        else:
            self.checkpoints: List[CheckpointMetadata] = []
            self.best_checkpoint: Optional[str] = None

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to file."""
        data = {
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "best_checkpoint": self.best_checkpoint,
        }
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: CheckpointMetadata,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            metadata: Checkpoint metadata
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        filename = f"checkpoint_epoch{metadata.epoch}_step{metadata.step}.pt"
        path = self.checkpoint_dir / filename

        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata.to_dict(),
        }, path)

        # Update metadata
        self.checkpoints.append(metadata)

        if is_best or self.best_checkpoint is None:
            self.best_checkpoint = filename
            # Also save as "best.pt" for easy access
            torch.save(
                torch.load(path),
                self.checkpoint_dir / "best.pt"
            )

        # Cleanup old checkpoints
        self._cleanup()

        self._save_metadata()
        logger.info(f"Saved checkpoint: {path}")

        return path

    def load(
        self,
        path: Optional[Path] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            path: Specific checkpoint to load
            load_best: If True, load the best checkpoint

        Returns:
            Checkpoint dict with model state and metadata
        """
        if load_best:
            if self.best_checkpoint:
                path = self.checkpoint_dir / self.best_checkpoint
            else:
                path = self.checkpoint_dir / "best.pt"

        if path is None:
            # Load latest
            if self.checkpoints:
                latest = self.checkpoints[-1]
                path = self.checkpoint_dir / f"checkpoint_epoch{latest.epoch}_step{latest.step}.pt"
            else:
                raise FileNotFoundError("No checkpoints found")

        checkpoint = torch.load(path, map_location="cpu")
        logger.info(f"Loaded checkpoint: {path}")

        return checkpoint

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond max_to_keep."""
        if len(self.checkpoints) <= self.max_to_keep:
            return

        # Sort by validation loss (or train loss if no val)
        def get_loss(c):
            return c.val_loss if c.val_loss is not None else c.train_loss

        sorted_checkpoints = sorted(self.checkpoints, key=get_loss)

        # Keep best and most recent
        to_keep = set()
        if self.keep_best:
            to_keep.add(sorted_checkpoints[0].step)

        # Keep most recent
        for c in self.checkpoints[-self.max_to_keep:]:
            to_keep.add(c.step)

        # Remove others
        for c in self.checkpoints:
            if c.step not in to_keep:
                filename = f"checkpoint_epoch{c.epoch}_step{c.step}.pt"
                path = self.checkpoint_dir / filename
                if path.exists():
                    path.unlink()
                    logger.debug(f"Removed old checkpoint: {path}")

        self.checkpoints = [c for c in self.checkpoints if c.step in to_keep]

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        return self.checkpoints.copy()

    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self.best_checkpoint:
            return self.checkpoint_dir / self.best_checkpoint
        return None
