COMPLETED

# PR-017: BC Checkpointing and Logging

## Dependencies
- PR-016 (Behavior Cloning Training Loop)

## Overview
Enhance checkpointing with versioning, metrics tracking, and model comparison utilities. Enable easy resumption and model selection.

## Tech Choices
- **Checkpoint Format:** PyTorch with metadata JSON
- **Tracking:** CSV + optional Weights & Biases
- **Model Selection:** Based on validation loss

## Tasks

### 1. Create checkpoint manager
Create `src/poke/training/checkpointing.py`:
```python
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
```

### 2. Create metrics logger
Create `src/poke/training/logging.py`:
```python
"""Training metrics logging."""
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class MetricsLogger:
    """Log training metrics to CSV and optionally wandb."""

    def __init__(
        self,
        log_dir: Path,
        use_wandb: bool = False,
        wandb_project: str = "poke-training",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV logging
        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None

        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project=wandb_project)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics for a step.

        Args:
            metrics: Dict of metric name -> value
            step: Global training step
        """
        # Add timestamp and step
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        # CSV logging
        self._log_csv(record)

        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def _log_csv(self, record: dict) -> None:
        """Append record to CSV file."""
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=list(record.keys())
            )
            self.csv_writer.writeheader()

        self.csv_writer.writerow(record)
        self.csv_file.flush()

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration."""
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.use_wandb:
            import wandb
            wandb.config.update(config)

    def close(self) -> None:
        """Close log files."""
        if self.csv_file:
            self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

### 3. Create model comparison utility
Create `src/poke/training/comparison.py`:
```python
"""Model comparison utilities."""
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch

from .checkpointing import CheckpointManager

@dataclass
class ModelInfo:
    """Information about a trained model."""
    name: str
    path: Path
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    val_accuracy: Optional[float]

def compare_checkpoints(
    checkpoint_dirs: List[Path],
    metric: str = "val_loss",
) -> List[ModelInfo]:
    """Compare models from different training runs.

    Args:
        checkpoint_dirs: List of checkpoint directories
        metric: Metric to sort by

    Returns:
        Sorted list of ModelInfo (best first)
    """
    models = []

    for dir_path in checkpoint_dirs:
        manager = CheckpointManager(dir_path)
        best_path = manager.get_best()

        if best_path and best_path.exists():
            checkpoint = torch.load(best_path, map_location="cpu")
            metadata = checkpoint.get("metadata", {})

            models.append(ModelInfo(
                name=dir_path.name,
                path=best_path,
                epoch=metadata.get("epoch", 0),
                train_loss=metadata.get("train_loss", float("inf")),
                val_loss=metadata.get("val_loss"),
                val_accuracy=metadata.get("val_accuracy"),
            ))

    # Sort by metric
    def get_metric(m):
        if metric == "val_loss":
            return m.val_loss or float("inf")
        elif metric == "val_accuracy":
            return -(m.val_accuracy or 0)
        else:
            return m.train_loss

    return sorted(models, key=get_metric)

def print_comparison(models: List[ModelInfo]) -> None:
    """Print model comparison table."""
    print(f"\n{'Name':<20} {'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val Acc':>10}")
    print("-" * 70)

    for m in models:
        val_loss = f"{m.val_loss:.4f}" if m.val_loss else "N/A"
        val_acc = f"{m.val_accuracy:.4f}" if m.val_accuracy else "N/A"
        print(f"{m.name:<20} {m.epoch:>6} {m.train_loss:>12.4f} {val_loss:>12} {val_acc:>10}")
```

### 4. Create comparison script
Create `scripts/compare_models.py`:
```python
#!/usr/bin/env python
"""Compare trained models."""
import argparse
from pathlib import Path

from poke.training.comparison import compare_checkpoints, print_comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Checkpoint directories to compare")
    parser.add_argument("--metric", default="val_loss", choices=["val_loss", "val_accuracy", "train_loss"])
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dirs]
    models = compare_checkpoints(dirs, metric=args.metric)
    print_comparison(models)

if __name__ == "__main__":
    main()
```

### 5. Write tests
Create `tests/training/test_checkpointing.py`:
```python
"""Tests for checkpointing."""
import pytest
import torch
from pathlib import Path

from poke.training.checkpointing import CheckpointManager, CheckpointMetadata

@pytest.fixture
def checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"

@pytest.fixture
def simple_model():
    return torch.nn.Linear(10, 10)

def test_save_and_load(checkpoint_dir, simple_model):
    manager = CheckpointManager(checkpoint_dir)
    optimizer = torch.optim.Adam(simple_model.parameters())

    metadata = CheckpointMetadata(
        epoch=1,
        step=100,
        timestamp="2024-01-01",
        train_loss=0.5,
        train_accuracy=0.7,
    )

    path = manager.save(simple_model, optimizer, metadata)

    assert path.exists()

    loaded = manager.load(path)
    assert "model_state_dict" in loaded
    assert loaded["metadata"]["epoch"] == 1

def test_best_checkpoint_tracking(checkpoint_dir, simple_model):
    manager = CheckpointManager(checkpoint_dir)
    optimizer = torch.optim.Adam(simple_model.parameters())

    # Save first checkpoint
    meta1 = CheckpointMetadata(epoch=1, step=100, timestamp="", train_loss=0.5, train_accuracy=0.7)
    manager.save(simple_model, optimizer, meta1, is_best=True)

    # Save second (better) checkpoint
    meta2 = CheckpointMetadata(epoch=2, step=200, timestamp="", train_loss=0.3, train_accuracy=0.8)
    manager.save(simple_model, optimizer, meta2, is_best=True)

    # Best should be the second one
    best_path = manager.get_best()
    assert "step200" in str(best_path)

def test_cleanup_old_checkpoints(checkpoint_dir, simple_model):
    manager = CheckpointManager(checkpoint_dir, max_to_keep=2)
    optimizer = torch.optim.Adam(simple_model.parameters())

    # Save 5 checkpoints
    for i in range(5):
        meta = CheckpointMetadata(
            epoch=i, step=i*100, timestamp="",
            train_loss=1.0 - i*0.1, train_accuracy=0.5
        )
        manager.save(simple_model, optimizer, meta)

    # Should only keep 2
    assert len(list(checkpoint_dir.glob("checkpoint_*.pt"))) <= 3  # +1 for best.pt
```

## Acceptance Criteria
- [ ] Checkpoints save with full metadata
- [ ] Best model tracked automatically
- [ ] Old checkpoints cleaned up
- [ ] Metrics logged to CSV
- [ ] Model comparison works across runs
- [ ] Wandb integration (when enabled)

## Notes
- Keep checkpoint files under 1GB
- Include config in metadata for reproducibility
- Use JSON for human-readable metadata

## Estimated Complexity
Low-Medium - Mostly file management and logging
