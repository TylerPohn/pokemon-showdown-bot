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
