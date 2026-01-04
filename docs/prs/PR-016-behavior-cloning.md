COMPLETED

# PR-016: Behavior Cloning Training Loop

## Dependencies
- PR-006 (Trajectory Converter) - for training data
- PR-013 (State Encoder)
- PR-014 (Action Masking)
- PR-015 (Policy Network)

## Overview
Implement supervised behavior cloning to train the policy on human demonstrations. This provides policy initialization for offline RL.

## Tech Choices
- **Loss:** Cross-entropy with label smoothing
- **Optimizer:** AdamW
- **Data Loading:** PyTorch DataLoader
- **Tracking:** Weights & Biases (optional)

## Tasks

### 1. Create dataset class
Create `src/poke/training/dataset.py`:
```python
"""Dataset classes for training."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from ..data.trajectory import Observation
from ..models.preprocessing import FeaturePreprocessor

class TrajectoryDataset(Dataset):
    """Dataset of (observation, action) pairs for behavior cloning."""

    def __init__(
        self,
        data_path: Path,
        preprocessor: Optional[FeaturePreprocessor] = None,
        max_samples: Optional[int] = None,
    ):
        self.preprocessor = preprocessor or FeaturePreprocessor()
        self.samples: List[Dict] = []

        # Load data
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                trajectory = json.loads(line)
                for step in trajectory["steps"]:
                    self.samples.append({
                        "observation": step["observation"],
                        "action_type": step["action_type"],
                        "action_target": step["action_target"],
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Preprocess observation
        obs = self.preprocessor.preprocess(sample["observation"])

        # Encode action (move 0-3 or switch 4-9)
        if sample["action_type"] == 0:  # Move
            action = sample["action_target"]
        else:  # Switch
            action = 4 + sample["action_target"]

        return {
            **obs,
            "action": torch.tensor(action, dtype=torch.long),
        }


def create_dataloader(
    data_path: Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for trajectory data."""
    dataset = TrajectoryDataset(data_path, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
```

### 2. Implement behavior cloning trainer
Create `src/poke/training/bc_trainer.py`:
```python
"""Behavior cloning trainer."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints/bc")
    save_every: int = 1  # Save every N epochs

    # Logging
    log_every: int = 100  # Log every N steps
    use_wandb: bool = False

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BCTrainer:
    """Trainer for behavior cloning."""

    def __init__(
        self,
        policy: nn.Module,
        config: BCConfig,
    ):
        self.policy = policy.to(config.device)
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        if config.use_wandb:
            import wandb
            wandb.init(project="poke-bc")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            actions = batch.pop("action")

            # Forward pass
            action_probs, _ = self.policy(batch)

            # Compute loss
            loss = self.criterion(
                action_probs.log(),
                actions
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            # Tracking
            total_loss += loss.item() * len(actions)
            predictions = action_probs.argmax(dim=-1)
            total_correct += (predictions == actions).sum().item()
            total_samples += len(actions)

            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_every == 0:
                pbar.set_postfix({
                    "loss": loss.item(),
                    "acc": total_correct / total_samples,
                })

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data."""
        self.policy.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            actions = batch.pop("action")

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
```

### 3. Create training script
Create `scripts/train_bc.py`:
```python
#!/usr/bin/env python
"""Train behavior cloning policy."""
import argparse
import logging
from pathlib import Path

from poke.models.config import EncoderConfig
from poke.models.factory import create_policy
from poke.training.dataset import create_dataloader
from poke.training.bc_trainer import BCTrainer, BCConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/bc")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create model
    encoder_config = EncoderConfig()
    policy = create_policy("mlp", encoder_config=encoder_config)

    # Create data loaders
    train_loader = create_dataloader(
        Path(args.data),
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            Path(args.val_data),
            batch_size=args.batch_size,
            shuffle=False,
        )

    # Create trainer
    config = BCConfig(
        train_data_path=Path(args.data),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=Path(args.checkpoint_dir),
        use_wandb=args.wandb,
    )
    trainer = BCTrainer(policy, config)

    # Train
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
```

### 4. Add data splitting utility
Create `scripts/split_data.py`:
```python
#!/usr/bin/env python
"""Split trajectory data into train/val sets."""
import argparse
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    train_path = input_path.with_suffix(".train.jsonl")
    val_path = input_path.with_suffix(".val.jsonl")

    with open(input_path) as f:
        lines = f.readlines()

    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - args.val_ratio))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    with open(train_path, "w") as f:
        f.writelines(train_lines)

    with open(val_path, "w") as f:
        f.writelines(val_lines)

    print(f"Train: {len(train_lines)} trajectories -> {train_path}")
    print(f"Val: {len(val_lines)} trajectories -> {val_path}")

if __name__ == "__main__":
    main()
```

### 5. Write unit tests
Create `tests/training/test_bc.py`:
```python
"""Tests for behavior cloning."""
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock

from poke.training.dataset import TrajectoryDataset
from poke.training.bc_trainer import BCTrainer, BCConfig

@pytest.fixture
def sample_trajectory_file(tmp_path):
    import json
    data = {
        "replay_id": "test",
        "player": "p1",
        "total_reward": 1.0,
        "steps": [
            {
                "observation": {
                    "turn": 1,
                    "team_id": 0,
                    "weather_id": 0,
                    "team_hp": [1.0] * 6,
                },
                "action_type": 0,
                "action_target": 0,
                "reward": 0,
                "done": False,
            },
            {
                "observation": {
                    "turn": 2,
                    "team_id": 0,
                    "weather_id": 0,
                    "team_hp": [0.8] * 6,
                },
                "action_type": 1,
                "action_target": 1,
                "reward": 1.0,
                "done": True,
            },
        ],
    }

    path = tmp_path / "trajectories.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(data) + "\n")

    return path

def test_dataset_loading(sample_trajectory_file):
    dataset = TrajectoryDataset(sample_trajectory_file)

    assert len(dataset) == 2

def test_dataset_sample_format(sample_trajectory_file):
    dataset = TrajectoryDataset(sample_trajectory_file)
    sample = dataset[0]

    assert "action" in sample
    assert sample["action"].dtype == torch.long
    assert 0 <= sample["action"].item() < 10

def test_bc_trainer_step():
    # Create mock policy
    policy = Mock()
    policy.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    policy.train = Mock()
    policy.to = Mock(return_value=policy)

    # Mock forward pass
    batch_size = 4
    action_probs = torch.softmax(torch.randn(batch_size, 10), dim=-1)
    policy.return_value = (action_probs, None)

    config = BCConfig(device="cpu")

    # Trainer creation should work
    # (full training test would need more mocking)
```

## Acceptance Criteria
- [ ] Dataset loads trajectory files correctly
- [ ] Training loop runs without errors
- [ ] Loss decreases over epochs
- [ ] Checkpoints save and load correctly
- [ ] Validation metrics computed correctly
- [ ] Wandb integration works (when enabled)

## Expected Results
- Training accuracy: 50-70% (human moves are noisy)
- Validation should not be much worse than training

## Notes
- Start with small learning rate to avoid divergence
- Label smoothing helps with noisy labels
- Monitor for overfitting on small datasets

## Estimated Complexity
Medium - Standard training loop with some customization
