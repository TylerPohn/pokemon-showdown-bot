COMPLETED

# PR-019: Offline RL Training Loop

## Dependencies
- PR-016 (Behavior Cloning) - for BC initialization
- PR-018 (Offline RL Algorithms)

## Overview
Implement the training loop for offline RL with BC initialization, data loading, and hyperparameter management.

## Tech Choices
- **Initialization:** Load from BC checkpoint
- **Data Format:** Same as BC (trajectory JSONL)
- **Scheduling:** Linear warmup, cosine decay

## Tasks

### 1. Create offline RL dataset
Create `src/poke/training/rl_dataset.py`:
```python
"""Dataset for offline RL training."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from ..models.preprocessing import FeaturePreprocessor

class OfflineRLDataset(Dataset):
    """Dataset of (s, a, r, s', done) tuples for offline RL."""

    def __init__(
        self,
        data_path: Path,
        preprocessor: Optional[FeaturePreprocessor] = None,
        action_dim: int = 10,
        max_samples: Optional[int] = None,
    ):
        self.preprocessor = preprocessor or FeaturePreprocessor()
        self.action_dim = action_dim
        self.transitions: List[Dict] = []

        # Load and flatten trajectories into transitions
        with open(data_path) as f:
            for line in f:
                trajectory = json.loads(line)
                steps = trajectory["steps"]

                for i in range(len(steps) - 1):
                    self.transitions.append({
                        "state": steps[i]["observation"],
                        "action_type": steps[i]["action_type"],
                        "action_target": steps[i]["action_target"],
                        "reward": steps[i]["reward"],
                        "next_state": steps[i + 1]["observation"],
                        "done": steps[i]["done"],
                    })

                    if max_samples and len(self.transitions) >= max_samples:
                        break

                if max_samples and len(self.transitions) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.transitions[idx]

        # Preprocess states
        state = self.preprocessor.preprocess(t["state"])
        next_state = self.preprocessor.preprocess(t["next_state"])

        # Encode action as one-hot
        if t["action_type"] == 0:  # Move
            action_idx = t["action_target"]
        else:  # Switch
            action_idx = 4 + t["action_target"]

        action = F.one_hot(
            torch.tensor(action_idx),
            self.action_dim
        ).float()

        return {
            "state": state,
            "action": action,
            "reward": torch.tensor(t["reward"], dtype=torch.float32),
            "next_state": next_state,
            "done": torch.tensor(t["done"], dtype=torch.float32),
        }


def create_rl_dataloader(
    data_path: Path,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create DataLoader for offline RL."""
    dataset = OfflineRLDataset(data_path, **kwargs)

    def collate_fn(batch):
        # Custom collation to handle nested dicts
        result = {}
        result["states"] = _collate_observations([b["state"] for b in batch])
        result["next_states"] = _collate_observations([b["next_state"] for b in batch])
        result["actions"] = torch.stack([b["action"] for b in batch])
        result["rewards"] = torch.stack([b["reward"] for b in batch])
        result["dones"] = torch.stack([b["done"] for b in batch])
        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

def _collate_observations(obs_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate list of observation dicts into batched dict."""
    keys = obs_list[0].keys()
    return {
        key: torch.stack([o[key] for o in obs_list])
        for key in keys
    }
```

### 2. Create training loop
Create `src/poke/training/rl_trainer.py`:
```python
"""Training loop for offline RL."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .iql import IQL, IQLConfig
from .cql import CQL, CQLConfig
from .checkpointing import CheckpointManager, CheckpointMetadata
from .logging import MetricsLogger

logger = logging.getLogger(__name__)

@dataclass
class OfflineRLConfig:
    """Configuration for offline RL training."""
    # Algorithm
    algorithm: str = "iql"  # "iql" or "cql"

    # Data
    train_data_path: Path = Path("data/processed/trajectories.jsonl")

    # Training
    batch_size: int = 256
    num_steps: int = 100000
    eval_every: int = 5000
    log_every: int = 100

    # Initialization
    bc_checkpoint: Optional[Path] = None

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints/rl")
    save_every: int = 10000

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Algorithm-specific
    iql_config: Optional[IQLConfig] = None
    cql_config: Optional[CQLConfig] = None


class OfflineRLTrainer:
    """Trainer for offline RL algorithms."""

    def __init__(
        self,
        policy: torch.nn.Module,
        state_dim: int,
        config: OfflineRLConfig,
    ):
        self.config = config
        self.state_dim = state_dim

        # Initialize algorithm
        if config.algorithm == "iql":
            algo_config = config.iql_config or IQLConfig()
            self.algorithm = IQL(
                policy=policy,
                state_dim=state_dim,
                config=algo_config,
                device=config.device,
            )
        elif config.algorithm == "cql":
            algo_config = config.cql_config or CQLConfig()
            self.algorithm = CQL(
                policy=policy,
                state_dim=state_dim,
                config=algo_config,
                device=config.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        # Load BC initialization if provided
        if config.bc_checkpoint:
            self._load_bc_checkpoint(config.bc_checkpoint)

        # Checkpointing and logging
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.metrics_logger = MetricsLogger(config.checkpoint_dir / "logs")

        self.global_step = 0

    def _load_bc_checkpoint(self, path: Path) -> None:
        """Load policy weights from BC checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.algorithm.policy.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded BC checkpoint: {path}")

    def train(self, dataloader: DataLoader) -> None:
        """Run training loop."""
        logger.info(f"Starting {self.config.algorithm.upper()} training")
        logger.info(f"Total steps: {self.config.num_steps}")

        data_iter = iter(dataloader)
        pbar = tqdm(total=self.config.num_steps, desc="Training")

        running_metrics: Dict[str, float] = {}

        while self.global_step < self.config.num_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Update
            metrics = self.algorithm.update(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in running_metrics:
                    running_metrics[key] = 0.0
                running_metrics[key] += value

            self.global_step += 1
            pbar.update(1)

            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_metrics = {
                    k: v / self.config.log_every
                    for k, v in running_metrics.items()
                }
                self.metrics_logger.log(avg_metrics, self.global_step)

                pbar.set_postfix(avg_metrics)
                running_metrics.clear()

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()
        logger.info("Training complete")

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        metadata = CheckpointMetadata(
            epoch=0,  # Offline RL doesn't have epochs
            step=self.global_step,
            timestamp="",
            train_loss=0.0,  # Could track running loss
            train_accuracy=0.0,
        )

        self.checkpoint_manager.save(
            model=self.algorithm.policy,
            optimizer=self.algorithm.policy_optimizer,
            metadata=metadata,
        )


def create_trainer(
    policy: torch.nn.Module,
    state_dim: int,
    config: OfflineRLConfig,
) -> OfflineRLTrainer:
    """Create offline RL trainer."""
    return OfflineRLTrainer(policy, state_dim, config)
```

### 3. Create training script
Create `scripts/train_rl.py`:
```python
#!/usr/bin/env python
"""Train offline RL policy."""
import argparse
import logging
from pathlib import Path

from poke.models.config import EncoderConfig
from poke.models.factory import create_policy
from poke.training.rl_dataset import create_rl_dataloader
from poke.training.rl_trainer import OfflineRLConfig, create_trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--algorithm", default="iql", choices=["iql", "cql"])
    parser.add_argument("--bc-checkpoint", help="BC checkpoint for initialization")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-dir", default="checkpoints/rl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create model
    encoder_config = EncoderConfig()
    policy = create_policy("mlp", encoder_config=encoder_config)

    # Create data loader
    dataloader = create_rl_dataloader(
        Path(args.data),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create trainer
    config = OfflineRLConfig(
        algorithm=args.algorithm,
        train_data_path=Path(args.data),
        bc_checkpoint=Path(args.bc_checkpoint) if args.bc_checkpoint else None,
        num_steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=args.device,
    )

    trainer = create_trainer(
        policy=policy,
        state_dim=encoder_config.output_dim,
        config=config,
    )

    # Train
    trainer.train(dataloader)

if __name__ == "__main__":
    main()
```

### 4. Add learning rate scheduling
Create `src/poke/training/scheduling.py`:
```python
"""Learning rate schedulers."""
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Create cosine decay schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress))
        )

    return LambdaLR(optimizer, lr_lambda)
```

### 5. Write tests
Create `tests/training/test_rl_trainer.py`:
```python
"""Tests for offline RL training."""
import pytest
import torch
from pathlib import Path
import json

from poke.training.rl_dataset import OfflineRLDataset
from poke.training.scheduling import get_cosine_schedule_with_warmup

@pytest.fixture
def sample_trajectory_file(tmp_path):
    data = {
        "replay_id": "test",
        "player": "p1",
        "total_reward": 1.0,
        "steps": [
            {
                "observation": {"turn": i, "team_id": 0, "weather_id": 0, "team_hp": [1.0]*6},
                "action_type": 0,
                "action_target": i % 4,
                "reward": 0 if i < 9 else 1.0,
                "done": i == 9,
            }
            for i in range(10)
        ],
    }

    path = tmp_path / "trajectories.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(data) + "\n")

    return path

def test_rl_dataset_transitions(sample_trajectory_file):
    dataset = OfflineRLDataset(sample_trajectory_file)

    # 10 steps = 9 transitions
    assert len(dataset) == 9

def test_rl_dataset_sample_format(sample_trajectory_file):
    dataset = OfflineRLDataset(sample_trajectory_file)
    sample = dataset[0]

    assert "state" in sample
    assert "action" in sample
    assert "reward" in sample
    assert "next_state" in sample
    assert "done" in sample

    assert sample["action"].shape == (10,)  # One-hot
    assert sample["action"].sum() == 1.0  # Valid one-hot

def test_cosine_schedule():
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
    )

    # Check warmup
    for _ in range(100):
        scheduler.step()

    # Should be at peak after warmup
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-3, rel=0.1)

    # Check decay
    for _ in range(900):
        scheduler.step()

    # Should be at minimum after full training
    assert scheduler.get_last_lr()[0] < 0.5 * 1e-3
```

## Acceptance Criteria
- [ ] Dataset correctly creates transitions from trajectories
- [ ] BC checkpoint loads and initializes policy
- [ ] Training loop runs for specified steps
- [ ] Metrics logged correctly
- [ ] LR scheduling works
- [ ] Checkpoints save/load correctly

## Notes
- Always initialize from BC for better stability
- Monitor Q-values and policy loss for divergence
- IQL is recommended as the default algorithm

## Estimated Complexity
Medium - Integration of multiple components
