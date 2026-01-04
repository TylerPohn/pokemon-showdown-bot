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
