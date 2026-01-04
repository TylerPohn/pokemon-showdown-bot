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
