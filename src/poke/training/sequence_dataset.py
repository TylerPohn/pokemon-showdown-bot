"""Sequence-based dataset for transformer training.

This module provides datasets that return sequences of turns for training
decoder-only transformers that need battle history context.

Unlike the single-turn dataset, this returns:
- Sequences of observations (for context)
- The action at the last timestep (prediction target)
- Optional: full action sequences for sequence-level training
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

from ..models.preprocessing import FeaturePreprocessor


class SequenceTrajectoryDataset(Dataset):
    """Dataset that returns sequences of turns for transformer training.

    Each sample contains:
    - observations: Sequence of turn observations [seq_len, ...]
    - action: Target action for the last turn
    - action_mask: Valid action mask for the last turn
    - rewards: Optional reward sequence for RL training
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        seq_len: int = 50,
        preprocessor: Optional[FeaturePreprocessor] = None,
        max_samples: Optional[int] = None,
        overlap: bool = True,
        include_rewards: bool = False,
    ):
        """Initialize the sequence dataset.

        Args:
            data_path: Path to trajectory JSONL file
            seq_len: Maximum sequence length (number of turns)
            preprocessor: Feature preprocessor for observations
            max_samples: Maximum number of samples to load
            overlap: If True, create overlapping sequences from trajectories
            include_rewards: If True, include reward sequences
        """
        super().__init__()
        self.seq_len = seq_len
        self.preprocessor = preprocessor or FeaturePreprocessor()
        self.include_rewards = include_rewards

        # Load and process trajectories into sequences
        self.sequences: List[Dict] = []
        self._load_trajectories(data_path, max_samples, overlap)

    def _load_trajectories(
        self,
        data_path: Union[str, Path],
        max_samples: Optional[int],
        overlap: bool,
    ) -> None:
        """Load trajectories and split into sequences."""
        data_path = Path(data_path)

        with open(data_path, "r") as f:
            for line in f:
                if max_samples and len(self.sequences) >= max_samples:
                    break

                trajectory = json.loads(line)
                steps = trajectory["steps"]

                if len(steps) < 2:
                    continue

                # Create sequences from this trajectory
                if overlap:
                    # Overlapping sequences: one per timestep
                    for end_idx in range(1, len(steps)):
                        start_idx = max(0, end_idx - self.seq_len + 1)
                        self.sequences.append({
                            "steps": steps[start_idx:end_idx + 1],
                            "trajectory_id": trajectory.get("replay_id", "unknown"),
                        })

                        if max_samples and len(self.sequences) >= max_samples:
                            break
                else:
                    # Non-overlapping: chunk into seq_len pieces
                    for i in range(0, len(steps), self.seq_len):
                        chunk = steps[i:i + self.seq_len]
                        if len(chunk) >= 2:  # Need at least 2 steps
                            self.sequences.append({
                                "steps": chunk,
                                "trajectory_id": trajectory.get("replay_id", "unknown"),
                            })

                            if max_samples and len(self.sequences) >= max_samples:
                                break

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample.

        Returns:
            Dictionary containing:
            - observations: [seq_len, feat_dim] observation sequence
            - action: Target action for last timestep
            - seq_len: Actual sequence length (for masking)
            - rewards: (optional) [seq_len] reward sequence
        """
        sequence_data = self.sequences[idx]
        steps = sequence_data["steps"]

        # Process observations
        observations = []
        actions = []
        rewards = []

        for step in steps:
            # Preprocess observation
            obs = self.preprocessor.preprocess(step["observation"])
            observations.append(obs)

            # Encode action (move: 0-3, switch: 4-9)
            action_type = step["action_type"]
            action_target = step["action_target"]
            if action_type == 0:  # Move
                action = action_target
            else:  # Switch
                action = 4 + action_target
            actions.append(action)

            if self.include_rewards:
                rewards.append(step.get("reward", 0.0))

        # Pad sequences to seq_len
        actual_len = len(observations)

        # Stack observations into tensors
        # Each observation is a dict of tensors, we need to stack them
        stacked_obs = self._stack_observations(observations)

        # Pad if needed
        if actual_len < self.seq_len:
            stacked_obs = self._pad_observations(stacked_obs, actual_len)

        # Target is the last action
        target_action = actions[-1]

        result = {
            "observations": stacked_obs,
            "action": torch.tensor(target_action, dtype=torch.long),
            "seq_len": torch.tensor(actual_len, dtype=torch.long),
        }

        if self.include_rewards:
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            if actual_len < self.seq_len:
                reward_tensor = torch.nn.functional.pad(
                    reward_tensor, (0, self.seq_len - actual_len)
                )
            result["rewards"] = reward_tensor

        return result

    def _stack_observations(
        self,
        observations: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Stack a list of observation dicts into a single dict of tensors."""
        if not observations:
            raise ValueError("Cannot stack empty observation list")

        keys = observations[0].keys()
        return {
            key: torch.stack([obs[key] for obs in observations])
            for key in keys
        }

    def _pad_observations(
        self,
        observations: Dict[str, torch.Tensor],
        actual_len: int,
    ) -> Dict[str, torch.Tensor]:
        """Pad observations to seq_len."""
        pad_len = self.seq_len - actual_len
        padded = {}

        for key, tensor in observations.items():
            # Pad along first dimension (sequence)
            if tensor.dim() == 1:
                padded[key] = torch.nn.functional.pad(tensor, (0, pad_len))
            else:
                pad_shape = [0] * (2 * (tensor.dim() - 1)) + [0, pad_len]
                padded[key] = torch.nn.functional.pad(tensor, pad_shape[::-1])

        return padded


class SequenceCollator:
    """Custom collator for sequence batches.

    Handles variable-length sequences and creates attention masks.
    """

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of sequence samples.

        Args:
            batch: List of samples from SequenceTrajectoryDataset

        Returns:
            Collated batch with attention masks
        """
        # Stack observations
        obs_keys = batch[0]["observations"].keys()
        observations = {
            key: torch.stack([sample["observations"][key] for sample in batch])
            for key in obs_keys
        }

        # Stack other tensors
        actions = torch.stack([sample["action"] for sample in batch])
        seq_lens = torch.stack([sample["seq_len"] for sample in batch])

        # Create attention mask (1 = attend, 0 = mask)
        batch_size = len(batch)
        max_len = self.max_seq_len
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(seq_lens):
            attention_mask[i, :length] = True

        result = {
            "observations": observations,
            "action": actions,
            "seq_len": seq_lens,
            "attention_mask": attention_mask,
        }

        # Include rewards if present
        if "rewards" in batch[0]:
            result["rewards"] = torch.stack([sample["rewards"] for sample in batch])

        return result


def create_sequence_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 32,
    seq_len: int = 50,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    include_rewards: bool = False,
) -> DataLoader:
    """Create a DataLoader for sequence training.

    Args:
        data_path: Path to trajectory JSONL file
        batch_size: Batch size
        seq_len: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        max_samples: Maximum number of samples
        include_rewards: Whether to include reward sequences

    Returns:
        Configured DataLoader
    """
    dataset = SequenceTrajectoryDataset(
        data_path=data_path,
        seq_len=seq_len,
        max_samples=max_samples,
        include_rewards=include_rewards,
    )

    collator = SequenceCollator(max_seq_len=seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


class RLSequenceDataset(Dataset):
    """Dataset for offline RL training with sequences.

    Returns full transition information:
    - observations: Sequence of states
    - actions: Sequence of actions taken
    - rewards: Sequence of rewards received
    - dones: Sequence of done flags
    - next_observations: Sequence of next states (for TD learning)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        seq_len: int = 50,
        preprocessor: Optional[FeaturePreprocessor] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.preprocessor = preprocessor or FeaturePreprocessor()
        self.sequences: List[Dict] = []
        self._load_trajectories(data_path, max_samples)

    def _load_trajectories(
        self,
        data_path: Union[str, Path],
        max_samples: Optional[int],
    ) -> None:
        """Load trajectories for RL training."""
        data_path = Path(data_path)

        with open(data_path, "r") as f:
            for line in f:
                if max_samples and len(self.sequences) >= max_samples:
                    break

                trajectory = json.loads(line)
                steps = trajectory["steps"]

                if len(steps) < 2:
                    continue

                # For RL, we want overlapping sequences
                for end_idx in range(1, len(steps)):
                    start_idx = max(0, end_idx - self.seq_len + 1)
                    self.sequences.append({
                        "steps": steps[start_idx:end_idx + 1],
                        "total_reward": trajectory.get("total_reward", 0.0),
                    })

                    if max_samples and len(self.sequences) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an RL training sample."""
        sequence_data = self.sequences[idx]
        steps = sequence_data["steps"]

        observations = []
        actions = []
        rewards = []
        dones = []

        for i, step in enumerate(steps):
            # Preprocess observation
            obs = self.preprocessor.preprocess(step["observation"])
            observations.append(obs)

            # Encode action
            action_type = step["action_type"]
            action_target = step["action_target"]
            if action_type == 0:
                action = action_target
            else:
                action = 4 + action_target
            actions.append(action)

            rewards.append(step.get("reward", 0.0))
            dones.append(float(step.get("done", i == len(steps) - 1)))

        # Stack observations
        stacked_obs = self._stack_observations(observations)
        actual_len = len(observations)

        # Pad if needed
        if actual_len < self.seq_len:
            stacked_obs = self._pad_observations(stacked_obs, actual_len)
            actions = actions + [0] * (self.seq_len - actual_len)
            rewards = rewards + [0.0] * (self.seq_len - actual_len)
            dones = dones + [1.0] * (self.seq_len - actual_len)

        return {
            "observations": stacked_obs,
            "actions": torch.tensor(actions[:self.seq_len], dtype=torch.long),
            "rewards": torch.tensor(rewards[:self.seq_len], dtype=torch.float32),
            "dones": torch.tensor(dones[:self.seq_len], dtype=torch.float32),
            "seq_len": torch.tensor(actual_len, dtype=torch.long),
        }

    def _stack_observations(
        self,
        observations: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        keys = observations[0].keys()
        return {
            key: torch.stack([obs[key] for obs in observations])
            for key in keys
        }

    def _pad_observations(
        self,
        observations: Dict[str, torch.Tensor],
        actual_len: int,
    ) -> Dict[str, torch.Tensor]:
        pad_len = self.seq_len - actual_len
        padded = {}
        for key, tensor in observations.items():
            if tensor.dim() == 1:
                padded[key] = torch.nn.functional.pad(tensor, (0, pad_len))
            else:
                pad_shape = [0] * (2 * (tensor.dim() - 1)) + [0, pad_len]
                padded[key] = torch.nn.functional.pad(tensor, pad_shape[::-1])
        return padded
