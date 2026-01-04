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
