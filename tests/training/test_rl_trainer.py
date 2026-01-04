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
