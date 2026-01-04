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
