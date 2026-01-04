"""Tests for policy networks."""
import pytest
import torch

from poke.models.config import EncoderConfig
from poke.models.policy import MLPPolicy, ResidualBlock
from poke.models.factory import create_policy

@pytest.fixture
def config():
    return EncoderConfig()

@pytest.fixture
def sample_observation():
    batch_size = 4
    return {
        "team_id": torch.zeros(batch_size, dtype=torch.long),
        "weather": torch.zeros(batch_size, dtype=torch.long),
        "terrain": torch.zeros(batch_size, dtype=torch.long),
        "pokemon_features": torch.randn(batch_size, 12 * 32),
        "hazards": torch.randn(batch_size, 16),
    }

def test_mlp_policy_output_shape(config, sample_observation):
    policy = MLPPolicy(encoder_config=config)

    action_probs, value = policy(sample_observation)

    assert action_probs.shape == (4, 10)
    assert value.shape == (4,)

def test_action_probs_sum_to_one(config, sample_observation):
    policy = MLPPolicy(encoder_config=config)

    action_probs, _ = policy(sample_observation)

    sums = action_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)

def test_masked_probs_zero(config, sample_observation):
    policy = MLPPolicy(encoder_config=config)

    # Mask out actions 5-9
    mask = torch.tensor([[True] * 5 + [False] * 5] * 4)

    action_probs, _ = policy(sample_observation, action_mask=mask)

    # Masked actions should be ~0
    assert (action_probs[:, 5:] < 1e-6).all()

def test_factory_creates_mlp(config):
    policy = create_policy("mlp", encoder_config=config)

    assert isinstance(policy, MLPPolicy)

def test_residual_block_shape():
    block = ResidualBlock(dim=64, hidden_dim=64)

    x = torch.randn(4, 64)
    y = block(x)

    assert y.shape == x.shape
