"""Tests for state encoder."""
import pytest
import torch

from poke.models.config import EncoderConfig
from poke.models.state_encoder import StateEncoder
from poke.models.preprocessing import FeaturePreprocessor

@pytest.fixture
def config():
    return EncoderConfig()

@pytest.fixture
def encoder(config):
    return StateEncoder(config)

def test_encoder_output_shape(encoder, config):
    batch_size = 4

    observation = {
        "team_id": torch.zeros(batch_size, dtype=torch.long),
        "weather": torch.zeros(batch_size, dtype=torch.long),
        "terrain": torch.zeros(batch_size, dtype=torch.long),
        "pokemon_features": torch.zeros(batch_size, 12 * 32),
        "hazards": torch.zeros(batch_size, 16),
    }

    output = encoder(observation)

    assert output.shape == (batch_size, config.output_dim)

def test_preprocessor():
    preprocessor = FeaturePreprocessor()

    raw_obs = {
        "team_id": 5,
        "weather_id": 1,
        "terrain_id": 0,
        "team_hp": [1.0, 0.8, 0.5, 0.0, 1.0, 1.0],
        "team_fainted": [False, False, False, True, False, False],
        "own_stealth_rock": True,
        "own_spikes": 2,
    }

    processed = preprocessor.preprocess(raw_obs)

    assert "team_id" in processed
    assert "pokemon_features" in processed
    assert processed["team_id"].item() == 5
