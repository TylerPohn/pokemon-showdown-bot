"""Tests for observation building."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from poke.agents.observation import ObservationBuilder, ObservationConfig

@pytest.fixture
def mock_battle():
    battle = Mock()
    battle.active_pokemon = Mock()
    battle.active_pokemon.current_hp_fraction = 0.75
    battle.active_pokemon.status = None
    battle.active_pokemon.boosts = {}
    battle.active_pokemon.must_recharge = False
    battle.active_pokemon.is_dynamaxed = False
    battle.active_pokemon.terastallized = False

    battle.team = {}
    battle.opponent_team = {}
    battle.opponent_active_pokemon = None
    battle.weather = None
    battle.fields = {}
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}

    return battle

def test_observation_shape(mock_battle):
    config = ObservationConfig()
    builder = ObservationBuilder(config)

    obs = builder.build(mock_battle)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert len(obs) == builder.observation_size

def test_team_id_in_observation(mock_battle):
    config = ObservationConfig(include_team_id=True)
    builder = ObservationBuilder(config)
    builder.set_team_id(5)

    obs = builder.build(mock_battle)

    # Team ID should be first feature (normalized)
    assert obs[0] == pytest.approx(5 / 99)  # max_teams - 1

def test_hp_encoding(mock_battle):
    config = ObservationConfig(include_team_id=False)
    builder = ObservationBuilder(config)

    mock_battle.active_pokemon.current_hp_fraction = 0.5
    obs = builder.build(mock_battle)

    # HP should be in the first position after team_id
    assert 0.5 in obs
