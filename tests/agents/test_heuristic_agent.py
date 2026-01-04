"""Tests for heuristic agent."""
import pytest
from unittest.mock import Mock, MagicMock

from poke.agents.heuristic_agent import MaxDamageAgent

@pytest.fixture
def mock_battle():
    battle = Mock()

    # Active Pokemon
    active = Mock()
    active.current_hp_fraction = 1.0
    active.types = ("Electric",)
    active.stats = {"atk": 100, "spa": 120}
    battle.active_pokemon = active

    # Opponent
    opponent = Mock()
    opponent.current_hp_fraction = 0.8
    opponent.types = ("Water",)
    opponent.stats = {"def": 80, "spd": 90}
    opponent.damage_multiplier = lambda m: 2.0 if m.type == "Electric" else 1.0
    battle.opponent_active_pokemon = opponent

    # Moves
    thunderbolt = Mock()
    thunderbolt.id = "thunderbolt"
    thunderbolt.base_power = 90
    thunderbolt.type = "Electric"
    thunderbolt.category = Mock()
    thunderbolt.category.name = "SPECIAL"
    thunderbolt.priority = 0
    thunderbolt.accuracy = 100

    tackle = Mock()
    tackle.id = "tackle"
    tackle.base_power = 40
    tackle.type = "Normal"
    tackle.category = Mock()
    tackle.category.name = "PHYSICAL"
    tackle.priority = 0
    tackle.accuracy = 100

    battle.available_moves = [thunderbolt, tackle]
    battle.available_switches = []

    return battle

def test_picks_higher_damage_move(mock_battle):
    # Create a minimal pool
    from poke.teams.loader import TeamPool
    pool = Mock(spec=TeamPool)
    pool.sample = Mock(return_value=Mock())
    pool.get_ids = Mock(return_value=["team1"])

    agent = MaxDamageAgent.__new__(MaxDamageAgent)
    agent.switch_threshold = 0.25
    agent.create_order = lambda x: x

    move = agent._best_move(mock_battle)

    assert move.id == "thunderbolt"

def test_should_switch_at_low_hp(mock_battle):
    mock_battle.active_pokemon.current_hp_fraction = 0.1

    switch_target = Mock()
    switch_target.current_hp_fraction = 0.8
    mock_battle.available_switches = [switch_target]

    agent = MaxDamageAgent.__new__(MaxDamageAgent)
    agent.switch_threshold = 0.25

    assert agent._should_switch(mock_battle) is True
