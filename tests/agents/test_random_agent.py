"""Tests for random agent."""
import pytest
from unittest.mock import Mock, MagicMock

from poke.agents.random_agent import RandomAgent, PureRandomAgent

@pytest.fixture
def mock_battle():
    battle = Mock()

    # Create mock moves
    move1 = Mock()
    move1.id = "thunderbolt"
    move2 = Mock()
    move2.id = "voltswitch"

    battle.available_moves = [move1, move2]

    # Create mock switches
    switch1 = Mock()
    switch1.species = "Charizard"
    battle.available_switches = [switch1]

    battle.team = {f"mon{i}": Mock() for i in range(6)}

    return battle

def test_pure_random_chooses_legal_action(mock_battle):
    agent = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent.create_order = lambda x: x  # Simplified

    result = agent.choose_move(mock_battle)

    # Should be one of the available actions
    all_actions = mock_battle.available_moves + mock_battle.available_switches
    assert result in all_actions

def test_random_agent_deterministic_with_seed(mock_battle):
    agent1 = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent2 = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent1.create_order = lambda x: x
    agent2.create_order = lambda x: x

    results1 = [agent1.choose_move(mock_battle) for _ in range(10)]
    results2 = [agent2.choose_move(mock_battle) for _ in range(10)]

    assert results1 == results2

def test_random_agent_different_with_different_seeds(mock_battle):
    agent1 = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent2 = PureRandomAgent(seed=123, battle_format="gen9ou")
    agent1.create_order = lambda x: x
    agent2.create_order = lambda x: x

    results1 = [agent1.choose_move(mock_battle) for _ in range(100)]
    results2 = [agent2.choose_move(mock_battle) for _ in range(100)]

    # Very unlikely to be identical
    assert results1 != results2

def test_teampreview_shuffles_order():
    agent = PureRandomAgent(seed=42, battle_format="gen9ou")

    battle = Mock()
    battle.team = {f"mon{i}": Mock() for i in range(6)}

    result = agent.teampreview(battle)

    assert result.startswith("/team ")
    order = result.replace("/team ", "")
    assert len(order) == 6
    assert set(order) == set("123456")
