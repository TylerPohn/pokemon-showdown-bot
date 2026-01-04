"""Tests for metrics aggregation."""
import pytest
from poke.evaluation.aggregator import MetricsAggregator

@pytest.fixture
def sample_battles():
    return [
        {
            "battle_id": "test1",
            "winner": "agent1",
            "turns": [
                {"player": "agent1", "action_type": "move", "action_name": "thunderbolt"},
                {"player": "agent1", "action_type": "move", "action_name": "thunderbolt"},
                {"player": "agent1", "action_type": "switch", "action_name": "charizard"},
            ],
        },
        {
            "battle_id": "test2",
            "winner": "agent2",
            "turns": [
                {"player": "agent1", "action_type": "move", "action_name": "flamethrower"},
            ],
        },
    ]

def test_compute_metrics(sample_battles):
    aggregator = MetricsAggregator()
    for battle in sample_battles:
        aggregator.add_battle(battle)

    metrics = aggregator.compute_metrics("agent1")

    assert metrics.total_battles == 2
    assert metrics.total_turns == 4
    assert "move" in metrics.action_distribution
    assert "switch" in metrics.action_distribution

def test_move_ranking(sample_battles):
    aggregator = MetricsAggregator()
    for battle in sample_battles:
        aggregator.add_battle(battle)

    ranking = aggregator.get_move_ranking(10)

    assert ranking[0][0] == "thunderbolt"
    assert ranking[0][1] == 2
