"""Tests for evaluation metrics."""
import pytest

from poke.evaluation.metrics import (
    compute_elo_ratings,
    compute_confidence_interval,
    EloRating,
)

def test_elo_ratings_basic():
    matchups = [
        {"agent1": "A", "agent2": "B", "agent1_wins": 7, "agent2_wins": 3, "total_battles": 10},
    ]

    ratings = compute_elo_ratings(matchups)

    # Winner should have higher rating
    assert ratings["A"].rating > ratings["B"].rating

def test_elo_initial_rating():
    matchups = [
        {"agent1": "A", "agent2": "B", "agent1_wins": 5, "agent2_wins": 5, "total_battles": 10},
    ]

    ratings = compute_elo_ratings(matchups)

    # Even match should keep ratings close to initial
    assert abs(ratings["A"].rating - 1500) < 50
    assert abs(ratings["B"].rating - 1500) < 50

def test_confidence_interval():
    ci_low, ci_high = compute_confidence_interval(wins=70, total=100)

    # 70% winrate should have CI containing 0.70
    assert ci_low < 0.70 < ci_high
    assert ci_low > 0.5  # Significantly above 50%

def test_confidence_interval_small_sample():
    ci_low, ci_high = compute_confidence_interval(wins=7, total=10)

    # Small sample should have wider CI
    assert ci_high - ci_low > 0.2

def test_confidence_interval_edge_cases():
    # Perfect winrate
    ci_low, ci_high = compute_confidence_interval(wins=10, total=10)
    assert ci_high <= 1.0

    # Zero winrate
    ci_low, ci_high = compute_confidence_interval(wins=0, total=10)
    assert ci_low >= 0.0

    # Empty
    ci_low, ci_high = compute_confidence_interval(wins=0, total=0)
    assert ci_low == 0.0
    assert ci_high == 1.0
