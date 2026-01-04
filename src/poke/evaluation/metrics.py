"""Evaluation metrics."""
import math
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class EloRating:
    """Elo rating for an agent."""
    name: str
    rating: float = 1500.0
    games_played: int = 0

def compute_elo_ratings(
    matchups: List[Dict],
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
) -> Dict[str, EloRating]:
    """Compute Elo ratings from matchup results.

    Args:
        matchups: List of matchup result dicts
        k_factor: Elo K-factor
        initial_rating: Starting rating

    Returns:
        Dict mapping agent name to EloRating
    """
    ratings: Dict[str, EloRating] = {}

    for matchup in matchups:
        agent1 = matchup["agent1"]
        agent2 = matchup["agent2"]

        # Initialize if needed
        if agent1 not in ratings:
            ratings[agent1] = EloRating(name=agent1, rating=initial_rating)
        if agent2 not in ratings:
            ratings[agent2] = EloRating(name=agent2, rating=initial_rating)

        r1 = ratings[agent1].rating
        r2 = ratings[agent2].rating

        # Expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 - e1

        # Actual scores
        total = matchup["total_battles"]
        s1 = matchup["agent1_wins"] / total
        s2 = matchup["agent2_wins"] / total

        # Update ratings
        ratings[agent1].rating += k_factor * total * (s1 - e1)
        ratings[agent2].rating += k_factor * total * (s2 - e2)
        ratings[agent1].games_played += total
        ratings[agent2].games_played += total

    return ratings

def compute_confidence_interval(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for winrate.

    Uses Wilson score interval.
    """
    if total == 0:
        return (0.0, 1.0)

    z = 1.96  # 95% confidence
    p = wins / total
    n = total

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - spread), min(1, center + spread))

@dataclass
class AgentMetrics:
    """Comprehensive metrics for an agent."""
    name: str
    total_wins: int
    total_losses: int
    winrate: float
    winrate_ci_low: float
    winrate_ci_high: float
    elo_rating: float

def compute_agent_metrics(
    agent_stats: Dict[str, Dict],
    elo_ratings: Dict[str, EloRating],
) -> Dict[str, AgentMetrics]:
    """Compute comprehensive metrics for all agents."""
    metrics = {}

    for name, stats in agent_stats.items():
        ci_low, ci_high = compute_confidence_interval(
            stats["wins"], stats["total"]
        )

        metrics[name] = AgentMetrics(
            name=name,
            total_wins=stats["wins"],
            total_losses=stats["losses"],
            winrate=stats["winrate"],
            winrate_ci_low=ci_low,
            winrate_ci_high=ci_high,
            elo_rating=elo_ratings.get(name, EloRating(name)).rating,
        )

    return metrics
