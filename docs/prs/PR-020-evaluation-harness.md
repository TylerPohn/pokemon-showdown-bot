COMPLETED

# PR-020: Evaluation Harness

## Dependencies
- PR-003 (poke-env Integration)
- PR-011 (Random Agent)
- PR-012 (Heuristic Agent)
- PR-015 (Policy Network)

## Overview
Implement an evaluation harness to measure agent performance through battles against baselines and policy snapshots.

## Tech Choices
- **Battle Runner:** Async poke-env battles
- **Metrics:** Winrate, Elo estimation, per-matchup stats
- **Output:** JSON reports + CSV summaries

## Tasks

### 1. Create neural network agent
Create `src/poke/agents/nn_agent.py`:
```python
"""Neural network-based battle agent."""
import torch
from typing import Optional

from poke_env.environment import AbstractBattle

from .team_aware import TeamAwareAgent
from ..models.policy import MLPPolicy
from ..models.masking import ActionMask, ActionSpace
from ..models.preprocessing import FeaturePreprocessor
from ..teams.loader import TeamPool

class NeuralNetworkAgent(TeamAwareAgent):
    """Agent that uses a trained neural network policy."""

    def __init__(
        self,
        policy: MLPPolicy,
        team_pool: TeamPool,
        action_space: ActionSpace = None,
        deterministic: bool = False,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(team_pool=team_pool, **kwargs)

        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.deterministic = deterministic

        self.action_space = action_space or ActionSpace()
        self.action_mask = ActionMask(self.action_space)
        self.preprocessor = FeaturePreprocessor()

    def choose_move(self, battle: AbstractBattle):
        """Choose move using the neural network policy."""
        with torch.no_grad():
            # Get observation
            obs = self.get_observation(battle)

            # Preprocess for model
            obs_dict = self.preprocessor.preprocess(obs.__dict__)
            obs_tensors = {
                k: v.unsqueeze(0).to(self.device)
                for k, v in obs_dict.items()
            }

            # Get action mask
            mask = self.action_mask.get_mask_tensor(battle, self.device)
            mask = mask.unsqueeze(0)

            # Get action from policy
            action_probs, _ = self.policy(obs_tensors, mask)

            if self.deterministic:
                action_idx = action_probs.argmax(dim=-1).item()
            else:
                action_idx = torch.multinomial(action_probs, 1).item()

        # Convert to battle order
        return self._action_to_order(action_idx, battle)

    def _action_to_order(self, action_idx: int, battle: AbstractBattle):
        """Convert action index to poke-env order."""
        action_type, target_idx = self.action_space.decode_action(action_idx)

        if action_type == "move":
            if target_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[target_idx])
        else:
            if target_idx < len(battle.available_switches):
                return self.create_order(battle.available_switches[target_idx])

        # Fallback
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])

        return self.choose_default_move()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        team_pool: TeamPool,
        **kwargs
    ) -> "NeuralNetworkAgent":
        """Load agent from checkpoint."""
        from ..models.config import EncoderConfig
        from ..models.factory import create_policy

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create policy
        config = EncoderConfig()
        policy = create_policy("mlp", encoder_config=config)
        policy.load_state_dict(checkpoint["model_state_dict"])

        return cls(policy=policy, team_pool=team_pool, **kwargs)
```

### 2. Create battle runner
Create `src/poke/evaluation/runner.py`:
```python
"""Battle evaluation runner."""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from poke_env.player import Player

logger = logging.getLogger(__name__)

@dataclass
class BattleResult:
    """Result of a single battle."""
    battle_id: str
    winner: str
    loser: str
    turns: int
    timestamp: str

@dataclass
class MatchupResult:
    """Results of battles between two agents."""
    agent1_name: str
    agent2_name: str
    agent1_wins: int = 0
    agent2_wins: int = 0
    total_battles: int = 0
    battles: List[BattleResult] = field(default_factory=list)

    @property
    def agent1_winrate(self) -> float:
        if self.total_battles == 0:
            return 0.0
        return self.agent1_wins / self.total_battles

    def to_dict(self) -> dict:
        return {
            "agent1": self.agent1_name,
            "agent2": self.agent2_name,
            "agent1_wins": self.agent1_wins,
            "agent2_wins": self.agent2_wins,
            "total_battles": self.total_battles,
            "agent1_winrate": self.agent1_winrate,
        }


class BattleRunner:
    """Run evaluation battles between agents."""

    def __init__(
        self,
        battle_format: str = "gen9ou",
        concurrent_battles: int = 1,
    ):
        self.battle_format = battle_format
        self.concurrent_battles = concurrent_battles

    async def run_matchup(
        self,
        agent1: Player,
        agent2: Player,
        n_battles: int = 100,
    ) -> MatchupResult:
        """Run battles between two agents.

        Args:
            agent1: First agent
            agent2: Second agent
            n_battles: Number of battles to run

        Returns:
            MatchupResult with statistics
        """
        result = MatchupResult(
            agent1_name=agent1.__class__.__name__,
            agent2_name=agent2.__class__.__name__,
        )

        logger.info(f"Running {n_battles} battles: {result.agent1_name} vs {result.agent2_name}")

        await agent1.battle_against(agent2, n_battles=n_battles)

        result.agent1_wins = agent1.n_won_battles
        result.agent2_wins = agent2.n_won_battles
        result.total_battles = n_battles

        logger.info(f"Results: {result.agent1_wins}-{result.agent2_wins}")

        return result

    async def run_tournament(
        self,
        agents: List[Player],
        n_battles_per_matchup: int = 50,
    ) -> Dict[Tuple[str, str], MatchupResult]:
        """Run round-robin tournament between agents.

        Args:
            agents: List of agents to compete
            n_battles_per_matchup: Battles per pairing

        Returns:
            Dict mapping (agent1, agent2) to MatchupResult
        """
        results = {}

        for i, agent1 in enumerate(agents):
            for agent2 in agents[i + 1:]:
                key = (agent1.__class__.__name__, agent2.__class__.__name__)
                results[key] = await self.run_matchup(
                    agent1, agent2, n_battles_per_matchup
                )

        return results


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    matchups: List[MatchupResult]
    agent_stats: Dict[str, Dict]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "matchups": [m.to_dict() for m in self.matchups],
            "agent_stats": self.agent_stats,
        }

    def save(self, path: Path) -> None:
        """Save report to JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_results(cls, results: Dict) -> "EvaluationReport":
        """Create report from tournament results."""
        matchups = list(results.values())

        # Compute per-agent stats
        agent_stats: Dict[str, Dict] = {}
        for matchup in matchups:
            for name in [matchup.agent1_name, matchup.agent2_name]:
                if name not in agent_stats:
                    agent_stats[name] = {"wins": 0, "losses": 0, "total": 0}

            agent_stats[matchup.agent1_name]["wins"] += matchup.agent1_wins
            agent_stats[matchup.agent1_name]["losses"] += matchup.agent2_wins
            agent_stats[matchup.agent1_name]["total"] += matchup.total_battles

            agent_stats[matchup.agent2_name]["wins"] += matchup.agent2_wins
            agent_stats[matchup.agent2_name]["losses"] += matchup.agent1_wins
            agent_stats[matchup.agent2_name]["total"] += matchup.total_battles

        # Compute winrates
        for name, stats in agent_stats.items():
            stats["winrate"] = stats["wins"] / stats["total"] if stats["total"] > 0 else 0

        return cls(
            timestamp=datetime.now().isoformat(),
            matchups=matchups,
            agent_stats=agent_stats,
        )
```

### 3. Create metrics calculator
Create `src/poke/evaluation/metrics.py`:
```python
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
```

### 4. Create evaluation script
Create `scripts/evaluate.py`:
```python
#!/usr/bin/env python
"""Evaluate trained agents."""
import argparse
import asyncio
import logging
from pathlib import Path

from poke.agents.random_agent import PureRandomAgent
from poke.agents.heuristic_agent import MaxDamageAgent
from poke.agents.nn_agent import NeuralNetworkAgent
from poke.teams.loader import get_default_pool
from poke.evaluation.runner import BattleRunner, EvaluationReport
from poke.evaluation.metrics import compute_elo_ratings, compute_agent_metrics
from poke.config import BattleConfig

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--battles", type=int, default=100, help="Battles per matchup")
    parser.add_argument("--output", default="evaluation_report.json")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = BattleConfig()
    pool = get_default_pool()

    # Create agents
    agents = [
        PureRandomAgent(
            battle_format=config.battle_format,
            server_configuration=config.server_configuration,
        ),
        MaxDamageAgent(
            team_pool=pool,
            battle_format=config.battle_format,
            server_configuration=config.server_configuration,
        ),
        NeuralNetworkAgent.from_checkpoint(
            args.checkpoint,
            team_pool=pool,
            deterministic=args.deterministic,
            battle_format=config.battle_format,
            server_configuration=config.server_configuration,
        ),
    ]

    # Run tournament
    runner = BattleRunner(battle_format=config.battle_format)
    results = await runner.run_tournament(agents, n_battles_per_matchup=args.battles)

    # Create report
    report = EvaluationReport.from_results(results)

    # Compute Elo
    elo_ratings = compute_elo_ratings([m.to_dict() for m in report.matchups])
    agent_metrics = compute_agent_metrics(report.agent_stats, elo_ratings)

    # Print summary
    print("\n=== Evaluation Results ===\n")
    for name, metrics in sorted(agent_metrics.items(), key=lambda x: -x[1].elo_rating):
        print(f"{name}:")
        print(f"  Winrate: {metrics.winrate:.1%} ({metrics.winrate_ci_low:.1%}-{metrics.winrate_ci_high:.1%})")
        print(f"  Elo: {metrics.elo_rating:.0f}")
        print(f"  Record: {metrics.total_wins}W-{metrics.total_losses}L")
        print()

    # Save report
    report.save(Path(args.output))
    print(f"Report saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Write tests
Create `tests/evaluation/test_metrics.py`:
```python
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
```

## Acceptance Criteria
- [ ] Neural network agent runs battles correctly
- [ ] Battle runner executes matchups
- [ ] Elo ratings computed correctly
- [ ] Confidence intervals calculated
- [ ] JSON report generated
- [ ] Works with Showdown server

## Success Metrics
- Trained agent: >60% winrate vs random
- Trained agent: >50% winrate vs heuristic (target: 55%+)
- Zero illegal moves

## Estimated Complexity
Medium-High - Integration of agent, battles, and metrics
