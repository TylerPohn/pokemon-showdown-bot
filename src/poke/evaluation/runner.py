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
