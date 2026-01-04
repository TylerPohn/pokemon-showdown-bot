COMPLETED

# PR-021: Metrics Collection System

## Dependencies
- PR-020 (Evaluation Harness)

## Overview
Implement comprehensive metrics collection for detailed analysis of agent behavior, including per-turn decisions, action distributions, and game statistics.

## Tech Choices
- **Storage:** SQLite for structured data
- **Format:** Also export to CSV/JSON
- **Visualization:** Optional matplotlib integration

## Tasks

### 1. Create battle logger
Create `src/poke/evaluation/battle_logger.py`:
```python
"""Detailed battle logging."""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

from poke_env.environment import AbstractBattle

@dataclass
class TurnLog:
    """Log of a single turn."""
    turn: int
    player: str
    action_type: str  # "move" or "switch"
    action_name: str
    available_moves: List[str]
    available_switches: List[str]
    own_active_hp: float
    opp_active_hp: float
    own_team_alive: int
    opp_team_revealed: int
    weather: Optional[str] = None
    field_conditions: List[str] = field(default_factory=list)

@dataclass
class BattleLog:
    """Complete log of a battle."""
    battle_id: str
    player1: str
    player2: str
    winner: str
    turns: List[TurnLog] = field(default_factory=list)
    final_hp_diff: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "battle_id": self.battle_id,
            "player1": self.player1,
            "player2": self.player2,
            "winner": self.winner,
            "num_turns": len(self.turns),
            "final_hp_diff": self.final_hp_diff,
            "turns": [asdict(t) for t in self.turns],
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save battle log to JSON."""
        path.write_text(json.dumps(self.to_dict(), indent=2))


class BattleLogger:
    """Logger that records detailed battle information."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.battles: List[BattleLog] = []
        self.current_battle: Optional[BattleLog] = None

    def start_battle(self, battle: AbstractBattle, player1: str, player2: str) -> None:
        """Start logging a new battle."""
        self.current_battle = BattleLog(
            battle_id=battle.battle_tag,
            player1=player1,
            player2=player2,
            winner="",
        )

    def log_turn(
        self,
        battle: AbstractBattle,
        player: str,
        action_type: str,
        action_name: str,
    ) -> None:
        """Log a turn."""
        if self.current_battle is None:
            return

        turn_log = TurnLog(
            turn=battle.turn,
            player=player,
            action_type=action_type,
            action_name=action_name,
            available_moves=[m.id for m in battle.available_moves],
            available_switches=[p.species for p in battle.available_switches],
            own_active_hp=battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0,
            opp_active_hp=battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 1,
            own_team_alive=sum(1 for p in battle.team.values() if not p.fainted),
            opp_team_revealed=len(battle.opponent_team),
            weather=str(battle.weather) if battle.weather else None,
        )

        self.current_battle.turns.append(turn_log)

    def end_battle(self, winner: str) -> BattleLog:
        """End current battle and return log."""
        if self.current_battle is None:
            raise ValueError("No battle in progress")

        self.current_battle.winner = winner
        self.battles.append(self.current_battle)

        # Save individual battle
        path = self.output_dir / f"{self.current_battle.battle_id}.json"
        self.current_battle.save(path)

        result = self.current_battle
        self.current_battle = None
        return result

    def save_all(self) -> None:
        """Save all battle logs to a combined file."""
        combined_path = self.output_dir / "all_battles.json"
        data = [b.to_dict() for b in self.battles]
        combined_path.write_text(json.dumps(data, indent=2))
```

### 2. Create metrics aggregator
Create `src/poke/evaluation/aggregator.py`:
```python
"""Aggregate metrics from battle logs."""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json

@dataclass
class AggregateMetrics:
    """Aggregated metrics from multiple battles."""
    total_battles: int
    total_turns: int
    avg_game_length: float
    action_distribution: Dict[str, float]
    move_usage: Dict[str, int]
    switch_frequency: float
    avg_remaining_hp: float
    win_by_turns: Dict[str, float]  # Binned by game length

class MetricsAggregator:
    """Aggregate metrics from battle logs."""

    def __init__(self):
        self.battles = []

    def add_battle(self, battle_data: dict) -> None:
        """Add a battle to aggregate."""
        self.battles.append(battle_data)

    def load_from_directory(self, dir_path: Path) -> None:
        """Load all battle logs from a directory."""
        for file_path in dir_path.glob("*.json"):
            if file_path.name == "all_battles.json":
                continue
            with open(file_path) as f:
                self.add_battle(json.load(f))

    def compute_metrics(self, agent_name: str) -> AggregateMetrics:
        """Compute aggregate metrics for a specific agent."""
        total_turns = 0
        action_counts = defaultdict(int)
        move_usage = defaultdict(int)
        switch_count = 0
        total_actions = 0
        remaining_hp_sum = 0
        wins_by_length = defaultdict(int)
        total_by_length = defaultdict(int)

        for battle in self.battles:
            num_turns = len(battle["turns"])
            total_turns += num_turns

            # Bin by game length
            length_bin = "short" if num_turns < 20 else "medium" if num_turns < 40 else "long"
            total_by_length[length_bin] += 1
            if battle["winner"] == agent_name:
                wins_by_length[length_bin] += 1

            for turn in battle["turns"]:
                if turn["player"] == agent_name:
                    action_type = turn["action_type"]
                    action_counts[action_type] += 1
                    total_actions += 1

                    if action_type == "move":
                        move_usage[turn["action_name"]] += 1
                    else:
                        switch_count += 1

        # Compute aggregates
        avg_length = total_turns / len(self.battles) if self.battles else 0

        action_dist = {
            k: v / total_actions if total_actions > 0 else 0
            for k, v in action_counts.items()
        }

        switch_freq = switch_count / total_actions if total_actions > 0 else 0

        win_by_turns = {
            k: wins_by_length[k] / total_by_length[k] if total_by_length[k] > 0 else 0
            for k in ["short", "medium", "long"]
        }

        return AggregateMetrics(
            total_battles=len(self.battles),
            total_turns=total_turns,
            avg_game_length=avg_length,
            action_distribution=action_dist,
            move_usage=dict(move_usage),
            switch_frequency=switch_freq,
            avg_remaining_hp=0,  # Would need to compute from final state
            win_by_turns=win_by_turns,
        )

    def get_move_ranking(self, top_n: int = 20) -> List[tuple]:
        """Get most used moves."""
        move_counts = defaultdict(int)

        for battle in self.battles:
            for turn in battle["turns"]:
                if turn["action_type"] == "move":
                    move_counts[turn["action_name"]] += 1

        return sorted(move_counts.items(), key=lambda x: -x[1])[:top_n]

    def get_illegal_action_rate(self) -> float:
        """Compute rate of illegal action attempts."""
        # This would need integration with action masking logs
        return 0.0
```

### 3. Create visualization utilities
Create `src/poke/evaluation/visualization.py`:
```python
"""Visualization utilities for evaluation metrics."""
from typing import Dict, List, Optional
from pathlib import Path

def create_winrate_chart(
    agent_names: List[str],
    winrates: List[float],
    ci_lows: List[float],
    ci_highs: List[float],
    output_path: Optional[Path] = None,
) -> None:
    """Create bar chart of winrates with confidence intervals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(agent_names))
    yerr = [[w - l for w, l in zip(winrates, ci_lows)],
            [h - w for w, h in zip(winrates, ci_highs)]]

    bars = ax.bar(x, winrates, yerr=yerr, capsize=5, color='steelblue', alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='50% baseline')

    ax.set_ylabel('Winrate')
    ax.set_title('Agent Winrates vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def create_action_distribution_pie(
    action_dist: Dict[str, float],
    output_path: Optional[Path] = None,
) -> None:
    """Create pie chart of action distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    labels = list(action_dist.keys())
    sizes = list(action_dist.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title('Action Distribution')

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def create_elo_progression(
    agents: List[str],
    elo_history: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> None:
    """Create line chart of Elo rating progression."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for agent in agents:
        if agent in elo_history:
            ax.plot(elo_history[agent], label=agent)

    ax.set_xlabel('Games')
    ax.set_ylabel('Elo Rating')
    ax.set_title('Elo Rating Progression')
    ax.legend()
    ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5)

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()
```

### 4. Create metrics report script
Create `scripts/generate_report.py`:
```python
#!/usr/bin/env python
"""Generate evaluation report from battle logs."""
import argparse
from pathlib import Path
import json

from poke.evaluation.aggregator import MetricsAggregator
from poke.evaluation.visualization import (
    create_winrate_chart,
    create_action_distribution_pie,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True, help="Directory with battle logs")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--agent", required=True, help="Agent name to analyze")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and aggregate metrics
    aggregator = MetricsAggregator()
    aggregator.load_from_directory(logs_dir)

    metrics = aggregator.compute_metrics(args.agent)

    # Print summary
    print(f"\n=== Metrics for {args.agent} ===\n")
    print(f"Total battles: {metrics.total_battles}")
    print(f"Average game length: {metrics.avg_game_length:.1f} turns")
    print(f"Switch frequency: {metrics.switch_frequency:.1%}")
    print(f"\nAction distribution:")
    for action, pct in metrics.action_distribution.items():
        print(f"  {action}: {pct:.1%}")

    print(f"\nWinrate by game length:")
    for length, wr in metrics.win_by_turns.items():
        print(f"  {length}: {wr:.1%}")

    # Top moves
    print(f"\nTop 10 moves:")
    for move, count in aggregator.get_move_ranking(10):
        print(f"  {move}: {count}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "total_battles": metrics.total_battles,
            "avg_game_length": metrics.avg_game_length,
            "action_distribution": metrics.action_distribution,
            "switch_frequency": metrics.switch_frequency,
            "win_by_turns": metrics.win_by_turns,
            "top_moves": aggregator.get_move_ranking(20),
        }, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")

    # Generate visualizations
    create_action_distribution_pie(
        metrics.action_distribution,
        output_dir / "action_distribution.png"
    )

if __name__ == "__main__":
    main()
```

### 5. Write tests
Create `tests/evaluation/test_aggregator.py`:
```python
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
```

## Acceptance Criteria
- [ ] Battle logger captures turn-by-turn data
- [ ] Metrics aggregator computes statistics correctly
- [ ] Move usage tracked
- [ ] Switch frequency calculated
- [ ] Visualizations generated (when matplotlib available)
- [ ] JSON reports saved

## Notes
- Detailed logging can slow down evaluation; use sparingly
- Keep logs for debugging and analysis
- SQLite could be added for larger-scale analysis

## Estimated Complexity
Medium - Data aggregation and optional visualization
