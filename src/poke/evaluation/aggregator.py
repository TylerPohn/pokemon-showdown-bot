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
