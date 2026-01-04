"""Statistics collection for battle data."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path
import json

from .models import ParsedBattle

@dataclass
class DatasetStatistics:
    """Statistics about the dataset."""
    total_battles: int = 0
    total_turns: int = 0
    total_actions: int = 0

    # Distributions
    turns_per_battle: List[int] = field(default_factory=list)
    species_counts: Dict[str, int] = field(default_factory=dict)
    move_counts: Dict[str, int] = field(default_factory=dict)
    win_by_player: Dict[str, int] = field(default_factory=dict)

    # Computed metrics
    @property
    def avg_turns(self) -> float:
        if not self.turns_per_battle:
            return 0.0
        return sum(self.turns_per_battle) / len(self.turns_per_battle)

    @property
    def unique_species(self) -> int:
        return len(self.species_counts)

    @property
    def unique_moves(self) -> int:
        return len(self.move_counts)

    def to_dict(self) -> dict:
        return {
            "total_battles": self.total_battles,
            "total_turns": self.total_turns,
            "total_actions": self.total_actions,
            "avg_turns": self.avg_turns,
            "unique_species": self.unique_species,
            "unique_moves": self.unique_moves,
            "top_species": sorted(self.species_counts.items(), key=lambda x: -x[1])[:20],
            "top_moves": sorted(self.move_counts.items(), key=lambda x: -x[1])[:20],
        }

class StatisticsCollector:
    """Collect statistics from battle data."""

    def __init__(self):
        self.stats = DatasetStatistics()

    def process_battle(self, battle: ParsedBattle) -> None:
        """Process a single battle."""
        self.stats.total_battles += 1
        self.stats.total_turns += len(battle.turns)
        self.stats.turns_per_battle.append(len(battle.turns))

        # Track winner
        if battle.winner:
            player_idx = "p1" if battle.winner == battle.players[0] else "p2"
            self.stats.win_by_player[player_idx] = self.stats.win_by_player.get(player_idx, 0) + 1

        # Track species and moves
        for turn in battle.turns:
            # Species
            for p in turn.state_before.player1.team:
                self.stats.species_counts[p.species] = self.stats.species_counts.get(p.species, 0) + 1
            for p in turn.state_before.player2.team:
                self.stats.species_counts[p.species] = self.stats.species_counts.get(p.species, 0) + 1

            # Actions
            for action in [turn.p1_action, turn.p2_action]:
                if action:
                    self.stats.total_actions += 1
                    if action.action_type.value == "move":
                        self.stats.move_counts[action.target] = self.stats.move_counts.get(action.target, 0) + 1

    def process_file(self, path: Path) -> DatasetStatistics:
        """Process all battles in a file."""
        with open(path) as f:
            for line in f:
                try:
                    battle = ParsedBattle.model_validate_json(line)
                    self.process_battle(battle)
                except Exception:
                    pass
        return self.stats

    def save_report(self, path: Path) -> None:
        """Save statistics report to JSON."""
        path.write_text(json.dumps(self.stats.to_dict(), indent=2))
