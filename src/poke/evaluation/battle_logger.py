"""Detailed battle logging."""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path

from poke_env.battle import AbstractBattle

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
