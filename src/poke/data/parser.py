"""Parser for Pokemon Showdown battle logs."""
import re
import logging
from typing import Optional, Iterator

from .models import (
    ParsedBattle, TurnRecord, BattleState, PlayerState,
    PokemonState, Action, ActionType, FieldState, HazardState, Status
)

logger = logging.getLogger(__name__)

# Regex patterns for log parsing
PATTERNS = {
    "turn": re.compile(r"\|turn\|(\d+)"),
    "switch": re.compile(r"\|switch\|p(\d)a: ([^|]+)\|([^|]+)"),
    "move": re.compile(r"\|move\|p(\d)a: ([^|]+)\|([^|]+)"),
    "damage": re.compile(r"\|-damage\|p(\d)a: ([^|]+)\|(\d+)/(\d+)"),
    "faint": re.compile(r"\|faint\|p(\d)a: ([^|]+)"),
    "win": re.compile(r"\|win\|(.+)"),
    "weather": re.compile(r"\|-weather\|(\w+)"),
    "status": re.compile(r"\|-status\|p(\d)a: ([^|]+)\|(\w+)"),
    "hazard": re.compile(r"\|-sidestart\|p(\d): .+\|(.+)"),
    "hazard_end": re.compile(r"\|-sideend\|p(\d): .+\|(.+)"),
}

class BattleLogParser:
    """Parser for Pokemon Showdown battle logs."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset parser state."""
        self._p1_team: dict[str, PokemonState] = {}
        self._p2_team: dict[str, PokemonState] = {}
        self._p1_active: Optional[str] = None
        self._p2_active: Optional[str] = None
        self._p1_hazards = HazardState()
        self._p2_hazards = HazardState()
        self._field = FieldState()
        self._turn = 0
        self._turns: list[TurnRecord] = []
        self._pending_actions: dict[str, Action] = {}

    def parse(self, replay: dict) -> Optional[ParsedBattle]:
        """Parse a replay dict into structured battle data.

        Args:
            replay: Raw replay data from API

        Returns:
            ParsedBattle or None if parsing fails
        """
        self.reset()

        try:
            log = replay.get("log", "")
            lines = log.split("\n")

            for line in lines:
                self._process_line(line)

            return ParsedBattle(
                replay_id=replay["id"],
                format=replay.get("format", "unknown"),
                players=(replay.get("p1", ""), replay.get("p2", "")),
                rating=replay.get("rating"),
                winner=self._find_winner(lines),
                turns=self._turns,
            )

        except Exception as e:
            logger.warning(f"Failed to parse {replay.get('id', 'unknown')}: {e}")
            return None

    def _process_line(self, line: str) -> None:
        """Process a single log line."""
        # Turn marker
        if match := PATTERNS["turn"].match(line):
            self._on_new_turn(int(match.group(1)))

        # Switch
        elif match := PATTERNS["switch"].match(line):
            player = f"p{match.group(1)}"
            species = match.group(3).split(",")[0]
            self._on_switch(player, species)

        # Move
        elif match := PATTERNS["move"].match(line):
            player = f"p{match.group(1)}"
            move = match.group(3)
            self._on_move(player, move)

        # Damage
        elif match := PATTERNS["damage"].match(line):
            player = f"p{match.group(1)}"
            current = int(match.group(3))
            max_hp = int(match.group(4))
            self._on_damage(player, current / max_hp if max_hp > 0 else 0)

        # Faint
        elif match := PATTERNS["faint"].match(line):
            player = f"p{match.group(1)}"
            self._on_faint(player)

        # Weather
        elif match := PATTERNS["weather"].match(line):
            self._field.weather = match.group(1)

        # Status
        elif match := PATTERNS["status"].match(line):
            player = f"p{match.group(1)}"
            status = match.group(3)
            self._on_status(player, status)

    def _on_new_turn(self, turn: int) -> None:
        """Handle turn transition."""
        if self._turn > 0:
            # Save the previous turn
            state = self._get_current_state()
            record = TurnRecord(
                turn=self._turn,
                state_before=state,
                p1_action=self._pending_actions.get("p1"),
                p2_action=self._pending_actions.get("p2"),
            )
            self._turns.append(record)
            self._pending_actions.clear()

        self._turn = turn

    def _on_switch(self, player: str, species: str) -> None:
        """Handle Pokemon switch."""
        team = self._p1_team if player == "p1" else self._p2_team

        # Add to team if new
        if species not in team:
            team[species] = PokemonState(species=species, hp_fraction=1.0)

        # Update active
        if player == "p1":
            self._p1_active = species
        else:
            self._p2_active = species

        # Record action
        self._pending_actions[player] = Action(
            player=player,
            action_type=ActionType.SWITCH,
            target=species,
        )

    def _on_move(self, player: str, move: str) -> None:
        """Handle move usage."""
        self._pending_actions[player] = Action(
            player=player,
            action_type=ActionType.MOVE,
            target=move,
        )

    def _on_damage(self, player: str, hp_fraction: float) -> None:
        """Handle damage to active Pokemon."""
        active = self._p1_active if player == "p1" else self._p2_active
        team = self._p1_team if player == "p1" else self._p2_team
        if active and active in team:
            team[active].hp_fraction = hp_fraction

    def _on_faint(self, player: str) -> None:
        """Handle Pokemon fainting."""
        active = self._p1_active if player == "p1" else self._p2_active
        team = self._p1_team if player == "p1" else self._p2_team
        if active and active in team:
            team[active].fainted = True
            team[active].hp_fraction = 0.0

    def _on_status(self, player: str, status_str: str) -> None:
        """Handle status condition."""
        active = self._p1_active if player == "p1" else self._p2_active
        team = self._p1_team if player == "p1" else self._p2_team
        if active and active in team:
            try:
                team[active].status = Status(status_str)
            except ValueError:
                pass  # Unknown status

    def _get_current_state(self) -> BattleState:
        """Get current battle state snapshot."""
        return BattleState(
            turn=self._turn,
            player1=PlayerState(
                team=list(self._p1_team.values()),
                hazards=self._p1_hazards,
            ),
            player2=PlayerState(
                team=list(self._p2_team.values()),
                hazards=self._p2_hazards,
            ),
            field=self._field,
        )

    def _find_winner(self, lines: list[str]) -> Optional[str]:
        """Find winner from log lines."""
        for line in reversed(lines):
            if match := PATTERNS["win"].match(line):
                return match.group(1)
        return None
