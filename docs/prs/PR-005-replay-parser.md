COMPLETED

# PR-005: Replay Parser

## Dependencies
- PR-001 (Project Setup)
- PR-004 (Replay Scraper) - for input data

## Overview
Parse raw Pokemon Showdown replay logs into structured battle events. This transforms the text-based battle log into machine-readable format.

## Tech Choices
- **Data Models:** Pydantic for validation
- **Output Format:** JSONL of structured events

## Tasks

### 1. Define data models for battle events
Create `src/poke/data/models.py`:
```python
"""Data models for parsed battle data."""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

class ActionType(str, Enum):
    MOVE = "move"
    SWITCH = "switch"

class Status(str, Enum):
    NONE = ""
    BURN = "brn"
    FREEZE = "frz"
    PARALYSIS = "par"
    POISON = "psn"
    TOXIC = "tox"
    SLEEP = "slp"

class PokemonState(BaseModel):
    """State of a single Pokemon."""
    species: str
    hp_fraction: float  # 0.0 to 1.0
    status: Status = Status.NONE
    fainted: bool = False
    active: bool = False

class FieldState(BaseModel):
    """State of battlefield conditions."""
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    tailwind_p1: bool = False
    tailwind_p2: bool = False

class HazardState(BaseModel):
    """Entry hazards on each side."""
    stealth_rock: bool = False
    spikes: int = 0  # 0-3 layers
    toxic_spikes: int = 0  # 0-2 layers
    sticky_web: bool = False

class PlayerState(BaseModel):
    """State of one player's side."""
    team: List[PokemonState]
    hazards: HazardState = HazardState()

class BattleState(BaseModel):
    """Complete battle state at a point in time."""
    turn: int
    player1: PlayerState
    player2: PlayerState
    field: FieldState = FieldState()

class Action(BaseModel):
    """An action taken by a player."""
    player: str  # "p1" or "p2"
    action_type: ActionType
    target: str  # Move name or Pokemon species

class TurnRecord(BaseModel):
    """Record of a single turn."""
    turn: int
    state_before: BattleState
    p1_action: Optional[Action] = None
    p2_action: Optional[Action] = None
    winner: Optional[str] = None  # Set on final turn

class ParsedBattle(BaseModel):
    """A fully parsed battle."""
    replay_id: str
    format: str
    players: tuple[str, str]
    rating: Optional[int] = None
    winner: Optional[str] = None
    turns: List[TurnRecord]
```

### 2. Implement log parser
Create `src/poke/data/parser.py`:
```python
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
```

### 3. Create batch processing script
Create `scripts/parse_replays.py`:
```python
#!/usr/bin/env python
"""Parse raw replays into structured format."""
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

from poke.data.parser import BattleLogParser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output", help="Output JSONL file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".parsed.jsonl")

    parser = BattleLogParser()
    success = 0
    failed = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in tqdm(f_in, desc="Parsing"):
            replay = json.loads(line)
            parsed = parser.parse(replay)

            if parsed:
                f_out.write(parsed.model_dump_json() + "\n")
                success += 1
            else:
                failed += 1

    print(f"Parsed {success} battles, {failed} failed")

if __name__ == "__main__":
    main()
```

### 4. Write unit tests
Create `tests/data/test_parser.py`:
```python
"""Tests for replay parser."""
import pytest
from poke.data.parser import BattleLogParser
from poke.data.models import ActionType

SAMPLE_LOG = """
|j|☆Player1
|j|☆Player2
|player|p1|Player1|1
|player|p2|Player2|2
|turn|1
|switch|p1a: Pikachu|Pikachu, L50
|switch|p2a: Charizard|Charizard, L50
|move|p1a: Pikachu|Thunderbolt|p2a: Charizard
|-damage|p2a: Charizard|75/100
|turn|2
|move|p2a: Charizard|Flamethrower|p1a: Pikachu
|-damage|p1a: Pikachu|0/100
|faint|p1a: Pikachu
|win|Player2
"""

def test_parse_basic_battle():
    parser = BattleLogParser()
    replay = {
        "id": "test-123",
        "format": "gen9ou",
        "p1": "Player1",
        "p2": "Player2",
        "log": SAMPLE_LOG,
    }

    result = parser.parse(replay)

    assert result is not None
    assert result.replay_id == "test-123"
    assert result.winner == "Player2"
    assert len(result.turns) >= 1

def test_parse_switch_action():
    parser = BattleLogParser()
    replay = {"id": "test", "log": SAMPLE_LOG}

    result = parser.parse(replay)

    # First turn should have switch actions
    turn1 = result.turns[0]
    assert turn1.p1_action.action_type == ActionType.SWITCH
    assert turn1.p1_action.target == "Pikachu"

def test_parse_move_action():
    parser = BattleLogParser()
    replay = {"id": "test", "log": SAMPLE_LOG}

    result = parser.parse(replay)

    # Second turn should have move actions
    if len(result.turns) > 1:
        turn2 = result.turns[1]
        assert turn2.p1_action.action_type == ActionType.MOVE
        assert turn2.p1_action.target == "Thunderbolt"
```

## Acceptance Criteria
- [ ] Parses turn markers correctly
- [ ] Extracts switch and move actions
- [ ] Tracks HP damage across turns
- [ ] Handles faint events
- [ ] Identifies battle winner
- [ ] Produces valid Pydantic models
- [ ] Handles malformed logs gracefully

## Notes
- The Showdown log format is complex with many edge cases
- Start with core events and expand coverage iteratively
- Some log lines will be ignored initially (e.g., chat, spectator joins)

## Estimated Complexity
High - Complex text parsing with many edge cases
