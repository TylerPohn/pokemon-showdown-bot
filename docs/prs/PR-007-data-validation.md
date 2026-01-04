COMPLETED

# PR-007: Data Validation and Statistics

## Dependencies
- PR-001 (Project Setup)
- PR-005 (Replay Parser)
- PR-006 (Trajectory Converter)

## Overview
Validate the quality of parsed data and compute statistics to ensure the dataset is suitable for training.

## Tech Choices
- **Validation Framework:** Pydantic validators + custom checks
- **Statistics:** pandas for aggregation
- **Visualization:** matplotlib (optional)

## Tasks

### 1. Create data validator
Create `src/poke/data/validation.py`:
```python
"""Data validation utilities."""
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import json

from .models import ParsedBattle, ActionType

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of validating a single battle."""
    replay_id: str
    valid: bool
    errors: List[str]
    warnings: List[str]

@dataclass
class ValidationReport:
    """Aggregate validation report."""
    total_battles: int
    valid_battles: int
    invalid_battles: int
    error_counts: dict[str, int]
    warning_counts: dict[str, int]

class BattleValidator:
    """Validator for parsed battle data."""

    def __init__(self, min_turns: int = 3, max_turns: int = 200):
        self.min_turns = min_turns
        self.max_turns = max_turns

    def validate(self, battle: ParsedBattle) -> ValidationResult:
        """Validate a single parsed battle."""
        errors = []
        warnings = []

        # Check required fields
        if not battle.replay_id:
            errors.append("missing_replay_id")

        if not battle.turns:
            errors.append("no_turns")

        # Check turn count
        if len(battle.turns) < self.min_turns:
            warnings.append("too_few_turns")

        if len(battle.turns) > self.max_turns:
            warnings.append("too_many_turns")

        # Check for winner
        if battle.winner is None:
            warnings.append("no_winner")

        # Check turn consistency
        for i, turn in enumerate(battle.turns):
            expected_turn = i + 1
            if turn.turn != expected_turn:
                errors.append(f"turn_mismatch_{i}")
                break

        # Check for actions
        turns_without_actions = sum(
            1 for t in battle.turns
            if t.p1_action is None and t.p2_action is None
        )
        if turns_without_actions > len(battle.turns) * 0.5:
            warnings.append("many_turns_without_actions")

        # Check team sizes
        for turn in battle.turns:
            if len(turn.state_before.player1.team) > 6:
                errors.append("team_size_exceeded_p1")
                break
            if len(turn.state_before.player2.team) > 6:
                errors.append("team_size_exceeded_p2")
                break

        return ValidationResult(
            replay_id=battle.replay_id,
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_file(self, path: Path) -> ValidationReport:
        """Validate all battles in a JSONL file."""
        results = []

        with open(path) as f:
            for line in f:
                try:
                    battle = ParsedBattle.model_validate_json(line)
                    result = self.validate(battle)
                    results.append(result)
                except Exception as e:
                    results.append(ValidationResult(
                        replay_id="unknown",
                        valid=False,
                        errors=["parse_error"],
                        warnings=[],
                    ))

        # Aggregate
        error_counts: dict[str, int] = {}
        warning_counts: dict[str, int] = {}

        for r in results:
            for e in r.errors:
                error_counts[e] = error_counts.get(e, 0) + 1
            for w in r.warnings:
                warning_counts[w] = warning_counts.get(w, 0) + 1

        return ValidationReport(
            total_battles=len(results),
            valid_battles=sum(1 for r in results if r.valid),
            invalid_battles=sum(1 for r in results if not r.valid),
            error_counts=error_counts,
            warning_counts=warning_counts,
        )
```

### 2. Create statistics collector
Create `src/poke/data/statistics.py`:
```python
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
```

### 3. Create CLI scripts
Create `scripts/validate_data.py`:
```python
#!/usr/bin/env python
"""Validate parsed battle data."""
import argparse
from pathlib import Path

from poke.data.validation import BattleValidator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parsed battles JSONL")
    args = parser.parse_args()

    validator = BattleValidator()
    report = validator.validate_file(Path(args.input))

    print(f"\n=== Validation Report ===")
    print(f"Total battles: {report.total_battles}")
    print(f"Valid battles: {report.valid_battles} ({100*report.valid_battles/report.total_battles:.1f}%)")
    print(f"Invalid battles: {report.invalid_battles}")

    if report.error_counts:
        print(f"\nErrors:")
        for error, count in sorted(report.error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")

    if report.warning_counts:
        print(f"\nWarnings:")
        for warning, count in sorted(report.warning_counts.items(), key=lambda x: -x[1]):
            print(f"  {warning}: {count}")

if __name__ == "__main__":
    main()
```

Create `scripts/compute_stats.py`:
```python
#!/usr/bin/env python
"""Compute dataset statistics."""
import argparse
from pathlib import Path

from poke.data.statistics import StatisticsCollector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parsed battles JSONL")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    collector = StatisticsCollector()
    stats = collector.process_file(Path(args.input))

    print(f"\n=== Dataset Statistics ===")
    print(f"Total battles: {stats.total_battles}")
    print(f"Total turns: {stats.total_turns}")
    print(f"Average turns per battle: {stats.avg_turns:.1f}")
    print(f"Unique species: {stats.unique_species}")
    print(f"Unique moves: {stats.unique_moves}")

    print(f"\nTop 10 Species:")
    for species, count in sorted(stats.species_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {species}: {count}")

    print(f"\nTop 10 Moves:")
    for move, count in sorted(stats.move_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {move}: {count}")

    if args.output:
        collector.save_report(Path(args.output))
        print(f"\nSaved full report to {args.output}")

if __name__ == "__main__":
    main()
```

### 4. Write tests
Create `tests/data/test_validation.py`:
```python
"""Tests for data validation."""
import pytest
from poke.data.validation import BattleValidator
from poke.data.models import (
    ParsedBattle, TurnRecord, BattleState,
    PlayerState, PokemonState
)

@pytest.fixture
def valid_battle():
    return ParsedBattle(
        replay_id="test-123",
        format="gen9ou",
        players=("P1", "P2"),
        winner="P1",
        turns=[
            TurnRecord(
                turn=i,
                state_before=BattleState(
                    turn=i,
                    player1=PlayerState(team=[PokemonState(species="Pikachu", hp_fraction=1.0)]),
                    player2=PlayerState(team=[PokemonState(species="Charizard", hp_fraction=1.0)]),
                ),
            )
            for i in range(1, 11)
        ],
    )

def test_valid_battle_passes(valid_battle):
    validator = BattleValidator()
    result = validator.validate(valid_battle)
    assert result.valid
    assert len(result.errors) == 0

def test_short_battle_warns():
    battle = ParsedBattle(
        replay_id="test",
        format="gen9ou",
        players=("P1", "P2"),
        winner="P1",
        turns=[
            TurnRecord(
                turn=1,
                state_before=BattleState(
                    turn=1,
                    player1=PlayerState(team=[]),
                    player2=PlayerState(team=[]),
                ),
            ),
        ],
    )
    validator = BattleValidator(min_turns=3)
    result = validator.validate(battle)
    assert "too_few_turns" in result.warnings
```

## Acceptance Criteria
- [ ] Validates essential fields (replay_id, turns, winner)
- [ ] Checks for reasonable turn counts
- [ ] Detects turn number inconsistencies
- [ ] Computes species/move frequency distributions
- [ ] Generates human-readable reports
- [ ] Saves machine-readable JSON output

## Estimated Complexity
Low-Medium - Straightforward aggregation and validation logic
