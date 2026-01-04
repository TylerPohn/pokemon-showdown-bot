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
