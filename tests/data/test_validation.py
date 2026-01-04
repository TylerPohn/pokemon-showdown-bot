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
