"""Tests for trajectory converter."""
import pytest
from poke.data.converter import TrajectoryConverter
from poke.data.encoders import SpeciesEncoder
from poke.data.models import (
    ParsedBattle, TurnRecord, BattleState,
    PlayerState, PokemonState, Action, ActionType
)

@pytest.fixture
def sample_battle():
    return ParsedBattle(
        replay_id="test-123",
        format="gen9ou",
        players=("Player1", "Player2"),
        winner="Player1",
        turns=[
            TurnRecord(
                turn=1,
                state_before=BattleState(
                    turn=1,
                    player1=PlayerState(team=[
                        PokemonState(species="Pikachu", hp_fraction=1.0, active=True)
                    ]),
                    player2=PlayerState(team=[
                        PokemonState(species="Charizard", hp_fraction=1.0, active=True)
                    ]),
                ),
                p1_action=Action(player="p1", action_type=ActionType.MOVE, target="Thunderbolt"),
                p2_action=Action(player="p2", action_type=ActionType.MOVE, target="Flamethrower"),
            ),
        ],
    )

def test_convert_produces_two_trajectories(sample_battle):
    encoder = SpeciesEncoder(["Pikachu", "Charizard"])
    converter = TrajectoryConverter(encoder)

    trajectories = list(converter.convert(sample_battle))

    assert len(trajectories) == 2

def test_winner_gets_positive_reward(sample_battle):
    encoder = SpeciesEncoder(["Pikachu", "Charizard"])
    converter = TrajectoryConverter(encoder)

    trajectories = list(converter.convert(sample_battle))

    p1_traj = next(t for t in trajectories if t.player == "p1")
    p2_traj = next(t for t in trajectories if t.player == "p2")

    assert p1_traj.total_reward == 1.0
    assert p2_traj.total_reward == -1.0
