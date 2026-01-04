"""Integration tests for poke-env."""
import pytest
from poke_env.player import RandomPlayer

from poke.agents.base import BaseAgent
from poke.config import BattleConfig

class SimpleRandomAgent(BaseAgent):
    """Minimal agent that picks random legal moves."""

    def choose_move(self, battle):
        return self.choose_random_move(battle)

@pytest.mark.integration
def test_basic_battle(showdown_server):
    """Test that two agents can complete a battle."""
    config = BattleConfig()

    player1 = SimpleRandomAgent(
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )
    player2 = RandomPlayer(
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )

    # Run a single battle
    player1.battle_against(player2, n_battles=1)

    assert player1.n_finished_battles == 1
    assert player2.n_finished_battles == 1

@pytest.mark.integration
def test_multiple_battles(showdown_server):
    """Test running multiple sequential battles."""
    config = BattleConfig()

    player1 = SimpleRandomAgent(
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )
    player2 = SimpleRandomAgent(
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )

    player1.battle_against(player2, n_battles=5)

    assert player1.n_finished_battles == 5
