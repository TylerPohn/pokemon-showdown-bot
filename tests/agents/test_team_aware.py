"""Tests for team-aware agent."""
import pytest
from pathlib import Path

from poke.agents.team_aware import TeamAwareAgent
from poke.teams.loader import TeamPool
from poke.teams.parser import TeamParser
from poke_env.battle import AbstractBattle

SAMPLE_TEAM = """
Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Hidden Power Ice
"""


class ConcreteTeamAwareAgent(TeamAwareAgent):
    """Concrete implementation for testing."""

    def choose_move(self, battle: AbstractBattle):
        """Simple move selection for testing."""
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_default_move()


@pytest.fixture
def mini_pool(tmp_path):
    # Create temp team files
    team_dir = tmp_path / "teams"
    team_dir.mkdir()

    for i in range(3):
        (team_dir / f"team_{i}.txt").write_text(SAMPLE_TEAM)

    return TeamPool.from_directory(team_dir)


def test_team_aware_agent_init(mini_pool):
    agent = ConcreteTeamAwareAgent(
        team_pool=mini_pool,
        battle_format="gen9ou",
        max_concurrent_battles=1,
    )

    assert agent.observation_size > 0
