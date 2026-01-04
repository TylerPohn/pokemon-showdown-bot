COMPLETED

# PR-011: Random Agent

## Dependencies
- PR-003 (poke-env Integration)
- PR-010 (TeamID Observation Integration)

## Overview
Implement a random agent that selects uniformly from legal actions. This serves as the simplest baseline for comparison.

## Tech Choices
- **Base Class:** TeamAwareAgent (for team pool integration)
- **Selection:** Uniform random over legal moves and switches

## Tasks

### 1. Implement random agent
Create `src/poke/agents/random_agent.py`:
```python
"""Random baseline agent."""
import random
from typing import Optional

from poke_env.environment import AbstractBattle

from .team_aware import TeamAwareAgent
from ..teams.loader import TeamPool

class RandomAgent(TeamAwareAgent):
    """Agent that selects uniformly at random from legal actions."""

    def __init__(
        self,
        team_pool: TeamPool,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(team_pool=team_pool, **kwargs)
        self._rng = random.Random(seed)

    def choose_move(self, battle: AbstractBattle):
        """Choose a random legal move.

        Args:
            battle: Current battle state

        Returns:
            BattleOrder for a randomly selected action
        """
        # Collect all legal actions
        actions = []

        # Add available moves
        for move in battle.available_moves:
            actions.append(self.create_order(move))

        # Add available switches
        for pokemon in battle.available_switches:
            actions.append(self.create_order(pokemon))

        # If no actions available, struggle
        if not actions:
            return self.choose_default_move()

        return self._rng.choice(actions)

    def teampreview(self, battle: AbstractBattle):
        """Randomize team order for preview."""
        team_size = len(battle.team)
        order = list(range(1, team_size + 1))
        self._rng.shuffle(order)
        return "/team " + "".join(str(i) for i in order)


class PureRandomAgent(RandomAgent):
    """Random agent without team conditioning (for simpler baselines)."""

    def __init__(self, seed: Optional[int] = None, **kwargs):
        # Skip TeamAwareAgent init, go to Player directly
        from poke_env.player import Player
        Player.__init__(self, **kwargs)
        self._rng = random.Random(seed)

    def choose_move(self, battle: AbstractBattle):
        """Choose a random legal move."""
        actions = []

        for move in battle.available_moves:
            actions.append(self.create_order(move))

        for pokemon in battle.available_switches:
            actions.append(self.create_order(pokemon))

        if not actions:
            return self.choose_default_move()

        return self._rng.choice(actions)
```

### 2. Create test script
Create `scripts/test_random_agent.py`:
```python
#!/usr/bin/env python
"""Test random agent in battle."""
import argparse
import asyncio

from poke_env.player import RandomPlayer

from poke.agents.random_agent import RandomAgent, PureRandomAgent
from poke.teams.loader import get_default_pool
from poke.config import BattleConfig

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = BattleConfig()

    # Our random agent
    agent = PureRandomAgent(
        seed=args.seed,
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )

    # Opponent (poke-env's random player)
    opponent = RandomPlayer(
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )

    print(f"Running {args.battles} battles...")
    await agent.battle_against(opponent, n_battles=args.battles)

    wins = agent.n_won_battles
    print(f"\nResults:")
    print(f"  Wins: {wins}/{args.battles} ({100*wins/args.battles:.1f}%)")
    print(f"  Expected: ~50% (random vs random)")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Write unit tests
Create `tests/agents/test_random_agent.py`:
```python
"""Tests for random agent."""
import pytest
from unittest.mock import Mock, MagicMock

from poke.agents.random_agent import RandomAgent, PureRandomAgent

@pytest.fixture
def mock_battle():
    battle = Mock()

    # Create mock moves
    move1 = Mock()
    move1.id = "thunderbolt"
    move2 = Mock()
    move2.id = "voltswitch"

    battle.available_moves = [move1, move2]

    # Create mock switches
    switch1 = Mock()
    switch1.species = "Charizard"
    battle.available_switches = [switch1]

    battle.team = {f"mon{i}": Mock() for i in range(6)}

    return battle

def test_pure_random_chooses_legal_action(mock_battle):
    agent = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent.create_order = lambda x: x  # Simplified

    result = agent.choose_move(mock_battle)

    # Should be one of the available actions
    all_actions = mock_battle.available_moves + mock_battle.available_switches
    assert result in all_actions

def test_random_agent_deterministic_with_seed(mock_battle):
    agent1 = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent2 = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent1.create_order = lambda x: x
    agent2.create_order = lambda x: x

    results1 = [agent1.choose_move(mock_battle) for _ in range(10)]
    results2 = [agent2.choose_move(mock_battle) for _ in range(10)]

    assert results1 == results2

def test_random_agent_different_with_different_seeds(mock_battle):
    agent1 = PureRandomAgent(seed=42, battle_format="gen9ou")
    agent2 = PureRandomAgent(seed=123, battle_format="gen9ou")
    agent1.create_order = lambda x: x
    agent2.create_order = lambda x: x

    results1 = [agent1.choose_move(mock_battle) for _ in range(100)]
    results2 = [agent2.choose_move(mock_battle) for _ in range(100)]

    # Very unlikely to be identical
    assert results1 != results2

def test_teampreview_shuffles_order():
    agent = PureRandomAgent(seed=42, battle_format="gen9ou")

    battle = Mock()
    battle.team = {f"mon{i}": Mock() for i in range(6)}

    result = agent.teampreview(battle)

    assert result.startswith("/team ")
    order = result.replace("/team ", "")
    assert len(order) == 6
    assert set(order) == set("123456")
```

### 4. Add baseline agent registry
Create `src/poke/agents/__init__.py`:
```python
"""Agent implementations."""
from .base import BaseAgent
from .random_agent import RandomAgent, PureRandomAgent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "PureRandomAgent",
]

# Registry for easy access
BASELINE_AGENTS = {
    "random": PureRandomAgent,
    "random_team": RandomAgent,
}
```

## Acceptance Criteria
- [ ] Agent selects uniformly from all legal actions
- [ ] Agent can complete battles without errors
- [ ] Reproducible with seed
- [ ] ~50% winrate vs itself or other random agent
- [ ] Integrates with team pool when needed

## Notes
- This is the simplest possible agent
- Expected winrate vs random is 50%
- Useful for testing infrastructure

## Estimated Complexity
Low - Simple implementation on existing base classes
