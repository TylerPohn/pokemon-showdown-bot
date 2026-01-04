COMPLETED

# PR-003: poke-env Integration

## Dependencies
- PR-001 (Project Setup)
- PR-002 (Local Showdown Server)

## Overview
Integrate the `poke-env` library as our primary interface for Pokemon battles. Verify connectivity and create base player classes.

## Tech Choices
- **Library:** poke-env >= 0.7
- **Battle Format:** gen9ou

## Tasks

### 1. Verify poke-env installation
```python
# Should work after PR-001
import poke_env
print(poke_env.__version__)
```

### 2. Create environment configuration
Create `src/poke/config.py`:
```python
"""Environment configuration."""
from dataclasses import dataclass

@dataclass
class BattleConfig:
    """Configuration for battle environment."""
    battle_format: str = "gen9ou"
    server_url: str = "localhost:8000"
    start_timer_on_search: bool = False
    max_concurrent_battles: int = 1

    @property
    def server_configuration(self):
        """Get poke-env server configuration."""
        from poke_env import LocalhostServerConfiguration
        return LocalhostServerConfiguration
```

### 3. Create base player class
Create `src/poke/agents/base.py`:
```python
"""Base agent classes."""
from abc import abstractmethod
from typing import List

from poke_env.player import Player
from poke_env.environment import Battle, AbstractBattle

class BaseAgent(Player):
    """Base class for all Pokemon battle agents."""

    def __init__(self, battle_format: str = "gen9ou", **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.battle_count = 0

    @abstractmethod
    def choose_move(self, battle: AbstractBattle):
        """Choose a move for the current turn.

        Args:
            battle: Current battle state

        Returns:
            BattleOrder for the chosen action
        """
        pass

    def get_legal_moves(self, battle: AbstractBattle) -> List[str]:
        """Get list of legal move orders."""
        orders = []

        # Available moves
        for move in battle.available_moves:
            orders.append(self.create_order(move))

        # Available switches
        for pokemon in battle.available_switches:
            orders.append(self.create_order(pokemon))

        return orders

    def teampreview(self, battle: AbstractBattle):
        """Handle team preview phase.

        Default: send Pokemon in slot order.
        """
        return "/team 123456"
```

### 4. Create integration test
Create `tests/test_poke_env.py`:
```python
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
```

### 5. Create battle state inspection utilities
Create `src/poke/utils/battle_utils.py`:
```python
"""Utilities for inspecting battle state."""
from poke_env.environment import AbstractBattle, Pokemon

def summarize_battle(battle: AbstractBattle) -> dict:
    """Create a summary of the current battle state."""
    return {
        "turn": battle.turn,
        "player": battle.player_username,
        "active": battle.active_pokemon.species if battle.active_pokemon else None,
        "team_hp": [p.current_hp_fraction for p in battle.team.values()],
        "opponent_pokemon_seen": len(battle.opponent_team),
        "weather": str(battle.weather) if battle.weather else None,
        "fields": [str(f) for f in battle.fields],
        "available_moves": len(battle.available_moves),
        "available_switches": len(battle.available_switches),
    }

def format_pokemon_status(pokemon: Pokemon) -> str:
    """Format a Pokemon's status for logging."""
    status = f"{pokemon.species} ({pokemon.current_hp_fraction*100:.0f}%)"
    if pokemon.status:
        status += f" [{pokemon.status.name}]"
    return status
```

### 6. Add pytest markers configuration
Update `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests as integration tests (require showdown server)",
]
```

## Acceptance Criteria
- [ ] `poke-env` imports and connects to local server
- [ ] `BaseAgent` class provides foundation for custom agents
- [ ] Two agents can complete a full battle
- [ ] Battle state inspection utilities work
- [ ] Integration tests pass with running Showdown server

## Estimated Complexity
Medium - External library integration with async behavior
