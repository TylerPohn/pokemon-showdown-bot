COMPLETED

# PR-010: TeamID Observation Integration

## Dependencies
- PR-003 (poke-env Integration)
- PR-008 (Team File Format and Loader)

## Overview
Add TeamID to the agent's observation space. This allows the model to condition its policy on the team being used.

## Tech Choices
- **Encoding:** Integer ID (0 to N-1 for N teams)
- **Observation:** Concatenated with battle state features

## Tasks

### 1. Create observation builder
Create `src/poke/agents/observation.py`:
```python
"""Observation space construction for RL agents."""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from poke_env.environment import AbstractBattle, Pokemon

@dataclass
class ObservationConfig:
    """Configuration for observation space."""
    include_team_id: bool = True
    max_teams: int = 100  # Maximum number of teams for encoding
    include_opponent_pokemon: bool = True
    max_pokemon_per_side: int = 6
    include_field_conditions: bool = True

class ObservationBuilder:
    """Build observation vectors from battle state."""

    def __init__(self, config: ObservationConfig = None):
        self.config = config or ObservationConfig()
        self._team_id: Optional[int] = None

    def set_team_id(self, team_id: int) -> None:
        """Set the current team ID for observation."""
        self._team_id = team_id

    def build(self, battle: AbstractBattle) -> np.ndarray:
        """Build observation vector from battle state.

        Returns:
            1D numpy array with all features
        """
        features = []

        # Team ID (one-hot or integer)
        if self.config.include_team_id:
            team_features = self._encode_team_id()
            features.extend(team_features)

        # Active Pokemon features
        active_features = self._encode_active_pokemon(battle.active_pokemon)
        features.extend(active_features)

        # Team state
        team_features = self._encode_team(battle.team)
        features.extend(team_features)

        # Opponent state
        if self.config.include_opponent_pokemon:
            opp_active = self._encode_active_pokemon(
                battle.opponent_active_pokemon,
                is_opponent=True
            )
            features.extend(opp_active)

            opp_team = self._encode_opponent_team(battle)
            features.extend(opp_team)

        # Field conditions
        if self.config.include_field_conditions:
            field_features = self._encode_field(battle)
            features.extend(field_features)

        return np.array(features, dtype=np.float32)

    def _encode_team_id(self) -> List[float]:
        """Encode team ID as normalized float."""
        if self._team_id is None:
            return [0.0]
        # Normalize to [0, 1]
        return [self._team_id / max(1, self.config.max_teams - 1)]

    def _encode_active_pokemon(
        self,
        pokemon: Optional[Pokemon],
        is_opponent: bool = False
    ) -> List[float]:
        """Encode active Pokemon features."""
        if pokemon is None:
            return [0.0] * 10  # Placeholder size

        features = [
            pokemon.current_hp_fraction,
            1.0 if pokemon.status else 0.0,
            float(pokemon.boosts.get("atk", 0)) / 6,
            float(pokemon.boosts.get("def", 0)) / 6,
            float(pokemon.boosts.get("spa", 0)) / 6,
            float(pokemon.boosts.get("spd", 0)) / 6,
            float(pokemon.boosts.get("spe", 0)) / 6,
            1.0 if pokemon.must_recharge else 0.0,
            1.0 if pokemon.is_dynamaxed else 0.0,
            1.0 if pokemon.terastallized else 0.0,
        ]
        return features

    def _encode_team(self, team: dict[str, Pokemon]) -> List[float]:
        """Encode own team state."""
        features = []
        team_list = list(team.values())

        for i in range(self.config.max_pokemon_per_side):
            if i < len(team_list):
                mon = team_list[i]
                features.extend([
                    mon.current_hp_fraction,
                    1.0 if mon.fainted else 0.0,
                    1.0 if mon.status else 0.0,
                    1.0 if mon.active else 0.0,
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        return features

    def _encode_opponent_team(self, battle: AbstractBattle) -> List[float]:
        """Encode opponent team (imperfect information)."""
        features = []
        opp_team = list(battle.opponent_team.values())

        for i in range(self.config.max_pokemon_per_side):
            if i < len(opp_team):
                mon = opp_team[i]
                features.extend([
                    mon.current_hp_fraction if mon.current_hp_fraction else 1.0,
                    1.0 if mon.fainted else 0.0,
                    1.0,  # Revealed
                ])
            else:
                features.extend([1.0, 0.0, 0.0])  # Unknown

        return features

    def _encode_field(self, battle: AbstractBattle) -> List[float]:
        """Encode field conditions."""
        features = []

        # Weather (simplified)
        weather_active = 1.0 if battle.weather else 0.0
        features.append(weather_active)

        # Terrain
        terrain_active = 1.0 if battle.fields else 0.0
        features.append(terrain_active)

        # Own side hazards
        side = battle.side_conditions
        features.extend([
            1.0 if "stealthrock" in str(side).lower() else 0.0,
            1.0 if "spikes" in str(side).lower() else 0.0,
            1.0 if "toxicspikes" in str(side).lower() else 0.0,
            1.0 if "stickyweb" in str(side).lower() else 0.0,
        ])

        # Opponent side hazards
        opp_side = battle.opponent_side_conditions
        features.extend([
            1.0 if "stealthrock" in str(opp_side).lower() else 0.0,
            1.0 if "spikes" in str(opp_side).lower() else 0.0,
            1.0 if "toxicspikes" in str(opp_side).lower() else 0.0,
            1.0 if "stickyweb" in str(opp_side).lower() else 0.0,
        ])

        return features

    @property
    def observation_size(self) -> int:
        """Get the size of the observation vector."""
        # Calculate based on config
        size = 0
        if self.config.include_team_id:
            size += 1
        size += 10  # Active pokemon
        size += self.config.max_pokemon_per_side * 4  # Own team
        if self.config.include_opponent_pokemon:
            size += 10  # Opponent active
            size += self.config.max_pokemon_per_side * 3  # Opponent team
        if self.config.include_field_conditions:
            size += 10  # Field conditions
        return size
```

### 2. Create team-aware agent base
Create `src/poke/agents/team_aware.py`:
```python
"""Base class for team-aware agents."""
from typing import Optional

from poke_env.player import Player
from poke_env.environment import AbstractBattle

from .observation import ObservationBuilder, ObservationConfig
from ..teams.loader import TeamPool
from ..teams.integration import PoolTeambuilder

class TeamAwareAgent(Player):
    """Base agent that includes TeamID in observations."""

    def __init__(
        self,
        team_pool: TeamPool,
        obs_config: Optional[ObservationConfig] = None,
        **kwargs
    ):
        # Create teambuilder from pool
        self._teambuilder = PoolTeambuilder(team_pool)

        super().__init__(
            team=self._teambuilder,
            **kwargs
        )

        # Set up observation builder
        self._obs_builder = ObservationBuilder(obs_config or ObservationConfig())
        self._team_id_map = {tid: i for i, tid in enumerate(team_pool.get_ids())}

    def _battle_started_callback(self, battle: AbstractBattle) -> None:
        """Called when a battle starts."""
        # Update team ID in observation builder
        current_team_id = self._teambuilder.current_team_id
        if current_team_id:
            numeric_id = self._team_id_map.get(current_team_id, 0)
            self._obs_builder.set_team_id(numeric_id)

    def get_observation(self, battle: AbstractBattle):
        """Get observation vector for current battle state."""
        return self._obs_builder.build(battle)

    @property
    def observation_size(self) -> int:
        """Get observation vector size."""
        return self._obs_builder.observation_size
```

### 3. Add team ID to trajectory data
Update trajectory conversion to include team ID.

Create `src/poke/data/team_mapping.py`:
```python
"""Utilities for mapping teams to IDs in trajectory data."""
from pathlib import Path
from typing import Dict, Optional
import json

from ..teams.loader import TeamPool
from ..teams.models import Team

class TeamIDMapper:
    """Maps team content to numeric IDs for trajectory data."""

    def __init__(self, team_pool: Optional[TeamPool] = None):
        self._content_to_id: Dict[str, int] = {}
        self._id_to_content: Dict[int, str] = {}

        if team_pool:
            for i, team_id in enumerate(team_pool.get_ids()):
                team = team_pool[team_id]
                content_hash = Team.generate_id(team.to_showdown())
                self._content_to_id[content_hash] = i
                self._id_to_content[i] = content_hash

    def get_id(self, team_content: str) -> int:
        """Get numeric ID for team content.

        Returns 0 (unknown) if team not in pool.
        """
        content_hash = Team.generate_id(team_content)
        return self._content_to_id.get(content_hash, 0)

    def save(self, path: Path) -> None:
        """Save mapping to JSON."""
        path.write_text(json.dumps(self._content_to_id, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TeamIDMapper":
        """Load mapping from JSON."""
        mapper = cls()
        data = json.loads(path.read_text())
        mapper._content_to_id = data
        mapper._id_to_content = {v: k for k, v in data.items()}
        return mapper

    def __len__(self) -> int:
        return len(self._content_to_id)
```

### 4. Write tests
Create `tests/agents/test_observation.py`:
```python
"""Tests for observation building."""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from poke.agents.observation import ObservationBuilder, ObservationConfig

@pytest.fixture
def mock_battle():
    battle = Mock()
    battle.active_pokemon = Mock()
    battle.active_pokemon.current_hp_fraction = 0.75
    battle.active_pokemon.status = None
    battle.active_pokemon.boosts = {}
    battle.active_pokemon.must_recharge = False
    battle.active_pokemon.is_dynamaxed = False
    battle.active_pokemon.terastallized = False

    battle.team = {}
    battle.opponent_team = {}
    battle.opponent_active_pokemon = None
    battle.weather = None
    battle.fields = {}
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}

    return battle

def test_observation_shape(mock_battle):
    config = ObservationConfig()
    builder = ObservationBuilder(config)

    obs = builder.build(mock_battle)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert len(obs) == builder.observation_size

def test_team_id_in_observation(mock_battle):
    config = ObservationConfig(include_team_id=True)
    builder = ObservationBuilder(config)
    builder.set_team_id(5)

    obs = builder.build(mock_battle)

    # Team ID should be first feature (normalized)
    assert obs[0] == pytest.approx(5 / 99)  # max_teams - 1

def test_hp_encoding(mock_battle):
    config = ObservationConfig(include_team_id=False)
    builder = ObservationBuilder(config)

    mock_battle.active_pokemon.current_hp_fraction = 0.5
    obs = builder.build(mock_battle)

    # HP should be in the first position after team_id
    assert 0.5 in obs
```

Create `tests/agents/test_team_aware.py`:
```python
"""Tests for team-aware agent."""
import pytest
from pathlib import Path

from poke.agents.team_aware import TeamAwareAgent
from poke.teams.loader import TeamPool
from poke.teams.parser import TeamParser

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

@pytest.fixture
def mini_pool(tmp_path):
    # Create temp team files
    team_dir = tmp_path / "teams"
    team_dir.mkdir()

    for i in range(3):
        (team_dir / f"team_{i}.txt").write_text(SAMPLE_TEAM)

    return TeamPool.from_directory(team_dir)

def test_team_aware_agent_init(mini_pool):
    agent = TeamAwareAgent(
        team_pool=mini_pool,
        battle_format="gen9ou",
        max_concurrent_battles=1,
    )

    assert agent.observation_size > 0
```

## Acceptance Criteria
- [ ] ObservationBuilder produces fixed-size numpy arrays
- [ ] Team ID is included in observation when configured
- [ ] TeamAwareAgent integrates team selection with observation
- [ ] Observation size is deterministic and documented
- [ ] All features normalized to reasonable ranges

## Notes
- Keep observation compact to simplify neural network input
- Normalize all features to [-1, 1] or [0, 1] range
- Unknown opponent information should use sensible defaults

## Estimated Complexity
Medium - Careful feature engineering and poke-env integration
