COMPLETED

# PR-006: Trajectory Converter

## Dependencies
- PR-001 (Project Setup)
- PR-005 (Replay Parser) - for parsed battle data

## Overview
Convert parsed battle data into training trajectories suitable for reinforcement learning. Each trajectory is a sequence of (state, action, reward) tuples from one player's perspective.

## Tech Choices
- **Output Format:** HuggingFace Datasets compatible
- **Reward Scheme:** Sparse terminal (+1 win, -1 loss) with optional shaping

## Tasks

### 1. Define trajectory schema
Create `src/poke/data/trajectory.py`:
```python
"""Trajectory data structures for RL training."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class Observation:
    """Observation at a single timestep.

    All fields are designed to be easily converted to tensors.
    """
    turn: int

    # Active Pokemon (own)
    active_species_id: int
    active_hp: float  # 0-1
    active_status: int  # Encoded status

    # Own team state (6 Pokemon max)
    team_hp: List[float]  # 6 floats, 0-1
    team_status: List[int]  # 6 ints
    team_fainted: List[bool]  # 6 bools

    # Known opponent state
    opp_active_species_id: int
    opp_active_hp: float
    opp_team_revealed: List[int]  # Species IDs, 0 = unknown

    # Field conditions
    weather_id: int
    terrain_id: int

    # Hazards (own side)
    own_stealth_rock: bool
    own_spikes: int
    own_toxic_spikes: int
    own_sticky_web: bool

    # Hazards (opponent side)
    opp_stealth_rock: bool
    opp_spikes: int
    opp_toxic_spikes: int
    opp_sticky_web: bool

    # Team ID (for conditioning)
    team_id: int = 0

@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    observation: Observation
    action_type: int  # 0 = move, 1 = switch
    action_target: int  # Move index (0-3) or switch target (0-5)
    reward: float
    done: bool

@dataclass
class Trajectory:
    """Complete trajectory from one player's perspective."""
    replay_id: str
    player: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0

    def __len__(self) -> int:
        return len(self.steps)
```

### 2. Create species/move encoders
Create `src/poke/data/encoders.py`:
```python
"""Encoders for converting Pokemon data to numeric IDs."""
import json
from pathlib import Path
from typing import Dict, Optional

class SpeciesEncoder:
    """Encode Pokemon species names to numeric IDs."""

    def __init__(self, species_list: Optional[list[str]] = None):
        self._species_to_id: Dict[str, int] = {"<UNK>": 0}
        self._id_to_species: Dict[int, str] = {0: "<UNK>"}

        if species_list:
            for species in species_list:
                self.add(species)

    def add(self, species: str) -> int:
        """Add a species and return its ID."""
        normalized = self._normalize(species)
        if normalized not in self._species_to_id:
            new_id = len(self._species_to_id)
            self._species_to_id[normalized] = new_id
            self._id_to_species[new_id] = normalized
        return self._species_to_id[normalized]

    def encode(self, species: str) -> int:
        """Encode a species name to ID."""
        normalized = self._normalize(species)
        return self._species_to_id.get(normalized, 0)

    def decode(self, id: int) -> str:
        """Decode an ID to species name."""
        return self._id_to_species.get(id, "<UNK>")

    def _normalize(self, species: str) -> str:
        """Normalize species name."""
        return species.lower().replace(" ", "").replace("-", "")

    def save(self, path: Path) -> None:
        """Save encoder to JSON."""
        path.write_text(json.dumps(self._species_to_id))

    @classmethod
    def load(cls, path: Path) -> "SpeciesEncoder":
        """Load encoder from JSON."""
        encoder = cls()
        data = json.loads(path.read_text())
        encoder._species_to_id = data
        encoder._id_to_species = {v: k for k, v in data.items()}
        return encoder

    def __len__(self) -> int:
        return len(self._species_to_id)


class MoveEncoder:
    """Encode move names to numeric IDs."""

    def __init__(self):
        self._move_to_id: Dict[str, int] = {"<UNK>": 0}
        self._id_to_move: Dict[int, str] = {0: "<UNK>"}

    def add(self, move: str) -> int:
        normalized = move.lower().replace(" ", "")
        if normalized not in self._move_to_id:
            new_id = len(self._move_to_id)
            self._move_to_id[normalized] = new_id
            self._id_to_move[new_id] = normalized
        return self._move_to_id[normalized]

    def encode(self, move: str) -> int:
        normalized = move.lower().replace(" ", "")
        return self._move_to_id.get(normalized, 0)

    def decode(self, id: int) -> str:
        return self._id_to_move.get(id, "<UNK>")

    def __len__(self) -> int:
        return len(self._move_to_id)


class StatusEncoder:
    """Encode status conditions to numeric IDs."""
    STATUS_MAP = {
        "": 0,
        "brn": 1,
        "frz": 2,
        "par": 3,
        "psn": 4,
        "tox": 5,
        "slp": 6,
    }

    @classmethod
    def encode(cls, status: str) -> int:
        return cls.STATUS_MAP.get(status.lower(), 0)


class WeatherEncoder:
    """Encode weather conditions."""
    WEATHER_MAP = {
        "": 0,
        "none": 0,
        "sunnyday": 1,
        "raindance": 2,
        "sandstorm": 3,
        "hail": 4,
        "snow": 5,
    }

    @classmethod
    def encode(cls, weather: Optional[str]) -> int:
        if weather is None:
            return 0
        return cls.WEATHER_MAP.get(weather.lower(), 0)
```

### 3. Implement trajectory converter
Create `src/poke/data/converter.py`:
```python
"""Convert parsed battles to training trajectories."""
import logging
from typing import Iterator, Optional

from .models import ParsedBattle, TurnRecord, ActionType
from .trajectory import Trajectory, TrajectoryStep, Observation
from .encoders import SpeciesEncoder, StatusEncoder, WeatherEncoder

logger = logging.getLogger(__name__)

class TrajectoryConverter:
    """Convert parsed battles to RL trajectories."""

    def __init__(
        self,
        species_encoder: SpeciesEncoder,
        reward_shaping: bool = False,
    ):
        self.species_encoder = species_encoder
        self.reward_shaping = reward_shaping

    def convert(self, battle: ParsedBattle) -> Iterator[Trajectory]:
        """Convert a parsed battle to trajectories.

        Yields two trajectories: one for each player.
        """
        for player in ["p1", "p2"]:
            try:
                trajectory = self._convert_player(battle, player)
                if trajectory and len(trajectory) > 0:
                    yield trajectory
            except Exception as e:
                logger.warning(f"Failed to convert {battle.replay_id} for {player}: {e}")

    def _convert_player(self, battle: ParsedBattle, player: str) -> Optional[Trajectory]:
        """Convert battle from one player's perspective."""
        trajectory = Trajectory(
            replay_id=battle.replay_id,
            player=player,
        )

        # Determine if this player won
        is_winner = battle.winner == battle.players[0 if player == "p1" else 1]

        for i, turn in enumerate(battle.turns):
            # Get action for this player
            action = turn.p1_action if player == "p1" else turn.p2_action
            if action is None:
                continue

            # Build observation
            obs = self._build_observation(turn, player)

            # Determine reward
            is_last = (i == len(battle.turns) - 1)
            if is_last:
                reward = 1.0 if is_winner else -1.0
            elif self.reward_shaping:
                reward = self._compute_shaped_reward(turn, player)
            else:
                reward = 0.0

            # Convert action
            action_type = 0 if action.action_type == ActionType.MOVE else 1
            action_target = self._encode_action_target(action)

            step = TrajectoryStep(
                observation=obs,
                action_type=action_type,
                action_target=action_target,
                reward=reward,
                done=is_last,
            )

            trajectory.steps.append(step)
            trajectory.total_reward += reward

        return trajectory

    def _build_observation(self, turn: TurnRecord, player: str) -> Observation:
        """Build observation from turn state."""
        state = turn.state_before
        own_state = state.player1 if player == "p1" else state.player2
        opp_state = state.player2 if player == "p1" else state.player1

        # Find active Pokemon
        active = next((p for p in own_state.team if p.active), None)
        opp_active = next((p for p in opp_state.team if p.active), None)

        return Observation(
            turn=state.turn,
            active_species_id=self.species_encoder.encode(active.species) if active else 0,
            active_hp=active.hp_fraction if active else 0.0,
            active_status=StatusEncoder.encode(active.status.value) if active else 0,
            team_hp=[p.hp_fraction for p in own_state.team[:6]] + [0.0] * (6 - len(own_state.team)),
            team_status=[StatusEncoder.encode(p.status.value) for p in own_state.team[:6]] + [0] * (6 - len(own_state.team)),
            team_fainted=[p.fainted for p in own_state.team[:6]] + [False] * (6 - len(own_state.team)),
            opp_active_species_id=self.species_encoder.encode(opp_active.species) if opp_active else 0,
            opp_active_hp=opp_active.hp_fraction if opp_active else 1.0,
            opp_team_revealed=[self.species_encoder.encode(p.species) for p in opp_state.team[:6]] + [0] * (6 - len(opp_state.team)),
            weather_id=WeatherEncoder.encode(state.field.weather),
            terrain_id=0,  # TODO: terrain encoding
            own_stealth_rock=own_state.hazards.stealth_rock,
            own_spikes=own_state.hazards.spikes,
            own_toxic_spikes=own_state.hazards.toxic_spikes,
            own_sticky_web=own_state.hazards.sticky_web,
            opp_stealth_rock=opp_state.hazards.stealth_rock,
            opp_spikes=opp_state.hazards.spikes,
            opp_toxic_spikes=opp_state.hazards.toxic_spikes,
            opp_sticky_web=opp_state.hazards.sticky_web,
        )

    def _encode_action_target(self, action) -> int:
        """Encode action target to index."""
        # For now, return 0. In practice, need move/pokemon index mapping.
        return 0

    def _compute_shaped_reward(self, turn: TurnRecord, player: str) -> float:
        """Compute intermediate reward for shaping."""
        # TODO: Implement reward shaping (damage dealt, KOs, etc.)
        return 0.0
```

### 4. Create batch conversion script
Create `scripts/convert_trajectories.py`:
```python
#!/usr/bin/env python
"""Convert parsed battles to training trajectories."""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from poke.data.models import ParsedBattle
from poke.data.converter import TrajectoryConverter
from poke.data.encoders import SpeciesEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parsed battles JSONL")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable reward shaping")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: build species encoder
    print("Building species encoder...")
    species_encoder = SpeciesEncoder()
    with open(input_path) as f:
        for line in f:
            battle = ParsedBattle.model_validate_json(line)
            for turn in battle.turns:
                for p in turn.state_before.player1.team:
                    species_encoder.add(p.species)
                for p in turn.state_before.player2.team:
                    species_encoder.add(p.species)

    species_encoder.save(output_dir / "species_encoder.json")
    print(f"Found {len(species_encoder)} unique species")

    # Second pass: convert trajectories
    converter = TrajectoryConverter(
        species_encoder=species_encoder,
        reward_shaping=args.reward_shaping,
    )

    output_file = output_dir / "trajectories.jsonl"
    count = 0

    with open(input_path) as f_in, open(output_file, "w") as f_out:
        for line in tqdm(f_in, desc="Converting"):
            battle = ParsedBattle.model_validate_json(line)
            for trajectory in converter.convert(battle):
                # Serialize trajectory
                data = {
                    "replay_id": trajectory.replay_id,
                    "player": trajectory.player,
                    "total_reward": trajectory.total_reward,
                    "steps": [
                        {
                            "observation": step.observation.__dict__,
                            "action_type": step.action_type,
                            "action_target": step.action_target,
                            "reward": step.reward,
                            "done": step.done,
                        }
                        for step in trajectory.steps
                    ],
                }
                f_out.write(json.dumps(data) + "\n")
                count += 1

    print(f"Converted {count} trajectories to {output_file}")

if __name__ == "__main__":
    main()
```

### 5. Write unit tests
Create `tests/data/test_converter.py`:
```python
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
```

## Acceptance Criteria
- [ ] Converts parsed battles to player-perspective trajectories
- [ ] Produces two trajectories per battle (one per player)
- [ ] Correctly assigns win/loss rewards
- [ ] Encodes species to numeric IDs
- [ ] Saves encoder for inference time
- [ ] Output format is HuggingFace Datasets compatible

## Estimated Complexity
Medium - Data transformation with encoding logic
