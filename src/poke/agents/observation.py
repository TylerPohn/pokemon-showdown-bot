"""Observation space construction for RL agents."""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from poke_env.battle import AbstractBattle, Pokemon

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
