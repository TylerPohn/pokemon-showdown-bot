COMPLETED

# PR-013: State Encoder / Feature Engineering

## Dependencies
- PR-001 (Project Setup)
- PR-006 (Trajectory Converter) - for data format
- PR-010 (TeamID Observation Integration) - for observation format

## Overview
Create a neural network-compatible state encoder that transforms battle observations into fixed-size feature vectors. This is the input layer for all policy networks.

## Tech Choices
- **Framework:** PyTorch
- **Encoding:** Learned embeddings for categorical features
- **Output:** Concatenated dense vector

## Tasks

### 1. Define encoding dimensions
Create `src/poke/models/config.py`:
```python
"""Model configuration."""
from dataclasses import dataclass

@dataclass
class EncoderConfig:
    """Configuration for state encoder."""
    # Vocabulary sizes
    num_species: int = 1500  # All Pokemon species
    num_moves: int = 1000    # All moves
    num_items: int = 500     # All items
    num_abilities: int = 300 # All abilities

    # Embedding dimensions
    species_embed_dim: int = 64
    move_embed_dim: int = 32
    item_embed_dim: int = 16
    ability_embed_dim: int = 16

    # Team encoding
    max_team_size: int = 6
    max_moves_per_pokemon: int = 4

    # Output
    hidden_dim: int = 256
    output_dim: int = 256

    # Misc
    num_weather: int = 10
    num_terrain: int = 10
    num_status: int = 10
    num_teams: int = 100
```

### 2. Implement species/move embeddings
Create `src/poke/models/embeddings.py`:
```python
"""Embedding layers for Pokemon data."""
import torch
import torch.nn as nn

from .config import EncoderConfig

class SpeciesEmbedding(nn.Module):
    """Learnable embedding for Pokemon species."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.num_species,
            embedding_dim=config.species_embed_dim,
            padding_idx=0,  # Unknown species
        )

    def forward(self, species_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            species_ids: [batch, num_pokemon] tensor of species IDs

        Returns:
            [batch, num_pokemon, embed_dim] embeddings
        """
        return self.embedding(species_ids)


class MoveEmbedding(nn.Module):
    """Learnable embedding for moves."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.num_moves,
            embedding_dim=config.move_embed_dim,
            padding_idx=0,
        )

    def forward(self, move_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            move_ids: [batch, num_pokemon, 4] tensor of move IDs

        Returns:
            [batch, num_pokemon, 4, embed_dim] embeddings
        """
        return self.embedding(move_ids)


class CategoricalEmbedding(nn.Module):
    """Generic categorical embedding."""

    def __init__(self, num_categories: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids)
```

### 3. Implement Pokemon encoder
Create `src/poke/models/pokemon_encoder.py`:
```python
"""Encoder for individual Pokemon state."""
import torch
import torch.nn as nn

from .config import EncoderConfig
from .embeddings import SpeciesEmbedding, MoveEmbedding, CategoricalEmbedding

class PokemonEncoder(nn.Module):
    """Encode a single Pokemon's state to a vector.

    Combines:
    - Species embedding
    - Move embeddings (pooled)
    - Item embedding
    - Ability embedding
    - Continuous features (HP, stats, boosts)
    - Status encoding
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Categorical embeddings
        self.species_embed = SpeciesEmbedding(config)
        self.move_embed = MoveEmbedding(config)
        self.item_embed = CategoricalEmbedding(config.num_items, config.item_embed_dim)
        self.ability_embed = CategoricalEmbedding(config.num_abilities, config.ability_embed_dim)
        self.status_embed = CategoricalEmbedding(config.num_status, 8)

        # Continuous feature projection
        # HP, boosts (6), etc.
        self.continuous_dim = 10
        self.continuous_proj = nn.Linear(self.continuous_dim, 32)

        # Combine all features
        combined_dim = (
            config.species_embed_dim +
            config.move_embed_dim +  # Pooled moves
            config.item_embed_dim +
            config.ability_embed_dim +
            8 +  # Status
            32   # Continuous
        )

        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim // 2),
        )

    def forward(
        self,
        species_id: torch.Tensor,
        move_ids: torch.Tensor,
        item_id: torch.Tensor,
        ability_id: torch.Tensor,
        status_id: torch.Tensor,
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            species_id: [batch] species IDs
            move_ids: [batch, 4] move IDs
            item_id: [batch] item IDs
            ability_id: [batch] ability IDs
            status_id: [batch] status IDs
            continuous: [batch, continuous_dim] continuous features

        Returns:
            [batch, output_dim//2] encoded Pokemon
        """
        # Embeddings
        species_emb = self.species_embed(species_id)  # [batch, embed_dim]
        move_emb = self.move_embed(move_ids)  # [batch, 4, embed_dim]
        move_emb = move_emb.mean(dim=1)  # Pool moves
        item_emb = self.item_embed(item_id)
        ability_emb = self.ability_embed(ability_id)
        status_emb = self.status_embed(status_id)

        # Continuous
        cont_proj = self.continuous_proj(continuous)

        # Concatenate
        combined = torch.cat([
            species_emb,
            move_emb,
            item_emb,
            ability_emb,
            status_emb,
            cont_proj,
        ], dim=-1)

        return self.output_proj(combined)
```

### 4. Implement full state encoder
Create `src/poke/models/state_encoder.py`:
```python
"""Full battle state encoder."""
import torch
import torch.nn as nn

from .config import EncoderConfig
from .embeddings import CategoricalEmbedding

class StateEncoder(nn.Module):
    """Encode complete battle state to a fixed-size vector.

    Encodes:
    - Active Pokemon (both sides)
    - Team states (both sides)
    - Field conditions
    - Team ID (for conditioning)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Team ID embedding
        self.team_embed = CategoricalEmbedding(config.num_teams, 32)

        # Field condition embeddings
        self.weather_embed = CategoricalEmbedding(config.num_weather, 16)
        self.terrain_embed = CategoricalEmbedding(config.num_terrain, 16)

        # Pokemon state (simplified for now)
        # In practice, would use PokemonEncoder
        self.pokemon_dim = 32  # Per pokemon
        self.num_pokemon = 12  # 6 per side

        # Hazard features
        self.hazard_dim = 8  # Per side

        # Input size calculation
        input_dim = (
            32 +  # Team ID
            16 + 16 +  # Weather + Terrain
            self.pokemon_dim * self.num_pokemon +  # All Pokemon
            self.hazard_dim * 2  # Both sides' hazards
        )

        # Main encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, observation: dict) -> torch.Tensor:
        """
        Args:
            observation: Dictionary with battle state tensors

        Returns:
            [batch, output_dim] encoded state
        """
        batch_size = observation["team_id"].shape[0]

        # Encode components
        team_emb = self.team_embed(observation["team_id"])
        weather_emb = self.weather_embed(observation["weather"])
        terrain_emb = self.terrain_embed(observation["terrain"])

        # Pokemon features (simplified - just use raw features)
        pokemon_features = observation["pokemon_features"]  # [batch, 12, feat_dim]
        pokemon_flat = pokemon_features.view(batch_size, -1)

        # Hazards
        hazards = observation["hazards"]  # [batch, 16]

        # Concatenate all
        combined = torch.cat([
            team_emb,
            weather_emb,
            terrain_emb,
            pokemon_flat,
            hazards,
        ], dim=-1)

        return self.encoder(combined)

    @classmethod
    def observation_to_tensor(cls, obs: dict, device: str = "cpu") -> dict:
        """Convert numpy observation dict to tensor dict."""
        return {
            key: torch.tensor(val, device=device)
            for key, val in obs.items()
        }
```

### 5. Create feature preprocessor
Create `src/poke/models/preprocessing.py`:
```python
"""Preprocessing utilities for model input."""
import torch
import numpy as np
from typing import Dict, Any

class FeaturePreprocessor:
    """Convert raw observations to model-ready tensors."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def preprocess(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess a single observation.

        Args:
            observation: Raw observation dict

        Returns:
            Dict of tensors ready for model input
        """
        return {
            "team_id": self._to_tensor(observation.get("team_id", 0), dtype=torch.long),
            "weather": self._to_tensor(observation.get("weather_id", 0), dtype=torch.long),
            "terrain": self._to_tensor(observation.get("terrain_id", 0), dtype=torch.long),
            "pokemon_features": self._encode_pokemon(observation),
            "hazards": self._encode_hazards(observation),
        }

    def preprocess_batch(self, observations: list) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of observations."""
        processed = [self.preprocess(obs) for obs in observations]
        return {
            key: torch.stack([p[key] for p in processed])
            for key in processed[0].keys()
        }

    def _to_tensor(self, value, dtype=torch.float32) -> torch.Tensor:
        """Convert value to tensor."""
        if isinstance(value, torch.Tensor):
            return value.to(dtype)
        return torch.tensor(value, dtype=dtype)

    def _encode_pokemon(self, obs: dict) -> torch.Tensor:
        """Encode all Pokemon features."""
        features = []

        # Own team (6 Pokemon)
        for i in range(6):
            hp = obs.get(f"team_hp", [1.0] * 6)[i] if i < len(obs.get("team_hp", [])) else 0.0
            fainted = obs.get(f"team_fainted", [False] * 6)[i] if i < len(obs.get("team_fainted", [])) else False

            pokemon_feat = [
                hp,
                float(fainted),
                # Add more features as needed
            ]
            # Pad to fixed size
            pokemon_feat.extend([0.0] * (32 - len(pokemon_feat)))
            features.extend(pokemon_feat[:32])

        # Opponent team (6 Pokemon)
        for i in range(6):
            opp_hp = obs.get(f"opp_team_revealed", [0] * 6)
            revealed = i < len(opp_hp) and opp_hp[i] > 0

            pokemon_feat = [
                obs.get("opp_active_hp", 1.0) if i == 0 else 1.0,
                float(revealed),
            ]
            pokemon_feat.extend([0.0] * (32 - len(pokemon_feat)))
            features.extend(pokemon_feat[:32])

        return torch.tensor(features, dtype=torch.float32)

    def _encode_hazards(self, obs: dict) -> torch.Tensor:
        """Encode hazard state."""
        return torch.tensor([
            float(obs.get("own_stealth_rock", False)),
            float(obs.get("own_spikes", 0)) / 3,
            float(obs.get("own_toxic_spikes", 0)) / 2,
            float(obs.get("own_sticky_web", False)),
            float(obs.get("opp_stealth_rock", False)),
            float(obs.get("opp_spikes", 0)) / 3,
            float(obs.get("opp_toxic_spikes", 0)) / 2,
            float(obs.get("opp_sticky_web", False)),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Padding
        ], dtype=torch.float32)
```

### 6. Write unit tests
Create `tests/models/test_encoder.py`:
```python
"""Tests for state encoder."""
import pytest
import torch

from poke.models.config import EncoderConfig
from poke.models.state_encoder import StateEncoder
from poke.models.preprocessing import FeaturePreprocessor

@pytest.fixture
def config():
    return EncoderConfig()

@pytest.fixture
def encoder(config):
    return StateEncoder(config)

def test_encoder_output_shape(encoder, config):
    batch_size = 4

    observation = {
        "team_id": torch.zeros(batch_size, dtype=torch.long),
        "weather": torch.zeros(batch_size, dtype=torch.long),
        "terrain": torch.zeros(batch_size, dtype=torch.long),
        "pokemon_features": torch.zeros(batch_size, 12 * 32),
        "hazards": torch.zeros(batch_size, 16),
    }

    output = encoder(observation)

    assert output.shape == (batch_size, config.output_dim)

def test_preprocessor():
    preprocessor = FeaturePreprocessor()

    raw_obs = {
        "team_id": 5,
        "weather_id": 1,
        "terrain_id": 0,
        "team_hp": [1.0, 0.8, 0.5, 0.0, 1.0, 1.0],
        "team_fainted": [False, False, False, True, False, False],
        "own_stealth_rock": True,
        "own_spikes": 2,
    }

    processed = preprocessor.preprocess(raw_obs)

    assert "team_id" in processed
    assert "pokemon_features" in processed
    assert processed["team_id"].item() == 5
```

## Acceptance Criteria
- [ ] Encoder produces fixed-size output regardless of input
- [ ] Handles missing/unknown values gracefully (padding)
- [ ] Species/move embeddings are learnable
- [ ] Batch processing works correctly
- [ ] Preprocessor handles raw observation dicts

## Notes
- Start with simple MLP encoder, can add attention later
- Embedding dimensions are hyperparameters to tune
- Unknown tokens should map to index 0 (padding)

## Estimated Complexity
Medium-High - Careful tensor shape management
