"""Full battle state encoder."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig, ScaledEncoderConfig
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


class ScaledPokemonEncoder(nn.Module):
    """Encode a single Pokemon's state with rich embeddings.

    Encodes:
    - Species (embedding)
    - Moves (embedding + attention pooling)
    - Item (embedding)
    - Ability (embedding)
    - Status (embedding)
    - Continuous stats (HP, boosts, etc.)

    Projects to d_model dimensions.
    """

    def __init__(self, config: ScaledEncoderConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.species_embed = nn.Embedding(
            config.num_species, config.species_embed_dim, padding_idx=0
        )
        self.move_embed = nn.Embedding(
            config.num_moves, config.move_embed_dim, padding_idx=0
        )
        self.item_embed = nn.Embedding(
            config.num_items, config.item_embed_dim, padding_idx=0
        )
        self.ability_embed = nn.Embedding(
            config.num_abilities, config.ability_embed_dim, padding_idx=0
        )
        self.status_embed = nn.Embedding(
            config.num_status, config.status_embed_dim, padding_idx=0
        )

        # Move pooling (attention over 4 moves)
        self.move_attention = nn.Sequential(
            nn.Linear(config.move_embed_dim, 1),
        )

        # Continuous feature projection
        # HP, boosts (6), fainted flag, known flag, etc.
        self.continuous_dim = 16
        self.continuous_proj = nn.Linear(self.continuous_dim, config.species_embed_dim)

        # Calculate combined dimension
        combined_dim = (
            config.species_embed_dim +  # Species
            config.move_embed_dim +     # Pooled moves
            config.item_embed_dim +     # Item
            config.ability_embed_dim +  # Ability
            config.status_embed_dim +   # Status
            config.species_embed_dim    # Continuous features projected
        )

        # Project to d_model
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(
        self,
        species_id: torch.Tensor,       # [batch]
        move_ids: torch.Tensor,         # [batch, 4]
        item_id: torch.Tensor,          # [batch]
        ability_id: torch.Tensor,       # [batch]
        status_id: torch.Tensor,        # [batch]
        continuous: torch.Tensor,       # [batch, continuous_dim]
    ) -> torch.Tensor:
        """Encode a single Pokemon.

        Returns:
            [batch, d_model] encoded Pokemon representation
        """
        # Embed categorical features
        species_emb = self.species_embed(species_id)  # [batch, species_dim]

        # Embed and pool moves
        move_emb = self.move_embed(move_ids)  # [batch, 4, move_dim]
        move_attn = self.move_attention(move_emb)  # [batch, 4, 1]
        move_attn = F.softmax(move_attn, dim=1)
        move_pooled = (move_emb * move_attn).sum(dim=1)  # [batch, move_dim]

        item_emb = self.item_embed(item_id)        # [batch, item_dim]
        ability_emb = self.ability_embed(ability_id)  # [batch, ability_dim]
        status_emb = self.status_embed(status_id)  # [batch, status_dim]

        # Project continuous features
        cont_proj = self.continuous_proj(continuous)  # [batch, species_dim]

        # Combine all
        combined = torch.cat([
            species_emb,
            move_pooled,
            item_emb,
            ability_emb,
            status_emb,
            cont_proj,
        ], dim=-1)

        return self.output_proj(combined)


class ScaledStateEncoder(nn.Module):
    """Scaled state encoder for the 200M parameter model.

    Encodes a single turn observation to d_model dimensions.
    Produces a sequence of tokens suitable for transformer input:
    - [CLS] token for action prediction
    - 6 own Pokemon tokens
    - 6 opponent Pokemon tokens (revealed only)
    - Field condition token

    Total: 14 tokens per turn.
    """

    def __init__(self, config: ScaledEncoderConfig):
        super().__init__()
        self.config = config

        # Embeddings for field conditions
        self.team_embed = nn.Embedding(
            config.num_teams, config.team_embed_dim, padding_idx=0
        )
        self.weather_embed = nn.Embedding(
            config.num_weather, config.weather_embed_dim, padding_idx=0
        )
        self.terrain_embed = nn.Embedding(
            config.num_terrain, config.terrain_embed_dim, padding_idx=0
        )

        # Pokemon encoder (shared for all Pokemon)
        self.pokemon_encoder = ScaledPokemonEncoder(config)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Field condition projection
        field_dim = (
            config.team_embed_dim +
            config.weather_embed_dim +
            config.terrain_embed_dim +
            32  # Hazards and other field state
        )
        self.field_proj = nn.Sequential(
            nn.Linear(field_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        # Type embeddings to distinguish token types
        # 0: CLS, 1: own Pokemon, 2: opponent Pokemon, 3: field
        self.type_embed = nn.Embedding(4, config.d_model)

        # Output dimension
        self.output_dim = config.d_model
        self.num_tokens = 14  # CLS + 6 own + 6 opp + 1 field

    def forward(
        self,
        observation: dict,
        return_sequence: bool = True,
    ) -> torch.Tensor:
        """Encode a single turn observation.

        Args:
            observation: Dictionary containing:
                - team_id: [batch]
                - weather: [batch]
                - terrain: [batch]
                - pokemon_features: [batch, 12, feat_dim] (simplified)
                - hazards: [batch, 16]

                For full encoding (optional):
                - own_pokemon: Dict with species, moves, items, abilities, status, continuous
                - opp_pokemon: Same structure for opponent

            return_sequence: If True, return [batch, num_tokens, d_model]
                           If False, return [batch, d_model] (CLS only)

        Returns:
            Encoded state tensor
        """
        batch_size = observation["team_id"].shape[0]
        device = observation["team_id"].device

        # Encode field conditions
        team_emb = self.team_embed(observation["team_id"])
        weather_emb = self.weather_embed(observation["weather"])
        terrain_emb = self.terrain_embed(observation["terrain"])
        hazards = observation["hazards"]

        field_input = torch.cat([
            team_emb,
            weather_emb,
            terrain_emb,
            hazards[:, :32] if hazards.shape[-1] >= 32 else F.pad(hazards, (0, 32 - hazards.shape[-1])),
        ], dim=-1)
        field_token = self.field_proj(field_input)  # [batch, d_model]

        # For simplified observation format, create Pokemon tokens from flat features
        if "pokemon_features" in observation:
            # Simplified path: project flat features to d_model
            pokemon_features = observation["pokemon_features"]  # [batch, 12, feat_dim]

            # If features are too small, pad them
            feat_dim = pokemon_features.shape[-1]
            if feat_dim < self.config.d_model:
                # Simple linear projection from feature dim to d_model
                if not hasattr(self, "_simple_pokemon_proj"):
                    self._simple_pokemon_proj = nn.Linear(
                        feat_dim, self.config.d_model
                    ).to(device)
                pokemon_tokens = self._simple_pokemon_proj(pokemon_features)
            else:
                pokemon_tokens = pokemon_features[:, :, :self.config.d_model]

            own_pokemon = pokemon_tokens[:, :6, :]   # [batch, 6, d_model]
            opp_pokemon = pokemon_tokens[:, 6:, :]   # [batch, 6, d_model]
        else:
            # Full encoding path (when rich Pokemon data is available)
            own_pokemon = self._encode_pokemon_list(observation["own_pokemon"])
            opp_pokemon = self._encode_pokemon_list(observation["opp_pokemon"])

        # Build token sequence
        # [CLS] + own_pokemon (6) + opp_pokemon (6) + field (1) = 14 tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]

        tokens = torch.cat([
            cls_tokens,           # [batch, 1, d_model]
            own_pokemon,          # [batch, 6, d_model]
            opp_pokemon,          # [batch, 6, d_model]
            field_token.unsqueeze(1),  # [batch, 1, d_model]
        ], dim=1)  # [batch, 14, d_model]

        # Add type embeddings
        type_ids = torch.tensor(
            [0] + [1]*6 + [2]*6 + [3],
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)
        type_emb = self.type_embed(type_ids)
        tokens = tokens + type_emb

        if return_sequence:
            return tokens  # [batch, 14, d_model]
        else:
            return tokens[:, 0, :]  # [batch, d_model] - CLS token only

    def _encode_pokemon_list(self, pokemon_data: dict) -> torch.Tensor:
        """Encode a list of 6 Pokemon using the full encoder.

        Args:
            pokemon_data: Dictionary with batched Pokemon data

        Returns:
            [batch, 6, d_model] encoded Pokemon
        """
        batch_size = pokemon_data["species"].shape[0]
        encoded_list = []

        for i in range(6):
            encoded = self.pokemon_encoder(
                species_id=pokemon_data["species"][:, i],
                move_ids=pokemon_data["moves"][:, i, :],
                item_id=pokemon_data["item"][:, i],
                ability_id=pokemon_data["ability"][:, i],
                status_id=pokemon_data["status"][:, i],
                continuous=pokemon_data["continuous"][:, i, :],
            )
            encoded_list.append(encoded)

        return torch.stack(encoded_list, dim=1)

    @classmethod
    def observation_to_tensor(cls, obs: dict, device: str = "cpu") -> dict:
        """Convert numpy observation dict to tensor dict."""
        return {
            key: torch.tensor(val, device=device)
            for key, val in obs.items()
        }
