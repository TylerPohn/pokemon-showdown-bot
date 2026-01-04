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
