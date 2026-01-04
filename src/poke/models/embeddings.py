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
