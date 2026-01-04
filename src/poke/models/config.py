"""Model configuration."""
from dataclasses import dataclass
from typing import Optional


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


@dataclass
class ScaledEncoderConfig:
    """Configuration for 200M parameter Metamon-style model.

    This config creates a decoder-only causal transformer with:
    - ~198M parameters total
    - HL-Gauss value classification
    - RoPE positional encoding
    - SwiGLU activation
    """
    # === Transformer Architecture ===
    d_model: int = 1024          # Hidden dimension
    n_layers: int = 14           # Number of transformer layers
    n_heads: int = 16            # Number of attention heads
    d_ff: int = 4096             # Feed-forward hidden dimension (4x d_model)
    dropout: float = 0.1         # Dropout rate

    # === Vocabulary Sizes ===
    num_species: int = 1500      # All Pokemon species
    num_moves: int = 1000        # All moves
    num_items: int = 500         # All items
    num_abilities: int = 300     # All abilities
    num_weather: int = 10        # Weather conditions
    num_terrain: int = 10        # Terrain types
    num_status: int = 10         # Status conditions
    num_teams: int = 100         # Team IDs for conditioning

    # === Scaled Embedding Dimensions ===
    species_embed_dim: int = 128   # Up from 64
    move_embed_dim: int = 64       # Up from 32
    item_embed_dim: int = 48       # Up from 16
    ability_embed_dim: int = 48    # Up from 16
    status_embed_dim: int = 32     # Status embedding
    weather_embed_dim: int = 32    # Weather embedding
    terrain_embed_dim: int = 32    # Terrain embedding
    team_embed_dim: int = 64       # Team ID embedding

    # === Sequence Modeling ===
    max_seq_len: int = 200         # Max battle history length (turns)
    max_team_size: int = 6         # Pokemon per team
    max_moves_per_pokemon: int = 4 # Moves per Pokemon

    # === Value Classification (HL-Gauss) ===
    num_value_bins: int = 101      # Bins for value distribution
    value_min: float = -1.0        # Min value (loss)
    value_max: float = 1.0         # Max value (win)
    hl_gauss_sigma: float = 0.75   # Gaussian smoothing sigma

    # === Action Space ===
    action_dim: int = 10           # 4 moves + 6 switches

    # === Training ===
    use_gradient_checkpointing: bool = True  # Save memory
    use_flash_attention: bool = True         # Fast attention (if available)

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        # Embeddings
        embed_params = (
            self.num_species * self.species_embed_dim +
            self.num_moves * self.move_embed_dim +
            self.num_items * self.item_embed_dim +
            self.num_abilities * self.ability_embed_dim +
            self.num_weather * self.weather_embed_dim +
            self.num_terrain * self.terrain_embed_dim +
            self.num_status * self.status_embed_dim +
            self.num_teams * self.team_embed_dim +
            self.max_seq_len * self.d_model  # Position embeddings
        )

        # Per transformer layer
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.d_model * self.d_model
        # FFN: SwiGLU uses 2/3 of d_ff for gate
        ffn_hidden = int(2 * self.d_ff / 3)
        ffn_params = self.d_model * ffn_hidden * 3  # w1, w2, w3
        # LayerNorm (2 per layer)
        norm_params = 4 * self.d_model
        layer_params = attn_params + ffn_params + norm_params

        # All transformer layers
        transformer_params = self.n_layers * layer_params

        # Output heads
        # Policy head
        policy_params = self.d_model * 512 + 512 * self.action_dim
        # Value head (HL-Gauss)
        value_params = self.d_model * 512 + 512 * self.num_value_bins

        return embed_params + transformer_params + policy_params + value_params


# Pre-configured model sizes
SMALL_CONFIG = ScaledEncoderConfig(
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=1024,
)  # ~15M params

MEDIUM_CONFIG = ScaledEncoderConfig(
    d_model=512,
    n_layers=8,
    n_heads=8,
    d_ff=2048,
)  # ~50M params

LARGE_CONFIG = ScaledEncoderConfig(
    d_model=1024,
    n_layers=14,
    n_heads=16,
    d_ff=4096,
)  # ~200M params
