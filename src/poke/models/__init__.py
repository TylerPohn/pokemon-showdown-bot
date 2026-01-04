"""Neural network models for Pokemon battle AI."""

from .config import EncoderConfig
from .embeddings import SpeciesEmbedding, MoveEmbedding, CategoricalEmbedding
from .pokemon_encoder import PokemonEncoder
from .state_encoder import StateEncoder
from .preprocessing import FeaturePreprocessor
from .action_space import ActionSpace, ActionType
from .masking import ActionMask, MaskedPolicy, ActionSelector
from .mask_utils import validate_mask, mask_to_description, check_mask_consistency
from .policy import MLPPolicy, ResidualBlock
from .transformer_policy import TransformerPolicy
from .factory import create_policy
from .utils import count_parameters, save_checkpoint, load_checkpoint

__all__ = [
    # Config
    "EncoderConfig",
    # Embeddings
    "SpeciesEmbedding",
    "MoveEmbedding",
    "CategoricalEmbedding",
    # Encoders
    "PokemonEncoder",
    "StateEncoder",
    "FeaturePreprocessor",
    # Action space
    "ActionSpace",
    "ActionType",
    # Masking
    "ActionMask",
    "MaskedPolicy",
    "ActionSelector",
    "validate_mask",
    "mask_to_description",
    "check_mask_consistency",
    # Policy networks
    "MLPPolicy",
    "ResidualBlock",
    "TransformerPolicy",
    "create_policy",
    # Utilities
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
]
