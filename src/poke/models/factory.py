"""Model factory for creating policy networks."""
from typing import Optional, Union

import torch.nn as nn

from .config import EncoderConfig, ScaledEncoderConfig, SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG
from .policy import MLPPolicy


def create_policy(
    model_type: str = "mlp",
    encoder_config: Optional[EncoderConfig] = None,
    **kwargs
) -> nn.Module:
    """Create a policy network.

    Args:
        model_type: Type of policy:
            - 'mlp': Small MLP policy (~0.86M params)
            - 'transformer': Medium transformer policy (~1.67M params)
            - 'decoder': Large decoder-only transformer (~200M params)
            - 'decoder-small': Small decoder (~15M params)
            - 'decoder-medium': Medium decoder (~50M params)
        encoder_config: Configuration for state encoder (used for mlp/transformer)
        **kwargs: Additional model-specific arguments

    Returns:
        Policy network instance
    """
    if model_type == "mlp":
        if encoder_config is None:
            encoder_config = EncoderConfig()
        return MLPPolicy(
            encoder_config=encoder_config,
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 3),
            action_dim=kwargs.get("action_dim", 10),
            use_value_head=kwargs.get("use_value_head", True),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_type == "transformer":
        if encoder_config is None:
            encoder_config = EncoderConfig()
        from .transformer_policy import TransformerPolicy
        return TransformerPolicy(
            obs_dim=encoder_config.output_dim,
            **kwargs
        )
    elif model_type in ("decoder", "decoder-large"):
        from .decoder_policy import DecoderPolicy
        config = kwargs.get("config", LARGE_CONFIG)
        return DecoderPolicy(config)
    elif model_type == "decoder-small":
        from .decoder_policy import DecoderPolicy
        config = kwargs.get("config", SMALL_CONFIG)
        return DecoderPolicy(config)
    elif model_type == "decoder-medium":
        from .decoder_policy import DecoderPolicy
        config = kwargs.get("config", MEDIUM_CONFIG)
        return DecoderPolicy(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> dict:
    """Get information about a model.

    Args:
        model: The model to inspect

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "model_class": model.__class__.__name__,
    }
