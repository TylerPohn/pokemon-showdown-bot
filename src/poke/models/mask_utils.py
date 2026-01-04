"""Utilities for action mask validation and debugging."""
import torch
import numpy as np
from typing import List

from .action_space import ActionSpace

def validate_mask(mask: np.ndarray, action_space: ActionSpace) -> List[str]:
    """Validate an action mask.

    Returns list of issues found (empty if valid).
    """
    issues = []

    # Check shape
    if mask.shape != (action_space.total_actions,):
        issues.append(f"Wrong shape: {mask.shape}")

    # Check at least one action is legal
    if not mask.any():
        issues.append("No legal actions")

    return issues

def mask_to_description(mask: np.ndarray, action_space: ActionSpace) -> str:
    """Convert mask to human-readable description."""
    parts = []

    for i, legal in enumerate(mask):
        if legal:
            action_type, target = action_space.decode_action(i)
            parts.append(f"{action_type}_{target}")

    return ", ".join(parts) if parts else "No legal actions"

def check_mask_consistency(
    probs: torch.Tensor,
    mask: torch.Tensor,
    tolerance: float = 1e-6
) -> bool:
    """Check that probabilities respect the mask.

    All masked positions should have probability ~0.
    """
    illegal_probs = probs[~mask]
    return (illegal_probs.abs() < tolerance).all().item()
