COMPLETED

# PR-014: Action Masking System

## Dependencies
- PR-001 (Project Setup)
- PR-003 (poke-env Integration)

## Overview
Implement action masking to ensure the policy only outputs probabilities over legal actions. Illegal actions must have zero probability.

## Tech Choices
- **Masking Strategy:** Additive mask with -inf for illegal actions
- **Integration:** Applied before softmax in policy head

## Tasks

### 1. Define action space
Create `src/poke/models/action_space.py`:
```python
"""Action space definitions for Pokemon battles."""
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

class ActionType(IntEnum):
    """Types of actions in battle."""
    MOVE_1 = 0
    MOVE_2 = 1
    MOVE_3 = 2
    MOVE_4 = 3
    SWITCH_1 = 4
    SWITCH_2 = 5
    SWITCH_3 = 6
    SWITCH_4 = 7
    SWITCH_5 = 8
    SWITCH_6 = 9

@dataclass
class ActionSpace:
    """Action space for Pokemon battles.

    Actions 0-3: Use move in slot 1-4
    Actions 4-9: Switch to Pokemon in slot 1-6
    """
    num_moves: int = 4
    num_switches: int = 6

    @property
    def total_actions(self) -> int:
        return self.num_moves + self.num_switches

    def decode_action(self, action_idx: int) -> Tuple[str, int]:
        """Decode action index to type and target.

        Returns:
            (action_type, target_index) where action_type is 'move' or 'switch'
        """
        if action_idx < self.num_moves:
            return ("move", action_idx)
        else:
            return ("switch", action_idx - self.num_moves)

    def encode_action(self, action_type: str, target_idx: int) -> int:
        """Encode action to index."""
        if action_type == "move":
            return target_idx
        else:
            return self.num_moves + target_idx
```

### 2. Implement mask generator
Create `src/poke/models/masking.py`:
```python
"""Action masking for legal move enforcement."""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from poke_env.environment import AbstractBattle

from .action_space import ActionSpace

class ActionMask:
    """Generate action masks from battle state."""

    def __init__(self, action_space: ActionSpace = None):
        self.action_space = action_space or ActionSpace()

    def get_mask(self, battle: AbstractBattle) -> np.ndarray:
        """Generate action mask for current battle state.

        Args:
            battle: Current battle state

        Returns:
            Boolean array where True = legal, False = illegal
        """
        mask = np.zeros(self.action_space.total_actions, dtype=bool)

        # Mark legal moves
        available_move_ids = set()
        for i, move in enumerate(battle.available_moves):
            if i < self.action_space.num_moves:
                mask[i] = True
                available_move_ids.add(i)

        # Mark legal switches
        for i, pokemon in enumerate(battle.available_switches):
            switch_idx = self.action_space.num_moves + i
            if switch_idx < self.action_space.total_actions:
                mask[switch_idx] = True

        return mask

    def get_mask_tensor(
        self,
        battle: AbstractBattle,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Get mask as PyTorch tensor."""
        mask = self.get_mask(battle)
        return torch.tensor(mask, dtype=torch.bool, device=device)


class MaskedPolicy(nn.Module):
    """Policy network with action masking.

    Applies mask before softmax to zero out illegal actions.
    """

    def __init__(self, input_dim: int, action_dim: int = 10):
        super().__init__()
        self.action_dim = action_dim

        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute action probabilities with masking.

        Args:
            state: [batch, state_dim] encoded state
            action_mask: [batch, action_dim] boolean mask (True = legal)

        Returns:
            [batch, action_dim] action probabilities
        """
        logits = self.policy_head(state)

        if action_mask is not None:
            # Apply mask: set illegal actions to -inf
            logits = self.apply_mask(logits, action_mask)

        return torch.softmax(logits, dim=-1)

    def get_logits(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get raw logits (for training with cross-entropy)."""
        logits = self.policy_head(state)

        if action_mask is not None:
            logits = self.apply_mask(logits, action_mask)

        return logits

    @staticmethod
    def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply boolean mask to logits.

        Sets masked (False) positions to -inf so softmax gives 0.
        """
        # Convert bool mask to float mask
        # True -> 0, False -> -inf
        mask_values = torch.where(
            mask,
            torch.zeros_like(logits),
            torch.full_like(logits, float("-inf"))
        )
        return logits + mask_values


class ActionSelector:
    """Select actions from policy output with masking."""

    def __init__(self, action_space: ActionSpace = None):
        self.action_space = action_space or ActionSpace()
        self.mask_generator = ActionMask(self.action_space)

    def select_action(
        self,
        policy: MaskedPolicy,
        state: torch.Tensor,
        battle: AbstractBattle,
        deterministic: bool = False
    ) -> int:
        """Select an action from the policy.

        Args:
            policy: Policy network
            state: Encoded state tensor
            battle: Current battle for mask generation
            deterministic: If True, select argmax; else sample

        Returns:
            Selected action index
        """
        with torch.no_grad():
            mask = self.mask_generator.get_mask_tensor(
                battle,
                device=state.device
            ).unsqueeze(0)

            probs = policy(state.unsqueeze(0), mask)

            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                action = torch.multinomial(probs, 1).item()

        return action

    def action_to_order(self, action: int, battle: AbstractBattle):
        """Convert action index to battle order.

        Args:
            action: Action index
            battle: Current battle state

        Returns:
            BattleOrder for poke-env
        """
        action_type, target_idx = self.action_space.decode_action(action)

        if action_type == "move":
            if target_idx < len(battle.available_moves):
                return battle.available_moves[target_idx]
        else:
            if target_idx < len(battle.available_switches):
                return battle.available_switches[target_idx]

        # Fallback: return first legal action
        if battle.available_moves:
            return battle.available_moves[0]
        if battle.available_switches:
            return battle.available_switches[0]

        return None
```

### 3. Create mask validation utilities
Create `src/poke/models/mask_utils.py`:
```python
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
```

### 4. Write unit tests
Create `tests/models/test_masking.py`:
```python
"""Tests for action masking."""
import pytest
import torch
import numpy as np
from unittest.mock import Mock

from poke.models.action_space import ActionSpace
from poke.models.masking import ActionMask, MaskedPolicy, ActionSelector

@pytest.fixture
def action_space():
    return ActionSpace()

@pytest.fixture
def mock_battle():
    battle = Mock()

    # 3 available moves
    battle.available_moves = [Mock(), Mock(), Mock()]

    # 2 available switches
    battle.available_switches = [Mock(), Mock()]

    return battle

def test_mask_shape(action_space, mock_battle):
    mask_gen = ActionMask(action_space)
    mask = mask_gen.get_mask(mock_battle)

    assert mask.shape == (10,)
    assert mask.dtype == bool

def test_mask_legal_moves(action_space, mock_battle):
    mask_gen = ActionMask(action_space)
    mask = mask_gen.get_mask(mock_battle)

    # First 3 moves should be legal
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == True
    assert mask[3] == False  # No 4th move

    # First 2 switches should be legal (indices 4-5)
    assert mask[4] == True
    assert mask[5] == True
    assert mask[6] == False

def test_masked_policy_zeros_illegal():
    policy = MaskedPolicy(input_dim=64, action_dim=10)

    state = torch.randn(1, 64)
    mask = torch.tensor([[True, True, False, False, True, False, False, False, False, False]])

    probs = policy(state, mask)

    # Illegal actions should have ~0 probability
    assert probs[0, 2].item() < 1e-6
    assert probs[0, 3].item() < 1e-6

    # Legal actions should sum to ~1
    legal_sum = probs[0, mask[0]].sum().item()
    assert abs(legal_sum - 1.0) < 1e-6

def test_action_space_encoding():
    space = ActionSpace()

    # Move encoding
    assert space.encode_action("move", 0) == 0
    assert space.encode_action("move", 3) == 3

    # Switch encoding
    assert space.encode_action("switch", 0) == 4
    assert space.encode_action("switch", 5) == 9

def test_action_space_decoding():
    space = ActionSpace()

    assert space.decode_action(0) == ("move", 0)
    assert space.decode_action(3) == ("move", 3)
    assert space.decode_action(4) == ("switch", 0)
    assert space.decode_action(9) == ("switch", 5)
```

## Acceptance Criteria
- [ ] Mask correctly identifies legal moves and switches
- [ ] Softmax output is zero for masked actions
- [ ] Policy samples only from legal actions
- [ ] Handles edge cases (no moves, no switches)
- [ ] Action encoding/decoding is consistent

## Notes
- The mask must be generated fresh each turn
- Some game states may have very few legal actions
- Mega evolution / Z-moves would extend the action space (future work)

## Estimated Complexity
Medium - Careful integration with policy network
