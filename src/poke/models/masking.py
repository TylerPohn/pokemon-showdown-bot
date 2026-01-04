"""Action masking for legal move enforcement."""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from poke_env.battle import AbstractBattle

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
