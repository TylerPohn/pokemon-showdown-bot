"""Critic networks for offline RL."""
import torch
import torch.nn as nn
from typing import Tuple

class QNetwork(nn.Module):
    """Q-network for action-value estimation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        input_dim = state_dim + action_dim  # State-action input

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
            ])

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for state-action pair.

        Args:
            state: [batch, state_dim] encoded states
            action: [batch, action_dim] one-hot actions

        Returns:
            [batch] Q-values
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Twin Q-networks for double Q-learning."""

    def __init__(self, state_dim: int, action_dim: int = 10, **kwargs):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, **kwargs)
        self.q2 = QNetwork(state_dim, action_dim, **kwargs)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values from both networks."""
        return self.q1(state, action), self.q2(state, action)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return minimum Q-value (for conservative estimation)."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """State value network for IQL."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = state_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
            ])

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute V-value for state."""
        return self.net(state).squeeze(-1)
