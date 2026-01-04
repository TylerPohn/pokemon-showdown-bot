"""Policy network implementations."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import EncoderConfig
from .state_encoder import StateEncoder
from .masking import MaskedPolicy

class MLPPolicy(nn.Module):
    """MLP-based policy network.

    Architecture:
    - State encoder (observations -> features)
    - MLP backbone with residual connections
    - Policy head (features -> action logits)
    - Optional value head for actor-critic
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        hidden_dim: int = 256,
        num_layers: int = 3,
        action_dim: int = 10,
        use_value_head: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = StateEncoder(encoder_config)
        self.action_dim = action_dim
        self.use_value_head = use_value_head

        # MLP backbone with residual connections
        self.backbone = nn.ModuleList()
        input_dim = encoder_config.output_dim

        for i in range(num_layers):
            self.backbone.append(
                ResidualBlock(
                    dim=hidden_dim if i > 0 else input_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        # Project to hidden dim if needed
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Value head (optional)
        if use_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(
        self,
        observation: dict,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            observation: Dict of observation tensors
            action_mask: [batch, action_dim] boolean mask

        Returns:
            (action_probs, value) tuple
        """
        # Encode observation
        features = self.encoder(observation)

        # Project if needed
        features = self.input_proj(features)

        # Pass through backbone
        for block in self.backbone:
            features = block(features)

        # Policy output
        logits = self.policy_head(features)

        if action_mask is not None:
            logits = self._apply_mask(logits, action_mask)

        action_probs = torch.softmax(logits, dim=-1)

        # Value output
        value = None
        if self.use_value_head:
            value = self.value_head(features).squeeze(-1)

        return action_probs, value

    def get_action(
        self,
        observation: dict,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        """Sample or select an action.

        Returns:
            (action, log_prob, value) tuple
        """
        action_probs, value = self.forward(observation, action_mask)

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

        log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)) + 1e-8)

        return action.item(), log_prob.squeeze(-1), value

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply action mask to logits."""
        mask_values = torch.where(
            mask,
            torch.zeros_like(logits),
            torch.full_like(logits, float("-inf"))
        )
        return logits + mask_values


class ResidualBlock(nn.Module):
    """Residual MLP block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

        # Handle dimension mismatch
        self.proj = nn.Linear(dim, dim) if dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))
