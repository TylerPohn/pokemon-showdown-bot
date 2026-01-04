"""Transformer-based policy for sequence modeling."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

class TransformerPolicy(nn.Module):
    """Transformer policy that conditions on turn history.

    Uses self-attention over past observations to inform decisions.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        action_dim: int = 10,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output heads
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs_sequence: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_sequence: [batch, seq_len, obs_dim] observation history
            action_mask: [batch, action_dim] mask for current turn

        Returns:
            (action_probs, value) for the last position
        """
        batch_size, seq_len, _ = obs_sequence.shape

        # Project and add positional encoding
        x = self.input_proj(obs_sequence)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)

        # Use last position for output
        last_hidden = x[:, -1, :]

        # Policy
        logits = self.policy_head(last_hidden)
        if action_mask is not None:
            logits = torch.where(
                action_mask,
                logits,
                torch.full_like(logits, float("-inf"))
            )
        action_probs = torch.softmax(logits, dim=-1)

        # Value
        value = self.value_head(last_hidden).squeeze(-1)

        return action_probs, value
