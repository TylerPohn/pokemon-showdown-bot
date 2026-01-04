COMPLETED

# PR-015: Policy Network Architecture

## Dependencies
- PR-013 (State Encoder)
- PR-014 (Action Masking)

## Overview
Implement the main policy network architecture that maps observations to action probabilities. Start with MLP baseline with option to extend to transformer.

## Tech Choices
- **Baseline:** MLP with residual connections
- **Advanced (optional):** Transformer over turn history
- **Output:** Action logits (masked before softmax)

## Tasks

### 1. Implement MLP policy
Create `src/poke/models/policy.py`:
```python
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
```

### 2. Implement transformer policy (optional)
Create `src/poke/models/transformer_policy.py`:
```python
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
```

### 3. Create model factory
Create `src/poke/models/factory.py`:
```python
"""Model factory for creating policy networks."""
from typing import Optional

from .config import EncoderConfig
from .policy import MLPPolicy

def create_policy(
    model_type: str = "mlp",
    encoder_config: Optional[EncoderConfig] = None,
    **kwargs
) -> MLPPolicy:
    """Create a policy network.

    Args:
        model_type: Type of policy ('mlp' or 'transformer')
        encoder_config: Configuration for state encoder
        **kwargs: Additional model-specific arguments

    Returns:
        Policy network instance
    """
    if encoder_config is None:
        encoder_config = EncoderConfig()

    if model_type == "mlp":
        return MLPPolicy(
            encoder_config=encoder_config,
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 3),
            action_dim=kwargs.get("action_dim", 10),
            use_value_head=kwargs.get("use_value_head", True),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif model_type == "transformer":
        from .transformer_policy import TransformerPolicy
        return TransformerPolicy(
            obs_dim=encoder_config.output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 4. Add model utilities
Create `src/poke/models/utils.py`:
```python
"""Model utilities."""
import torch
from pathlib import Path
from typing import Optional

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    **kwargs
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        **kwargs
    }
    torch.save(checkpoint, path)

def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
```

### 5. Write unit tests
Create `tests/models/test_policy.py`:
```python
"""Tests for policy networks."""
import pytest
import torch

from poke.models.config import EncoderConfig
from poke.models.policy import MLPPolicy, ResidualBlock
from poke.models.factory import create_policy

@pytest.fixture
def config():
    return EncoderConfig()

@pytest.fixture
def sample_observation():
    batch_size = 4
    return {
        "team_id": torch.zeros(batch_size, dtype=torch.long),
        "weather": torch.zeros(batch_size, dtype=torch.long),
        "terrain": torch.zeros(batch_size, dtype=torch.long),
        "pokemon_features": torch.randn(batch_size, 12 * 32),
        "hazards": torch.randn(batch_size, 16),
    }

def test_mlp_policy_output_shape(config, sample_observation):
    policy = MLPPolicy(encoder_config=config)

    action_probs, value = policy(sample_observation)

    assert action_probs.shape == (4, 10)
    assert value.shape == (4,)

def test_action_probs_sum_to_one(config, sample_observation):
    policy = MLPPolicy(encoder_config=config)

    action_probs, _ = policy(sample_observation)

    sums = action_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)

def test_masked_probs_zero(config, sample_observation):
    policy = MLPPolicy(encoder_config=config)

    # Mask out actions 5-9
    mask = torch.tensor([[True] * 5 + [False] * 5] * 4)

    action_probs, _ = policy(sample_observation, action_mask=mask)

    # Masked actions should be ~0
    assert (action_probs[:, 5:] < 1e-6).all()

def test_factory_creates_mlp(config):
    policy = create_policy("mlp", encoder_config=config)

    assert isinstance(policy, MLPPolicy)

def test_residual_block_shape():
    block = ResidualBlock(dim=64, hidden_dim=64)

    x = torch.randn(4, 64)
    y = block(x)

    assert y.shape == x.shape
```

## Acceptance Criteria
- [ ] MLP policy produces valid probability distributions
- [ ] Action masking zeros out illegal actions
- [ ] Value head outputs scalar values
- [ ] Residual connections don't change dimensions
- [ ] Factory creates correct model types
- [ ] Checkpoint save/load works

## Notes
- Start with MLP, only use transformer if needed
- Keep hidden dimensions reasonable (256-512)
- Dropout helps with overfitting on limited data

## Estimated Complexity
Medium - Standard neural network implementation
