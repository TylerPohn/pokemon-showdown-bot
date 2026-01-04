COMPLETED

# PR-018: Offline RL Algorithms (CQL/IQL)

## Dependencies
- PR-015 (Policy Network)
- PR-016 (Behavior Cloning) - for initialization

## Overview
Implement Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) for offline RL fine-tuning. These algorithms are designed to learn from fixed datasets without environment interaction.

## Tech Choices
- **Primary Algorithm:** IQL (simpler, more stable)
- **Secondary Algorithm:** CQL (for comparison)
- **Critic Architecture:** Twin Q-networks

## Tasks

### 1. Implement Q-network
Create `src/poke/models/critic.py`:
```python
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
```

### 2. Implement IQL
Create `src/poke/training/iql.py`:
```python
"""Implicit Q-Learning implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple

from ..models.critic import TwinQNetwork, ValueNetwork

@dataclass
class IQLConfig:
    """Configuration for IQL training."""
    gamma: float = 0.99
    tau: float = 0.005  # Target network update rate
    expectile: float = 0.7  # Expectile for value learning
    temperature: float = 3.0  # Policy extraction temperature
    learning_rate: float = 3e-4
    batch_size: int = 256

class IQL:
    """Implicit Q-Learning algorithm.

    Key idea: Learn value function V(s) as an expectile of Q(s,a),
    then extract policy by advantage-weighted regression.
    """

    def __init__(
        self,
        policy: nn.Module,
        state_dim: int,
        action_dim: int = 10,
        config: IQLConfig = None,
        device: str = "cuda",
    ):
        self.config = config or IQLConfig()
        self.device = device

        # Networks
        self.policy = policy.to(device)
        self.q_network = TwinQNetwork(state_dim, action_dim).to(device)
        self.target_q = TwinQNetwork(state_dim, action_dim).to(device)
        self.value_network = ValueNetwork(state_dim).to(device)

        # Copy initial weights to target
        self.target_q.load_state_dict(self.q_network.state_dict())

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )
        self.v_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=self.config.learning_rate
        )

    def compute_value_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expectile loss for value function.

        V(s) is trained to be the expectile of Q(s,a).
        """
        with torch.no_grad():
            q_values = self.target_q.min_q(states, actions)

        v_values = self.value_network(states)
        diff = q_values - v_values

        # Expectile loss: weight positive errors more
        weight = torch.where(
            diff > 0,
            self.config.expectile,
            1 - self.config.expectile
        )
        loss = (weight * diff.pow(2)).mean()

        return loss

    def compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TD loss for Q-functions."""
        with torch.no_grad():
            # Use value network for bootstrapping (no max over actions)
            next_v = self.value_network(next_states)
            targets = rewards + self.config.gamma * (1 - dones) * next_v

        q1, q2 = self.q_network(states, actions)
        loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)

        return loss

    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantage-weighted regression loss for policy."""
        with torch.no_grad():
            q_values = self.target_q.min_q(states, actions)
            v_values = self.value_network(states)
            advantages = q_values - v_values

            # Exponential advantage weighting
            weights = torch.exp(advantages / self.config.temperature)
            weights = torch.clamp(weights, max=100.0)  # Clip for stability

        # Get policy log probabilities
        action_probs, _ = self.policy({"encoded_state": states}, action_masks)
        action_indices = actions.argmax(dim=-1)
        log_probs = torch.log(action_probs.gather(-1, action_indices.unsqueeze(-1)) + 1e-8)

        # Weighted negative log likelihood
        loss = -(weights * log_probs.squeeze(-1)).mean()

        return loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one update step.

        Args:
            batch: Dictionary with states, actions, rewards, next_states, dones, masks

        Returns:
            Dictionary of loss values
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)  # One-hot
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        masks = batch.get("masks")
        if masks is not None:
            masks = masks.to(self.device)

        # Update value network
        v_loss = self.compute_value_loss(states, actions)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q-networks
        q_loss = self.compute_q_loss(states, actions, rewards, next_states, dones)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        policy_loss = self.compute_policy_loss(states, actions, masks)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target network
        self._update_target()

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def _update_target(self) -> None:
        """Soft update target network."""
        for param, target_param in zip(
            self.q_network.parameters(),
            self.target_q.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1 - self.config.tau) * target_param.data
            )
```

### 3. Implement CQL
Create `src/poke/training/cql.py`:
```python
"""Conservative Q-Learning implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

from ..models.critic import TwinQNetwork

@dataclass
class CQLConfig:
    """Configuration for CQL training."""
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 5.0  # CQL regularization weight
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_random_actions: int = 10  # For CQL penalty estimation

class CQL:
    """Conservative Q-Learning algorithm.

    Key idea: Add penalty for Q-values of out-of-distribution actions.
    """

    def __init__(
        self,
        policy: nn.Module,
        state_dim: int,
        action_dim: int = 10,
        config: CQLConfig = None,
        device: str = "cuda",
    ):
        self.config = config or CQLConfig()
        self.device = device
        self.action_dim = action_dim

        self.policy = policy.to(device)
        self.q_network = TwinQNetwork(state_dim, action_dim).to(device)
        self.target_q = TwinQNetwork(state_dim, action_dim).to(device)
        self.target_q.load_state_dict(self.q_network.state_dict())

        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.config.learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )

    def compute_cql_penalty(self, states: torch.Tensor) -> torch.Tensor:
        """Compute CQL penalty on Q-values for random actions."""
        batch_size = states.shape[0]

        # Sample random actions
        random_actions = torch.zeros(
            batch_size, self.config.num_random_actions, self.action_dim,
            device=self.device
        )
        for i in range(self.config.num_random_actions):
            idx = torch.randint(0, self.action_dim, (batch_size,), device=self.device)
            random_actions[:, i, :] = F.one_hot(idx, self.action_dim).float()

        # Compute Q-values for random actions
        states_expanded = states.unsqueeze(1).expand(-1, self.config.num_random_actions, -1)
        states_flat = states_expanded.reshape(-1, states.shape[-1])
        actions_flat = random_actions.reshape(-1, self.action_dim)

        q1, q2 = self.q_network(states_flat, actions_flat)
        q1 = q1.reshape(batch_size, -1)
        q2 = q2.reshape(batch_size, -1)

        # Log-sum-exp penalty
        penalty = torch.logsumexp(q1, dim=1).mean() + torch.logsumexp(q2, dim=1).mean()

        return penalty

    def compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CQL loss = TD loss + conservative penalty."""
        with torch.no_grad():
            # Get next actions from policy
            next_action_probs, _ = self.policy({"encoded_state": next_states})
            next_actions = F.one_hot(
                next_action_probs.argmax(dim=-1),
                self.action_dim
            ).float()

            # Target Q-values
            next_q = self.target_q.min_q(next_states, next_actions)
            targets = rewards + self.config.gamma * (1 - dones) * next_q

        # TD loss
        q1, q2 = self.q_network(states, actions)
        td_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)

        # CQL penalty
        cql_penalty = self.compute_cql_penalty(states)

        # Subtract Q-values of dataset actions (to avoid penalizing them)
        q1_data, q2_data = self.q_network(states, actions)
        cql_penalty = cql_penalty - q1_data.mean() - q2_data.mean()

        # Total loss
        loss = td_loss + self.config.alpha * cql_penalty

        return loss, td_loss, cql_penalty

    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Policy improvement via Q-maximization."""
        action_probs, _ = self.policy({"encoded_state": states})

        # Compute expected Q-value over action distribution
        expected_q = 0
        for a in range(self.action_dim):
            action_one_hot = F.one_hot(
                torch.full((states.shape[0],), a, device=self.device),
                self.action_dim
            ).float()
            q = self.q_network.min_q(states, action_one_hot)
            expected_q = expected_q + action_probs[:, a] * q

        # Maximize expected Q
        loss = -expected_q.mean()

        return loss

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one update step."""
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Update Q-networks
        q_loss, td_loss, cql_penalty = self.compute_q_loss(
            states, actions, rewards, next_states, dones
        )
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy
        policy_loss = self.compute_policy_loss(states, actions)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target
        for param, target_param in zip(
            self.q_network.parameters(),
            self.target_q.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1 - self.config.tau) * target_param.data
            )

        return {
            "q_loss": q_loss.item(),
            "td_loss": td_loss.item(),
            "cql_penalty": cql_penalty.item(),
            "policy_loss": policy_loss.item(),
        }
```

### 4. Write unit tests
Create `tests/training/test_offline_rl.py`:
```python
"""Tests for offline RL algorithms."""
import pytest
import torch

from poke.models.critic import QNetwork, TwinQNetwork, ValueNetwork
from poke.training.iql import IQL, IQLConfig

@pytest.fixture
def state_dim():
    return 64

@pytest.fixture
def action_dim():
    return 10

def test_q_network_output_shape(state_dim, action_dim):
    q_net = QNetwork(state_dim, action_dim)

    state = torch.randn(4, state_dim)
    action = torch.zeros(4, action_dim)
    action[:, 0] = 1  # One-hot

    q_value = q_net(state, action)

    assert q_value.shape == (4,)

def test_twin_q_min(state_dim, action_dim):
    twin_q = TwinQNetwork(state_dim, action_dim)

    state = torch.randn(4, state_dim)
    action = torch.zeros(4, action_dim)
    action[:, 0] = 1

    min_q = twin_q.min_q(state, action)

    assert min_q.shape == (4,)

def test_value_network(state_dim):
    v_net = ValueNetwork(state_dim)

    state = torch.randn(4, state_dim)
    value = v_net(state)

    assert value.shape == (4,)
```

## Acceptance Criteria
- [ ] IQL value function learns expectile correctly
- [ ] CQL penalty reduces OOD Q-values
- [ ] Policy improves over BC initialization
- [ ] Training is stable (no divergence)
- [ ] Target networks update correctly

## Notes
- IQL is generally more stable and easier to tune
- CQL requires careful alpha tuning
- Initialize from BC checkpoint for best results
- Monitor Q-value magnitudes for stability

## Estimated Complexity
High - Careful implementation of RL algorithms
