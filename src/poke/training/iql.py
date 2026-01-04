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
