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
