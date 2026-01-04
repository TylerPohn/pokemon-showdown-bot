"""Value classification heads for RL training.

This module implements the HL-Gauss (Histogram Loss with Gaussian smoothing)
value head, which replaces scalar regression with classification over value bins.

Key benefits of HL-Gauss over regression:
- More stable training with noisy targets
- Better scaling to large models
- Improved sample efficiency
- Natural uncertainty estimation

Reference: "Stop Regressing: Training Value Functions via Classification"
https://arxiv.org/abs/2403.03950
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HLGaussValueHead(nn.Module):
    """Histogram Loss with Gaussian smoothing for value estimation.

    Instead of predicting a scalar value, this head predicts a categorical
    distribution over value bins. During training, target values are converted
    to soft categorical targets using Gaussian smoothing.

    The expected value is recovered as the weighted sum of bin centers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_bins: int = 101,
        value_min: float = -1.0,
        value_max: float = 1.0,
        sigma: float = 0.75,
    ):
        """Initialize the HL-Gauss value head.

        Args:
            input_dim: Input feature dimension (d_model from transformer)
            hidden_dim: Hidden layer dimension
            num_bins: Number of bins in the value distribution
            value_min: Minimum value (corresponds to loss/worst outcome)
            value_max: Maximum value (corresponds to win/best outcome)
            sigma: Standard deviation for Gaussian soft targets
        """
        super().__init__()
        self.num_bins = num_bins
        self.value_min = value_min
        self.value_max = value_max
        self.sigma = sigma

        # Compute bin centers
        bin_centers = torch.linspace(value_min, value_max, num_bins)
        self.register_buffer("bin_centers", bin_centers)

        # Bin width for reference
        self.bin_width = (value_max - value_min) / (num_bins - 1)

        # Network: input -> hidden -> logits over bins
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins),
        )

        # Initialize output layer to small values for stable start
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits over value bins.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Logits over value bins [batch, num_bins]
        """
        return self.net(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get expected value from input features.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Expected values [batch]
        """
        logits = self.forward(x)
        return self.logits_to_value(logits)

    def logits_to_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to expected value.

        Args:
            logits: Logits over bins [batch, num_bins]

        Returns:
            Expected values [batch]
        """
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    def logits_to_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probability distribution.

        Args:
            logits: Logits over bins [batch, num_bins]

        Returns:
            Probability distribution [batch, num_bins]
        """
        return F.softmax(logits, dim=-1)

    def value_to_soft_targets(self, values: torch.Tensor) -> torch.Tensor:
        """Convert scalar values to Gaussian soft targets.

        This creates a categorical distribution centered on the target value
        with probability mass spread according to a Gaussian kernel.

        Args:
            values: Scalar target values [batch]

        Returns:
            Soft target distributions [batch, num_bins]
        """
        # Compute distances from each bin center to target values
        # Shape: [batch, num_bins]
        distances = self.bin_centers.unsqueeze(0) - values.unsqueeze(-1)

        # Gaussian kernel
        soft_targets = torch.exp(-0.5 * (distances / self.sigma) ** 2)

        # Normalize to valid probability distribution
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)

        return soft_targets

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute HL-Gauss loss.

        This uses cross-entropy between the predicted distribution and
        Gaussian-smoothed soft targets.

        Args:
            logits: Predicted logits [batch, num_bins]
            target_values: Scalar target values [batch]

        Returns:
            Scalar loss value
        """
        # Create soft targets from scalar values
        soft_targets = self.value_to_soft_targets(target_values)

        # Cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()

        return loss

    def compute_loss_with_metrics(
        self,
        logits: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute loss and additional metrics for logging.

        Args:
            logits: Predicted logits [batch, num_bins]
            target_values: Scalar target values [batch]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Main loss
        loss = self.compute_loss(logits, target_values)

        # Additional metrics
        with torch.no_grad():
            predicted_values = self.logits_to_value(logits)
            value_mse = F.mse_loss(predicted_values, target_values)
            value_mae = F.l1_loss(predicted_values, target_values)

            # Entropy of predicted distribution (for monitoring uncertainty)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        metrics = {
            "value_loss": loss.item(),
            "value_mse": value_mse.item(),
            "value_mae": value_mae.item(),
            "value_entropy": entropy.item(),
        }

        return loss, metrics


class TwoHotValueHead(nn.Module):
    """Two-hot encoding value head (alternative to HL-Gauss).

    This uses two-hot encoding where probability mass is placed on the
    two nearest bins proportional to distance. This is simpler than
    HL-Gauss but can be effective.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_bins: int = 101,
        value_min: float = -1.0,
        value_max: float = 1.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.value_min = value_min
        self.value_max = value_max

        bin_centers = torch.linspace(value_min, value_max, num_bins)
        self.register_buffer("bin_centers", bin_centers)
        self.bin_width = (value_max - value_min) / (num_bins - 1)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    def value_to_two_hot(self, values: torch.Tensor) -> torch.Tensor:
        """Convert values to two-hot encoding.

        Args:
            values: Scalar values [batch]

        Returns:
            Two-hot targets [batch, num_bins]
        """
        # Clamp values to valid range
        values = values.clamp(self.value_min, self.value_max)

        # Find lower bin index
        normalized = (values - self.value_min) / self.bin_width
        lower_idx = normalized.floor().long()
        lower_idx = lower_idx.clamp(0, self.num_bins - 2)

        # Compute weights for two-hot encoding
        upper_weight = normalized - lower_idx.float()
        lower_weight = 1.0 - upper_weight

        # Create two-hot targets
        targets = torch.zeros(values.shape[0], self.num_bins, device=values.device)
        targets.scatter_(1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        targets.scatter_(1, (lower_idx + 1).unsqueeze(-1), upper_weight.unsqueeze(-1))

        return targets

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss with two-hot targets."""
        two_hot_targets = self.value_to_two_hot(target_values)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(two_hot_targets * log_probs).sum(dim=-1).mean()


class ScalarValueHead(nn.Module):
    """Standard scalar regression value head (baseline).

    This is the traditional approach for value estimation,
    included for comparison with classification approaches.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar value.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Values [batch]
        """
        return self.net(x).squeeze(-1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss."""
        return F.mse_loss(predictions, target_values)


def create_value_head(
    head_type: str = "hl_gauss",
    input_dim: int = 1024,
    **kwargs,
) -> nn.Module:
    """Factory function to create value heads.

    Args:
        head_type: Type of value head ("hl_gauss", "two_hot", "scalar")
        input_dim: Input feature dimension
        **kwargs: Additional arguments passed to the value head

    Returns:
        Value head module
    """
    if head_type == "hl_gauss":
        return HLGaussValueHead(input_dim=input_dim, **kwargs)
    elif head_type == "two_hot":
        return TwoHotValueHead(input_dim=input_dim, **kwargs)
    elif head_type == "scalar":
        return ScalarValueHead(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown value head type: {head_type}")
