"""Training loop for offline RL."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .iql import IQL, IQLConfig
from .cql import CQL, CQLConfig
from .checkpointing import CheckpointManager, CheckpointMetadata
from .logging import MetricsLogger

logger = logging.getLogger(__name__)

@dataclass
class OfflineRLConfig:
    """Configuration for offline RL training."""
    # Algorithm
    algorithm: str = "iql"  # "iql" or "cql"

    # Data
    train_data_path: Path = Path("data/processed/trajectories.jsonl")

    # Training
    batch_size: int = 256
    num_steps: int = 100000
    eval_every: int = 5000
    log_every: int = 100

    # Initialization
    bc_checkpoint: Optional[Path] = None

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints/rl")
    save_every: int = 10000

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Algorithm-specific
    iql_config: Optional[IQLConfig] = None
    cql_config: Optional[CQLConfig] = None


class OfflineRLTrainer:
    """Trainer for offline RL algorithms."""

    def __init__(
        self,
        policy: torch.nn.Module,
        state_dim: int,
        config: OfflineRLConfig,
    ):
        self.config = config
        self.state_dim = state_dim

        # Initialize algorithm
        if config.algorithm == "iql":
            algo_config = config.iql_config or IQLConfig()
            self.algorithm = IQL(
                policy=policy,
                state_dim=state_dim,
                config=algo_config,
                device=config.device,
            )
        elif config.algorithm == "cql":
            algo_config = config.cql_config or CQLConfig()
            self.algorithm = CQL(
                policy=policy,
                state_dim=state_dim,
                config=algo_config,
                device=config.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")

        # Load BC initialization if provided
        if config.bc_checkpoint:
            self._load_bc_checkpoint(config.bc_checkpoint)

        # Checkpointing and logging
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.metrics_logger = MetricsLogger(config.checkpoint_dir / "logs")

        self.global_step = 0

    def _load_bc_checkpoint(self, path: Path) -> None:
        """Load policy weights from BC checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.algorithm.policy.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded BC checkpoint: {path}")

    def train(self, dataloader: DataLoader) -> None:
        """Run training loop."""
        logger.info(f"Starting {self.config.algorithm.upper()} training")
        logger.info(f"Total steps: {self.config.num_steps}")

        data_iter = iter(dataloader)
        pbar = tqdm(total=self.config.num_steps, desc="Training")

        running_metrics: Dict[str, float] = {}

        while self.global_step < self.config.num_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Update
            metrics = self.algorithm.update(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in running_metrics:
                    running_metrics[key] = 0.0
                running_metrics[key] += value

            self.global_step += 1
            pbar.update(1)

            # Logging
            if self.global_step % self.config.log_every == 0:
                avg_metrics = {
                    k: v / self.config.log_every
                    for k, v in running_metrics.items()
                }
                self.metrics_logger.log(avg_metrics, self.global_step)

                pbar.set_postfix(avg_metrics)
                running_metrics.clear()

            # Checkpointing
            if self.global_step % self.config.save_every == 0:
                self._save_checkpoint()

        pbar.close()
        self._save_checkpoint()
        logger.info("Training complete")

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        metadata = CheckpointMetadata(
            epoch=0,  # Offline RL doesn't have epochs
            step=self.global_step,
            timestamp="",
            train_loss=0.0,  # Could track running loss
            train_accuracy=0.0,
        )

        self.checkpoint_manager.save(
            model=self.algorithm.policy,
            optimizer=self.algorithm.policy_optimizer,
            metadata=metadata,
        )


def create_trainer(
    policy: torch.nn.Module,
    state_dim: int,
    config: OfflineRLConfig,
) -> OfflineRLTrainer:
    """Create offline RL trainer."""
    return OfflineRLTrainer(policy, state_dim, config)
