"""Global configuration for the poke project."""

import os
from dataclasses import dataclass, field
from typing import Optional

from poke_env import LocalhostServerConfiguration, ShowdownServerConfiguration


@dataclass
class BattleConfig:
    """Configuration for battle environment."""

    battle_format: str = "gen9ou"
    server_url: str = "localhost:8000"
    start_timer_on_search: bool = False
    max_concurrent_battles: int = 1
    team_size: int = 6
    server_type: str = "local"  # "local" or "showdown"

    @property
    def server_configuration(self):
        """Get poke-env server configuration."""
        if self.server_type == "local":
            return LocalhostServerConfiguration
        return ShowdownServerConfiguration


@dataclass
class DataConfig:
    """Configuration for data processing."""

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    replay_batch_size: int = 100
    min_elo: int = 1500


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 10
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    """Global configuration container."""

    battle: BattleConfig = field(default_factory=BattleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Environment variables
    showdown_username: Optional[str] = field(
        default_factory=lambda: os.getenv("SHOWDOWN_USERNAME")
    )
    showdown_password: Optional[str] = field(
        default_factory=lambda: os.getenv("SHOWDOWN_PASSWORD")
    )


# Global config instance
config = Config()
