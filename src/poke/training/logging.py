"""Training metrics logging."""
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class MetricsLogger:
    """Log training metrics to CSV and optionally wandb."""

    def __init__(
        self,
        log_dir: Path,
        use_wandb: bool = False,
        wandb_project: str = "poke-training",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV logging
        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None

        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            wandb.init(project=wandb_project)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics for a step.

        Args:
            metrics: Dict of metric name -> value
            step: Global training step
        """
        # Add timestamp and step
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        # CSV logging
        self._log_csv(record)

        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def _log_csv(self, record: dict) -> None:
        """Append record to CSV file."""
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=list(record.keys())
            )
            self.csv_writer.writeheader()

        self.csv_writer.writerow(record)
        self.csv_file.flush()

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration."""
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.use_wandb:
            import wandb
            wandb.config.update(config)

    def close(self) -> None:
        """Close log files."""
        if self.csv_file:
            self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
