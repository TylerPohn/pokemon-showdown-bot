#!/usr/bin/env python
"""Train offline RL policy."""
import argparse
import logging
from pathlib import Path

from poke.models.config import EncoderConfig
from poke.models.factory import create_policy
from poke.training.rl_dataset import create_rl_dataloader
from poke.training.rl_trainer import OfflineRLConfig, create_trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--algorithm", default="iql", choices=["iql", "cql"])
    parser.add_argument("--bc-checkpoint", help="BC checkpoint for initialization")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-dir", default="checkpoints/rl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create model
    encoder_config = EncoderConfig()
    policy = create_policy("mlp", encoder_config=encoder_config)

    # Create data loader
    dataloader = create_rl_dataloader(
        Path(args.data),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create trainer
    config = OfflineRLConfig(
        algorithm=args.algorithm,
        train_data_path=Path(args.data),
        bc_checkpoint=Path(args.bc_checkpoint) if args.bc_checkpoint else None,
        num_steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_dir=Path(args.checkpoint_dir),
        device=args.device,
    )

    trainer = create_trainer(
        policy=policy,
        state_dim=encoder_config.output_dim,
        config=config,
    )

    # Train
    trainer.train(dataloader)

if __name__ == "__main__":
    main()
