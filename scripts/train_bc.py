#!/usr/bin/env python
"""Train behavior cloning policy."""
import argparse
import logging
from pathlib import Path

from poke.models.config import EncoderConfig
from poke.models.factory import create_policy
from poke.training.dataset import create_dataloader
from poke.training.bc_trainer import BCTrainer, BCConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Training data path")
    parser.add_argument("--val-data", help="Validation data path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/bc")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create model
    encoder_config = EncoderConfig()
    policy = create_policy("mlp", encoder_config=encoder_config)

    # Create data loaders
    train_loader = create_dataloader(
        Path(args.data),
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            Path(args.val_data),
            batch_size=args.batch_size,
            shuffle=False,
        )

    # Create trainer
    config = BCConfig(
        train_data_path=Path(args.data),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=Path(args.checkpoint_dir),
        use_wandb=args.wandb,
    )
    trainer = BCTrainer(policy, config)

    # Train
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
