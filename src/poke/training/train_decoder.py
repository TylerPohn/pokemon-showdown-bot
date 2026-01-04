"""Training script for 200M decoder model.

This script trains the Metamon-style decoder transformer on battle replay data.
Supports three model sizes:
- decoder-small (~15M params): For validation/debugging
- decoder-medium (~50M params): For faster iteration
- decoder (~200M params): Full model for production

Usage:
    # Validate with small model
    python -m poke.training.train_decoder --model-size decoder-small --epochs 1

    # Full training
    python -m poke.training.train_decoder --model-size decoder --epochs 10

    # With wandb tracking
    python -m poke.training.train_decoder --use-wandb --wandb-project poke-decoder
"""
import argparse
import logging
from pathlib import Path

import torch

from poke.models.factory import create_policy, get_model_info
from poke.models.value_head import HLGaussValueHead
from poke.models.config import SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG
from poke.training.bc_trainer import BCConfig, BCTrainer
from poke.training.sequence_dataset import create_sequence_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train decoder model on battle replays")

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        default="decoder",
        choices=["decoder-small", "decoder-medium", "decoder"],
        help="Model size to train (default: decoder)",
    )

    # Data
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/trajectories.jsonl"),
        help="Path to trajectory JSONL file",
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        default=None,
        help="Path to validation data (optional)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="Maximum sequence length (turns of history)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples (for debugging)",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective_batch = batch_size * this)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Learning rate warmup steps",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )

    # Mixed precision
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="AMP dtype (bfloat16 recommended for A10/A100)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/decoder"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )

    # Logging
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="poke-decoder",
        help="Wandb project name",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log metrics every N steps",
    )

    # Hardware
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )

    return parser.parse_args()


def get_config_for_size(model_size: str):
    """Get the config for a model size."""
    configs = {
        "decoder-small": SMALL_CONFIG,
        "decoder-medium": MEDIUM_CONFIG,
        "decoder": LARGE_CONFIG,
    }
    return configs[model_size]


def main():
    args = parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, training on CPU (will be slow!)")

    # === Create Model ===
    logger.info(f"Creating {args.model_size} model...")
    policy = create_policy(args.model_size)
    config = get_config_for_size(args.model_size)

    # Add HL-Gauss value head
    value_head = HLGaussValueHead(
        input_dim=config.d_model,
        num_bins=config.num_value_bins,
        value_min=config.value_min,
        value_max=config.value_max,
        sigma=config.hl_gauss_sigma,
    )
    policy.set_value_head(value_head)

    # Log model info
    info = get_model_info(policy)
    logger.info(f"Model: {info['model_class']}")
    logger.info(f"Total Parameters: {info['total_params_m']:.2f}M")
    logger.info(f"Trainable Parameters: {info['trainable_params_m']:.2f}M")

    # === Create DataLoaders ===
    logger.info(f"Loading training data from {args.data_path}...")

    if not args.data_path.exists():
        raise FileNotFoundError(f"Training data not found: {args.data_path}")

    train_loader = create_sequence_dataloader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        shuffle=True,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    logger.info(f"Training samples: {len(train_loader.dataset):,}")
    logger.info(f"Training batches: {len(train_loader):,}")

    val_loader = None
    if args.val_data_path and args.val_data_path.exists():
        logger.info(f"Loading validation data from {args.val_data_path}...")
        val_loader = create_sequence_dataloader(
            data_path=args.val_data_path,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            shuffle=False,
            num_workers=args.num_workers,
        )
        logger.info(f"Validation samples: {len(val_loader.dataset):,}")

    # === Training Config ===
    effective_batch = args.batch_size * args.gradient_accumulation
    logger.info(f"Effective batch size: {effective_batch}")

    bc_config = BCConfig(
        train_data_path=args.data_path,
        val_data_path=args.val_data_path,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler="cosine",
        num_epochs=args.epochs,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        use_wandb=args.use_wandb,
        device=device,
    )

    # Wandb setup
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                "model_size": args.model_size,
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "total_params_m": info["total_params_m"],
                "batch_size": args.batch_size,
                "effective_batch_size": effective_batch,
                "learning_rate": args.lr,
                "warmup_steps": args.warmup_steps,
                "epochs": args.epochs,
                "seq_len": args.seq_len,
                "amp_dtype": args.amp_dtype,
            }
        )

    # === Create Trainer ===
    num_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    logger.info(f"Total training steps: {num_steps:,}")

    trainer = BCTrainer(policy, bc_config, num_training_steps=num_steps)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # === Train ===
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)

    logger.info("Training complete!")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
