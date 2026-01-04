#!/usr/bin/env python
"""Compare trained models."""
import argparse
from pathlib import Path

from poke.training.comparison import compare_checkpoints, print_comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Checkpoint directories to compare")
    parser.add_argument("--metric", default="val_loss", choices=["val_loss", "val_accuracy", "train_loss"])
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dirs]
    models = compare_checkpoints(dirs, metric=args.metric)
    print_comparison(models)

if __name__ == "__main__":
    main()
