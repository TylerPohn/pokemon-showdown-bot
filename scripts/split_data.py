#!/usr/bin/env python
"""Split trajectory data into train/val sets."""
import argparse
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    train_path = input_path.with_suffix(".train.jsonl")
    val_path = input_path.with_suffix(".val.jsonl")

    with open(input_path) as f:
        lines = f.readlines()

    random.shuffle(lines)

    split_idx = int(len(lines) * (1 - args.val_ratio))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    with open(train_path, "w") as f:
        f.writelines(train_lines)

    with open(val_path, "w") as f:
        f.writelines(val_lines)

    print(f"Train: {len(train_lines)} trajectories -> {train_path}")
    print(f"Val: {len(val_lines)} trajectories -> {val_path}")

if __name__ == "__main__":
    main()
