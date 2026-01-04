#!/usr/bin/env python
"""Compute dataset statistics."""
import argparse
from pathlib import Path

from poke.data.statistics import StatisticsCollector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parsed battles JSONL")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    collector = StatisticsCollector()
    stats = collector.process_file(Path(args.input))

    print(f"\n=== Dataset Statistics ===")
    print(f"Total battles: {stats.total_battles}")
    print(f"Total turns: {stats.total_turns}")
    print(f"Average turns per battle: {stats.avg_turns:.1f}")
    print(f"Unique species: {stats.unique_species}")
    print(f"Unique moves: {stats.unique_moves}")

    print(f"\nTop 10 Species:")
    for species, count in sorted(stats.species_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {species}: {count}")

    print(f"\nTop 10 Moves:")
    for move, count in sorted(stats.move_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {move}: {count}")

    if args.output:
        collector.save_report(Path(args.output))
        print(f"\nSaved full report to {args.output}")

if __name__ == "__main__":
    main()
