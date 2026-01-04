#!/usr/bin/env python
"""Compute statistics about the team pool."""
import argparse
from pathlib import Path
from collections import Counter

from poke.teams.loader import TeamPool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to team directory")
    args = parser.parse_args()

    pool = TeamPool.from_directory(Path(args.path))

    species_counts = Counter()
    item_counts = Counter()
    ability_counts = Counter()

    for team_id in pool.get_ids():
        team = pool[team_id]
        for mon in team.pokemon:
            species_counts[mon.species] += 1
            if mon.item:
                item_counts[mon.item] += 1
            if mon.ability:
                ability_counts[mon.ability] += 1

    print(f"\n=== Team Pool Statistics ===")
    print(f"Total teams: {len(pool)}")

    print(f"\nTop 10 Pokemon:")
    for species, count in species_counts.most_common(10):
        print(f"  {species}: {count}")

    print(f"\nTop 10 Items:")
    for item, count in item_counts.most_common(10):
        print(f"  {item}: {count}")

    print(f"\nTop 10 Abilities:")
    for ability, count in ability_counts.most_common(10):
        print(f"  {ability}: {count}")

if __name__ == "__main__":
    main()
