#!/usr/bin/env python
"""Validate team files in the pool."""
import argparse
from pathlib import Path

from poke.teams.loader import TeamPool

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to team directory")
    args = parser.parse_args()

    try:
        pool = TeamPool.from_directory(Path(args.path))
        print(f"Successfully loaded {len(pool)} teams")

        for team_id in pool.get_ids():
            team = pool[team_id]
            print(f"\n{team.name} ({team_id}):")
            print(f"  Pokemon: {len(team)}")

            issues = []
            if len(team) != 6:
                issues.append(f"Expected 6 Pokemon, found {len(team)}")

            for mon in team.pokemon:
                if len(mon.moves) != 4:
                    issues.append(f"{mon.species}: {len(mon.moves)} moves")

                total_evs = sum(mon.evs.values())
                if total_evs > 510:
                    issues.append(f"{mon.species}: {total_evs} EVs")

            if issues:
                for issue in issues:
                    print(f"  WARNING: {issue}")
            else:
                print("  Valid")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
