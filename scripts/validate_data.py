#!/usr/bin/env python
"""Validate parsed battle data."""
import argparse
from pathlib import Path

from poke.data.validation import BattleValidator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parsed battles JSONL")
    args = parser.parse_args()

    validator = BattleValidator()
    report = validator.validate_file(Path(args.input))

    print(f"\n=== Validation Report ===")
    print(f"Total battles: {report.total_battles}")
    print(f"Valid battles: {report.valid_battles} ({100*report.valid_battles/report.total_battles:.1f}%)")
    print(f"Invalid battles: {report.invalid_battles}")

    if report.error_counts:
        print(f"\nErrors:")
        for error, count in sorted(report.error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")

    if report.warning_counts:
        print(f"\nWarnings:")
        for warning, count in sorted(report.warning_counts.items(), key=lambda x: -x[1]):
            print(f"  {warning}: {count}")

if __name__ == "__main__":
    main()
