#!/usr/bin/env python
"""Fetch sample teams from Smogon.

This script provides utilities for fetching and formatting teams.
Due to website structure changes, manual curation may be required.
"""
import argparse
import re
from pathlib import Path

# Note: Smogon sample teams page structure varies.
# This provides the framework - manual intervention may be needed.

SMOGON_SAMPLE_TEAMS_URL = "https://www.smogon.com/forums/threads/gen-9-ou-sample-teams.3712513/"

def clean_team_text(text: str) -> str:
    """Clean and normalize team paste text."""
    # Remove extra whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def save_team(content: str, output_dir: Path, name: str) -> None:
    """Save a team to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^a-z0-9]+", "_", name.lower()) + ".txt"
    output_path = output_dir / filename
    output_path.write_text(content)
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="teams/gen9ou/v1", help="Output directory")
    parser.add_argument("--input", help="Input file with teams (one team per section)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.input:
        # Parse from input file
        content = Path(args.input).read_text()
        teams = content.split("===")  # Common delimiter

        for i, team in enumerate(teams):
            team = clean_team_text(team)
            if team and len(team) > 50:  # Minimum viable team
                name = f"team_{i+1:03d}"
                save_team(team, output_dir, name)
    else:
        print(f"Manual curation required.")
        print(f"1. Visit: {SMOGON_SAMPLE_TEAMS_URL}")
        print(f"2. Copy team pastes to a text file")
        print(f"3. Run: python {__file__} --input teams.txt")

if __name__ == "__main__":
    main()
