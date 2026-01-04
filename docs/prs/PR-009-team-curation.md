COMPLETED

# PR-009: Curate Sample Teams

## Dependencies
- PR-008 (Team File Format and Loader)

## Overview
Curate an initial set of Gen9 OU teams from public sources (Smogon sample teams). These form the static team pool for training and evaluation.

## Tech Choices
- **Primary Source:** Smogon Gen9 OU Sample Teams
- **Secondary Source:** (Optional) High-ladder teams from replay data
- **Minimum Pool Size:** 20 teams
- **Target Pool Size:** 50+ teams

## Tasks

### 1. Research team sources
Document available sources:
- Smogon forums: Gen9 OU Sample Teams thread
- Smogon strategy dex team archive
- Tournament teams from Smogon Tour / SPL

### 2. Create team curation script
Create `scripts/fetch_sample_teams.py`:
```python
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
```

### 3. Document manual curation process
Create `teams/README.md`:
```markdown
# Team Pool

This directory contains the static team pool for training and evaluation.

## Structure

```
teams/
└── gen9ou/
    └── v1/
        ├── team_001.txt
        ├── team_002.txt
        └── ...
```

## Adding Teams

1. **Format:** Use standard Showdown paste format
2. **One team per file:** Each `.txt` file contains exactly one team
3. **Naming:** Use descriptive names (e.g., `rain_pelipper_azu.txt`)
4. **Validation:** Run `python -m poke.teams.loader teams/gen9ou/v1` to verify

## Sources

### Smogon Sample Teams (Primary)
- Thread: https://www.smogon.com/forums/threads/gen-9-ou-sample-teams.3712513/
- These are curated, viable teams for the current metagame

### Tournament Teams
- SPL/SCL team dumps
- Smogon Tour winning teams

## Team Requirements

- Must be legal Gen9 OU
- 6 Pokemon required
- All moves, items, abilities must be legal
- EVs must total ≤ 508 per Pokemon

## Versioning

Team pools are versioned (v1, v2, etc.) to ensure reproducibility.
Once a version is released, it should NOT be modified.
Create a new version instead.
```

### 4. Curate initial team set
Manually curate at least 20 teams:

Example teams to include:
1. **Rain Balance** - Pelipper + Barraskewda/Kingdra
2. **Sun HO** - Torkoal + Venusaur
3. **Dragapult Offense** - Standard Dragapult offense
4. **Gholdengo Bulky** - Pivot-heavy Gholdengo teams
5. **Stall** - Classic stall with Dondozo/Toxapex
6. **Hyper Offense** - Lead + sweepers
7. **Balance** - Multiple balance cores
8. **Semi-Stall** - Defensive with win conditions

Create `teams/gen9ou/v1/sample_balance.txt`:
```
Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fighting
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Focus Blast
- Nasty Plot

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Rapid Spin
- Knock Off
- Ice Spinner

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Dark
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance

Dragapult @ Choice Specs
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- U-turn
- Thunderbolt

Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Fairy
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Salt Cure
- Recover
- Stealth Rock
- Body Press

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
- Future Sight
- Sludge Bomb
- Slack Off
- Thunder Wave
```

### 5. Create validation script
Create `scripts/validate_teams.py`:
```python
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
                    print(f"  ⚠ {issue}")
            else:
                print("  ✓ Valid")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
```

### 6. Add team pool statistics
Create `scripts/team_stats.py`:
```python
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
```

## Acceptance Criteria
- [ ] Minimum 20 teams in `teams/gen9ou/v1/`
- [ ] All teams pass validation (6 Pokemon, 4 moves each)
- [ ] Teams represent diverse archetypes (offense, balance, stall)
- [ ] Documentation explains team sources and curation
- [ ] Validation script catches common errors

## Notes
- Focus on quality over quantity initially
- Include teams from different archetypes for diversity
- Prefer tournament-proven teams over experimental builds
- This is a manual curation task; automation is secondary

## Estimated Complexity
Low - Manual curation with simple validation
