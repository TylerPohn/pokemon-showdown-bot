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
- EVs must total <= 508 per Pokemon

## Versioning

Team pools are versioned (v1, v2, etc.) to ensure reproducibility.
Once a version is released, it should NOT be modified.
Create a new version instead.
