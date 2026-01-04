COMPLETED

# PR-008: Team File Format and Loader

## Dependencies
- PR-001 (Project Setup)

## Overview
Define the team file format and implement a loader for static team pools. Teams are loaded before battles and sampled uniformly.

## Tech Choices
- **Format:** Pokemon Showdown paste format (standard)
- **Storage:** One team per file in `teams/gen9ou/v1/`
- **TeamID:** Derived from filename hash

## Tasks

### 1. Define team file structure
Create directory structure:
```
teams/
└── gen9ou/
    └── v1/
        ├── team_001.txt
        ├── team_002.txt
        └── ...
```

Each file uses Showdown paste format:
```
Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Hidden Power Ice

Charizard @ Heavy-Duty Boots
...
```

### 2. Create team data models
Create `src/poke/teams/models.py`:
```python
"""Data models for Pokemon teams."""
from dataclasses import dataclass, field
from typing import List, Optional
import hashlib

@dataclass
class Pokemon:
    """A single Pokemon on a team."""
    species: str
    nickname: Optional[str] = None
    item: Optional[str] = None
    ability: Optional[str] = None
    evs: dict[str, int] = field(default_factory=dict)
    ivs: dict[str, int] = field(default_factory=dict)
    nature: Optional[str] = None
    moves: List[str] = field(default_factory=list)
    level: int = 100
    gender: Optional[str] = None
    shiny: bool = False
    tera_type: Optional[str] = None

    def to_showdown(self) -> str:
        """Convert to Showdown paste format."""
        lines = []

        # Name line
        name_line = self.species
        if self.nickname:
            name_line = f"{self.nickname} ({self.species})"
        if self.gender:
            name_line += f" ({self.gender})"
        if self.item:
            name_line += f" @ {self.item}"
        lines.append(name_line)

        if self.ability:
            lines.append(f"Ability: {self.ability}")

        if self.level != 100:
            lines.append(f"Level: {self.level}")

        if self.shiny:
            lines.append("Shiny: Yes")

        if self.tera_type:
            lines.append(f"Tera Type: {self.tera_type}")

        if self.evs:
            ev_str = " / ".join(f"{v} {k}" for k, v in self.evs.items() if v > 0)
            if ev_str:
                lines.append(f"EVs: {ev_str}")

        if self.ivs:
            iv_str = " / ".join(f"{v} {k}" for k, v in self.ivs.items() if v < 31)
            if iv_str:
                lines.append(f"IVs: {iv_str}")

        if self.nature:
            lines.append(f"{self.nature} Nature")

        for move in self.moves:
            lines.append(f"- {move}")

        return "\n".join(lines)


@dataclass
class Team:
    """A complete Pokemon team."""
    team_id: str
    name: str
    pokemon: List[Pokemon]
    format: str = "gen9ou"
    source: Optional[str] = None  # e.g., "smogon_samples"

    def __len__(self) -> int:
        return len(self.pokemon)

    def to_showdown(self) -> str:
        """Convert team to Showdown paste format."""
        return "\n\n".join(p.to_showdown() for p in self.pokemon)

    @staticmethod
    def generate_id(content: str) -> str:
        """Generate stable team ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]
```

### 3. Implement team parser
Create `src/poke/teams/parser.py`:
```python
"""Parser for Showdown paste format."""
import re
from typing import Optional
from .models import Pokemon, Team

class TeamParser:
    """Parse Showdown paste format into Team objects."""

    # Regex patterns
    NAME_PATTERN = re.compile(
        r"^(?:(.+?)\s*\(([^)]+)\)|([^(@]+))(?:\s*\(([MF])\))?(?:\s*@\s*(.+))?$"
    )
    EV_PATTERN = re.compile(r"(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)", re.IGNORECASE)
    IV_PATTERN = re.compile(r"(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)", re.IGNORECASE)

    def parse_pokemon(self, text: str) -> Optional[Pokemon]:
        """Parse a single Pokemon from paste format."""
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return None

        pokemon = Pokemon(species="Unknown")

        # Parse first line (name/species/item)
        first_line = lines[0]
        if match := self.NAME_PATTERN.match(first_line):
            nickname, species1, species2, gender, item = match.groups()
            pokemon.species = species1 or species2 or "Unknown"
            pokemon.nickname = nickname if nickname else None
            pokemon.gender = gender
            pokemon.item = item

        # Parse remaining lines
        for line in lines[1:]:
            line_lower = line.lower()

            if line_lower.startswith("ability:"):
                pokemon.ability = line.split(":", 1)[1].strip()

            elif line_lower.startswith("level:"):
                try:
                    pokemon.level = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif line_lower.startswith("evs:"):
                ev_part = line.split(":", 1)[1]
                for match in self.EV_PATTERN.finditer(ev_part):
                    pokemon.evs[match.group(2)] = int(match.group(1))

            elif line_lower.startswith("ivs:"):
                iv_part = line.split(":", 1)[1]
                for match in self.IV_PATTERN.finditer(iv_part):
                    pokemon.ivs[match.group(2)] = int(match.group(1))

            elif "nature" in line_lower:
                pokemon.nature = line.split()[0]

            elif line_lower.startswith("tera type:"):
                pokemon.tera_type = line.split(":", 1)[1].strip()

            elif line_lower.startswith("shiny:"):
                pokemon.shiny = "yes" in line_lower

            elif line.startswith("-"):
                move = line[1:].strip()
                if move:
                    pokemon.moves.append(move)

        return pokemon

    def parse_team(self, text: str, team_id: Optional[str] = None, name: str = "Unnamed") -> Team:
        """Parse a full team from paste format."""
        # Split by double newline to get individual Pokemon
        pokemon_texts = re.split(r"\n\s*\n", text.strip())

        pokemon_list = []
        for ptext in pokemon_texts:
            if ptext.strip():
                mon = self.parse_pokemon(ptext)
                if mon:
                    pokemon_list.append(mon)

        if team_id is None:
            team_id = Team.generate_id(text)

        return Team(
            team_id=team_id,
            name=name,
            pokemon=pokemon_list,
        )
```

### 4. Implement team loader
Create `src/poke/teams/loader.py`:
```python
"""Team pool loader."""
import random
import logging
from pathlib import Path
from typing import List, Optional

from .models import Team
from .parser import TeamParser

logger = logging.getLogger(__name__)

class TeamPool:
    """A pool of teams for battle selection."""

    def __init__(self, teams: List[Team]):
        self._teams = teams
        self._by_id = {t.team_id: t for t in teams}

    def __len__(self) -> int:
        return len(self._teams)

    def __getitem__(self, team_id: str) -> Team:
        return self._by_id[team_id]

    def sample(self) -> Team:
        """Sample a random team from the pool."""
        return random.choice(self._teams)

    def get_ids(self) -> List[str]:
        """Get all team IDs."""
        return list(self._by_id.keys())

    @classmethod
    def from_directory(cls, path: Path, format: str = "gen9ou") -> "TeamPool":
        """Load team pool from a directory of team files."""
        parser = TeamParser()
        teams = []

        if not path.exists():
            raise FileNotFoundError(f"Team directory not found: {path}")

        for team_file in sorted(path.glob("*.txt")):
            try:
                content = team_file.read_text()
                team = parser.parse_team(
                    content,
                    name=team_file.stem,
                )
                team.format = format
                teams.append(team)
                logger.debug(f"Loaded team: {team.name} ({len(team)} Pokemon)")
            except Exception as e:
                logger.warning(f"Failed to load {team_file}: {e}")

        logger.info(f"Loaded {len(teams)} teams from {path}")
        return cls(teams)


def get_default_pool(format: str = "gen9ou", version: str = "v1") -> TeamPool:
    """Get the default team pool for a format."""
    # Look relative to project root
    base_path = Path("teams") / format / version
    return TeamPool.from_directory(base_path, format=format)
```

### 5. Add poke-env team integration
Create `src/poke/teams/integration.py`:
```python
"""Integration with poke-env for team usage."""
from poke_env.teambuilder import Teambuilder

from .models import Team
from .loader import TeamPool

class PoolTeambuilder(Teambuilder):
    """Teambuilder that samples from a team pool."""

    def __init__(self, pool: TeamPool):
        self.pool = pool
        self._current_team: Team | None = None

    def yield_team(self) -> str:
        """Return a team in Showdown format."""
        self._current_team = self.pool.sample()
        return self._current_team.to_showdown()

    @property
    def current_team_id(self) -> str | None:
        """Get the ID of the currently selected team."""
        return self._current_team.team_id if self._current_team else None
```

### 6. Write unit tests
Create `tests/teams/test_parser.py`:
```python
"""Tests for team parser."""
import pytest
from poke.teams.parser import TeamParser

SAMPLE_POKEMON = """
Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Hidden Power Ice
"""

SAMPLE_TEAM = """
Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Hidden Power Ice

Charizard @ Heavy-Duty Boots
Ability: Solar Power
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Flamethrower
- Dragon Pulse
- Roost
- Focus Blast
"""

def test_parse_pokemon():
    parser = TeamParser()
    mon = parser.parse_pokemon(SAMPLE_POKEMON)

    assert mon.species == "Pikachu"
    assert mon.item == "Light Ball"
    assert mon.ability == "Static"
    assert mon.nature == "Timid"
    assert len(mon.moves) == 4
    assert "Thunderbolt" in mon.moves

def test_parse_evs():
    parser = TeamParser()
    mon = parser.parse_pokemon(SAMPLE_POKEMON)

    assert mon.evs.get("SpA") == 252
    assert mon.evs.get("Spe") == 252
    assert mon.evs.get("SpD") == 4

def test_parse_team():
    parser = TeamParser()
    team = parser.parse_team(SAMPLE_TEAM, name="Test Team")

    assert len(team) == 2
    assert team.pokemon[0].species == "Pikachu"
    assert team.pokemon[1].species == "Charizard"

def test_team_id_generation():
    parser = TeamParser()
    team1 = parser.parse_team(SAMPLE_TEAM)
    team2 = parser.parse_team(SAMPLE_TEAM)

    # Same content should produce same ID
    assert team1.team_id == team2.team_id

def test_roundtrip():
    parser = TeamParser()
    team = parser.parse_team(SAMPLE_TEAM)
    output = team.to_showdown()

    # Parse the output again
    team2 = parser.parse_team(output)
    assert len(team2) == len(team)
```

## Acceptance Criteria
- [ ] Parses Showdown paste format correctly
- [ ] Loads teams from directory of `.txt` files
- [ ] Generates stable TeamIDs from content
- [ ] Integrates with poke-env Teambuilder
- [ ] Supports all Gen9 OU team features (items, abilities, EVs, Tera)
- [ ] Provides random team sampling

## Estimated Complexity
Medium - Format parsing with many edge cases
