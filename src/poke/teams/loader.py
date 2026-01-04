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
