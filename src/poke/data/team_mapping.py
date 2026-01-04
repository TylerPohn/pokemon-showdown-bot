"""Utilities for mapping teams to IDs in trajectory data."""
from pathlib import Path
from typing import Dict, Optional
import json

from ..teams.loader import TeamPool
from ..teams.models import Team

class TeamIDMapper:
    """Maps team content to numeric IDs for trajectory data."""

    def __init__(self, team_pool: Optional[TeamPool] = None):
        self._content_to_id: Dict[str, int] = {}
        self._id_to_content: Dict[int, str] = {}

        if team_pool:
            for i, team_id in enumerate(team_pool.get_ids()):
                team = team_pool[team_id]
                content_hash = Team.generate_id(team.to_showdown())
                self._content_to_id[content_hash] = i
                self._id_to_content[i] = content_hash

    def get_id(self, team_content: str) -> int:
        """Get numeric ID for team content.

        Returns 0 (unknown) if team not in pool.
        """
        content_hash = Team.generate_id(team_content)
        return self._content_to_id.get(content_hash, 0)

    def save(self, path: Path) -> None:
        """Save mapping to JSON."""
        path.write_text(json.dumps(self._content_to_id, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TeamIDMapper":
        """Load mapping from JSON."""
        mapper = cls()
        data = json.loads(path.read_text())
        mapper._content_to_id = data
        mapper._id_to_content = {v: k for k, v in data.items()}
        return mapper

    def __len__(self) -> int:
        return len(self._content_to_id)
