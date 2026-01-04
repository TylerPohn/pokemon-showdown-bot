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
        """Return a team in packed format for poke-env."""
        self._current_team = self.pool.sample()
        return self._current_team.to_packed()

    @property
    def current_team_id(self) -> str | None:
        """Get the ID of the currently selected team."""
        return self._current_team.team_id if self._current_team else None
