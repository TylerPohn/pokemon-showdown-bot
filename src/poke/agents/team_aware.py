"""Base class for team-aware agents."""
from typing import Optional

from poke_env.player import Player
from poke_env.battle import AbstractBattle

from .observation import ObservationBuilder, ObservationConfig
from ..teams.loader import TeamPool
from ..teams.integration import PoolTeambuilder

class TeamAwareAgent(Player):
    """Base agent that includes TeamID in observations."""

    def __init__(
        self,
        team_pool: TeamPool,
        obs_config: Optional[ObservationConfig] = None,
        **kwargs
    ):
        # Create teambuilder from pool
        self._teambuilder = PoolTeambuilder(team_pool)

        super().__init__(
            team=self._teambuilder,
            **kwargs
        )

        # Set up observation builder
        self._obs_builder = ObservationBuilder(obs_config or ObservationConfig())
        self._team_id_map = {tid: i for i, tid in enumerate(team_pool.get_ids())}

    def _battle_started_callback(self, battle: AbstractBattle) -> None:
        """Called when a battle starts."""
        # Update team ID in observation builder
        current_team_id = self._teambuilder.current_team_id
        if current_team_id:
            numeric_id = self._team_id_map.get(current_team_id, 0)
            self._obs_builder.set_team_id(numeric_id)

    def get_observation(self, battle: AbstractBattle):
        """Get observation vector for current battle state."""
        return self._obs_builder.build(battle)

    @property
    def observation_size(self) -> int:
        """Get observation vector size."""
        return self._obs_builder.observation_size
