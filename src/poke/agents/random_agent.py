"""Random baseline agent."""
import random
from typing import Optional

from poke_env.battle import AbstractBattle

from .team_aware import TeamAwareAgent
from ..teams.loader import TeamPool

class RandomAgent(TeamAwareAgent):
    """Agent that selects uniformly at random from legal actions."""

    def __init__(
        self,
        team_pool: TeamPool,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(team_pool=team_pool, **kwargs)
        self._rng = random.Random(seed)

    def choose_move(self, battle: AbstractBattle):
        """Choose a random legal move.

        Args:
            battle: Current battle state

        Returns:
            BattleOrder for a randomly selected action
        """
        # Collect all legal actions
        actions = []

        # Add available moves
        for move in battle.available_moves:
            actions.append(self.create_order(move))

        # Add available switches
        for pokemon in battle.available_switches:
            actions.append(self.create_order(pokemon))

        # If no actions available, struggle
        if not actions:
            return self.choose_default_move()

        return self._rng.choice(actions)

    def teampreview(self, battle: AbstractBattle):
        """Randomize team order for preview."""
        team_size = len(battle.team)
        order = list(range(1, team_size + 1))
        self._rng.shuffle(order)
        return "/team " + "".join(str(i) for i in order)


class PureRandomAgent(RandomAgent):
    """Random agent without team conditioning (for simpler baselines)."""

    def __init__(self, seed: Optional[int] = None, **kwargs):
        # Skip TeamAwareAgent init, go to Player directly
        from poke_env.player import Player
        Player.__init__(self, **kwargs)
        self._rng = random.Random(seed)

    def choose_move(self, battle: AbstractBattle):
        """Choose a random legal move."""
        actions = []

        for move in battle.available_moves:
            actions.append(self.create_order(move))

        for pokemon in battle.available_switches:
            actions.append(self.create_order(pokemon))

        if not actions:
            return self.choose_default_move()

        return self._rng.choice(actions)
