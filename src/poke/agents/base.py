"""Base agent classes."""
from abc import abstractmethod
from typing import List

from poke_env.player import Player
from poke_env.battle import Battle, AbstractBattle

class BaseAgent(Player):
    """Base class for all Pokemon battle agents."""

    def __init__(self, battle_format: str = "gen9ou", **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.battle_count = 0

    @abstractmethod
    def choose_move(self, battle: AbstractBattle):
        """Choose a move for the current turn.

        Args:
            battle: Current battle state

        Returns:
            BattleOrder for the chosen action
        """
        pass

    def get_legal_moves(self, battle: AbstractBattle) -> List[str]:
        """Get list of legal move orders."""
        orders = []

        # Available moves
        for move in battle.available_moves:
            orders.append(self.create_order(move))

        # Available switches
        for pokemon in battle.available_switches:
            orders.append(self.create_order(pokemon))

        return orders

    def teampreview(self, battle: AbstractBattle):
        """Handle team preview phase.

        Default: send Pokemon in slot order.
        """
        return "/team 123456"
