"""Action space definitions for Pokemon battles."""
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

class ActionType(IntEnum):
    """Types of actions in battle."""
    MOVE_1 = 0
    MOVE_2 = 1
    MOVE_3 = 2
    MOVE_4 = 3
    SWITCH_1 = 4
    SWITCH_2 = 5
    SWITCH_3 = 6
    SWITCH_4 = 7
    SWITCH_5 = 8
    SWITCH_6 = 9

@dataclass
class ActionSpace:
    """Action space for Pokemon battles.

    Actions 0-3: Use move in slot 1-4
    Actions 4-9: Switch to Pokemon in slot 1-6
    """
    num_moves: int = 4
    num_switches: int = 6

    @property
    def total_actions(self) -> int:
        return self.num_moves + self.num_switches

    def decode_action(self, action_idx: int) -> Tuple[str, int]:
        """Decode action index to type and target.

        Returns:
            (action_type, target_index) where action_type is 'move' or 'switch'
        """
        if action_idx < self.num_moves:
            return ("move", action_idx)
        else:
            return ("switch", action_idx - self.num_moves)

    def encode_action(self, action_type: str, target_idx: int) -> int:
        """Encode action to index."""
        if action_type == "move":
            return target_idx
        else:
            return self.num_moves + target_idx
