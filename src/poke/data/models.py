"""Data models for parsed battle data."""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel

class ActionType(str, Enum):
    MOVE = "move"
    SWITCH = "switch"

class Status(str, Enum):
    NONE = ""
    BURN = "brn"
    FREEZE = "frz"
    PARALYSIS = "par"
    POISON = "psn"
    TOXIC = "tox"
    SLEEP = "slp"

class PokemonState(BaseModel):
    """State of a single Pokemon."""
    species: str
    hp_fraction: float  # 0.0 to 1.0
    status: Status = Status.NONE
    fainted: bool = False
    active: bool = False

class FieldState(BaseModel):
    """State of battlefield conditions."""
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    tailwind_p1: bool = False
    tailwind_p2: bool = False

class HazardState(BaseModel):
    """Entry hazards on each side."""
    stealth_rock: bool = False
    spikes: int = 0  # 0-3 layers
    toxic_spikes: int = 0  # 0-2 layers
    sticky_web: bool = False

class PlayerState(BaseModel):
    """State of one player's side."""
    team: List[PokemonState]
    hazards: HazardState = HazardState()

class BattleState(BaseModel):
    """Complete battle state at a point in time."""
    turn: int
    player1: PlayerState
    player2: PlayerState
    field: FieldState = FieldState()

class Action(BaseModel):
    """An action taken by a player."""
    player: str  # "p1" or "p2"
    action_type: ActionType
    target: str  # Move name or Pokemon species

class TurnRecord(BaseModel):
    """Record of a single turn."""
    turn: int
    state_before: BattleState
    p1_action: Optional[Action] = None
    p2_action: Optional[Action] = None
    winner: Optional[str] = None  # Set on final turn

class ParsedBattle(BaseModel):
    """A fully parsed battle."""
    replay_id: str
    format: str
    players: tuple[str, str]
    rating: Optional[int] = None
    winner: Optional[str] = None
    turns: List[TurnRecord]
