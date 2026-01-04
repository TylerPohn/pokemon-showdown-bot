"""Trajectory data structures for RL training."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class Observation:
    """Observation at a single timestep.

    All fields are designed to be easily converted to tensors.
    """
    turn: int

    # Active Pokemon (own)
    active_species_id: int
    active_hp: float  # 0-1
    active_status: int  # Encoded status

    # Own team state (6 Pokemon max)
    team_hp: List[float]  # 6 floats, 0-1
    team_status: List[int]  # 6 ints
    team_fainted: List[bool]  # 6 bools

    # Known opponent state
    opp_active_species_id: int
    opp_active_hp: float
    opp_team_revealed: List[int]  # Species IDs, 0 = unknown

    # Field conditions
    weather_id: int
    terrain_id: int

    # Hazards (own side)
    own_stealth_rock: bool
    own_spikes: int
    own_toxic_spikes: int
    own_sticky_web: bool

    # Hazards (opponent side)
    opp_stealth_rock: bool
    opp_spikes: int
    opp_toxic_spikes: int
    opp_sticky_web: bool

    # Team ID (for conditioning)
    team_id: int = 0

@dataclass
class TrajectoryStep:
    """Single step in a trajectory."""
    observation: Observation
    action_type: int  # 0 = move, 1 = switch
    action_target: int  # Move index (0-3) or switch target (0-5)
    reward: float
    done: bool

@dataclass
class Trajectory:
    """Complete trajectory from one player's perspective."""
    replay_id: str
    player: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0

    def __len__(self) -> int:
        return len(self.steps)
