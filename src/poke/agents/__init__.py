"""Agent implementations."""
from .base import BaseAgent
from .random_agent import RandomAgent, PureRandomAgent
from .heuristic_agent import MaxDamageAgent, SmartHeuristicAgent
from .observation import ObservationBuilder, ObservationConfig
from .team_aware import TeamAwareAgent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "PureRandomAgent",
    "MaxDamageAgent",
    "SmartHeuristicAgent",
    "ObservationBuilder",
    "ObservationConfig",
    "TeamAwareAgent",
]

BASELINE_AGENTS = {
    "random": PureRandomAgent,
    "random_team": RandomAgent,
    "maxdamage": MaxDamageAgent,
    "smart": SmartHeuristicAgent,
}
