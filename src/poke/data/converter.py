"""Convert parsed battles to training trajectories."""
import logging
from typing import Iterator, Optional

from .models import ParsedBattle, TurnRecord, ActionType
from .trajectory import Trajectory, TrajectoryStep, Observation
from .encoders import SpeciesEncoder, StatusEncoder, WeatherEncoder

logger = logging.getLogger(__name__)

class TrajectoryConverter:
    """Convert parsed battles to RL trajectories."""

    def __init__(
        self,
        species_encoder: SpeciesEncoder,
        reward_shaping: bool = False,
    ):
        self.species_encoder = species_encoder
        self.reward_shaping = reward_shaping

    def convert(self, battle: ParsedBattle) -> Iterator[Trajectory]:
        """Convert a parsed battle to trajectories.

        Yields two trajectories: one for each player.
        """
        for player in ["p1", "p2"]:
            try:
                trajectory = self._convert_player(battle, player)
                if trajectory and len(trajectory) > 0:
                    yield trajectory
            except Exception as e:
                logger.warning(f"Failed to convert {battle.replay_id} for {player}: {e}")

    def _convert_player(self, battle: ParsedBattle, player: str) -> Optional[Trajectory]:
        """Convert battle from one player's perspective."""
        trajectory = Trajectory(
            replay_id=battle.replay_id,
            player=player,
        )

        # Determine if this player won
        is_winner = battle.winner == battle.players[0 if player == "p1" else 1]

        for i, turn in enumerate(battle.turns):
            # Get action for this player
            action = turn.p1_action if player == "p1" else turn.p2_action
            if action is None:
                continue

            # Build observation
            obs = self._build_observation(turn, player)

            # Determine reward
            is_last = (i == len(battle.turns) - 1)
            if is_last:
                reward = 1.0 if is_winner else -1.0
            elif self.reward_shaping:
                reward = self._compute_shaped_reward(turn, player)
            else:
                reward = 0.0

            # Convert action
            action_type = 0 if action.action_type == ActionType.MOVE else 1
            action_target = self._encode_action_target(action)

            step = TrajectoryStep(
                observation=obs,
                action_type=action_type,
                action_target=action_target,
                reward=reward,
                done=is_last,
            )

            trajectory.steps.append(step)
            trajectory.total_reward += reward

        return trajectory

    def _build_observation(self, turn: TurnRecord, player: str) -> Observation:
        """Build observation from turn state."""
        state = turn.state_before
        own_state = state.player1 if player == "p1" else state.player2
        opp_state = state.player2 if player == "p1" else state.player1

        # Find active Pokemon
        active = next((p for p in own_state.team if p.active), None)
        opp_active = next((p for p in opp_state.team if p.active), None)

        return Observation(
            turn=state.turn,
            active_species_id=self.species_encoder.encode(active.species) if active else 0,
            active_hp=active.hp_fraction if active else 0.0,
            active_status=StatusEncoder.encode(active.status.value) if active else 0,
            team_hp=[p.hp_fraction for p in own_state.team[:6]] + [0.0] * (6 - len(own_state.team)),
            team_status=[StatusEncoder.encode(p.status.value) for p in own_state.team[:6]] + [0] * (6 - len(own_state.team)),
            team_fainted=[p.fainted for p in own_state.team[:6]] + [False] * (6 - len(own_state.team)),
            opp_active_species_id=self.species_encoder.encode(opp_active.species) if opp_active else 0,
            opp_active_hp=opp_active.hp_fraction if opp_active else 1.0,
            opp_team_revealed=[self.species_encoder.encode(p.species) for p in opp_state.team[:6]] + [0] * (6 - len(opp_state.team)),
            weather_id=WeatherEncoder.encode(state.field.weather),
            terrain_id=0,  # TODO: terrain encoding
            own_stealth_rock=own_state.hazards.stealth_rock,
            own_spikes=own_state.hazards.spikes,
            own_toxic_spikes=own_state.hazards.toxic_spikes,
            own_sticky_web=own_state.hazards.sticky_web,
            opp_stealth_rock=opp_state.hazards.stealth_rock,
            opp_spikes=opp_state.hazards.spikes,
            opp_toxic_spikes=opp_state.hazards.toxic_spikes,
            opp_sticky_web=opp_state.hazards.sticky_web,
        )

    def _encode_action_target(self, action) -> int:
        """Encode action target to index."""
        # For now, return 0. In practice, need move/pokemon index mapping.
        return 0

    def _compute_shaped_reward(self, turn: TurnRecord, player: str) -> float:
        """Compute intermediate reward for shaping."""
        # TODO: Implement reward shaping (damage dealt, KOs, etc.)
        return 0.0
