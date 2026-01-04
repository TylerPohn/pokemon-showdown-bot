"""Neural network-based battle agent.

Supports both:
- MLPPolicy: Original small policy network
- DecoderPolicy: 200M parameter Metamon-style transformer
"""
from collections import deque
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn as nn

from poke_env.battle import AbstractBattle, Pokemon

from .team_aware import TeamAwareAgent
from ..models.policy import MLPPolicy
from ..models.masking import ActionMask, ActionSpace
from ..models.preprocessing import FeaturePreprocessor
from ..teams.loader import TeamPool


# Status encoding map
STATUS_ENCODING = {
    None: 0,
    "brn": 1,
    "frz": 2,
    "par": 3,
    "psn": 4,
    "tox": 5,
    "slp": 6,
}

# Weather encoding map
WEATHER_ENCODING = {
    None: 0,
    "sunnyday": 1,
    "raindance": 2,
    "sandstorm": 3,
    "hail": 4,
    "snow": 4,  # Gen 9 hail replacement
}

# Terrain encoding map
TERRAIN_ENCODING = {
    None: 0,
    "electricterrain": 1,
    "grassyterrain": 2,
    "mistyterrain": 3,
    "psychicterrain": 4,
}


class NeuralNetworkAgent(TeamAwareAgent):
    """Agent that uses a trained neural network policy."""

    def __init__(
        self,
        policy: MLPPolicy,
        team_pool: TeamPool,
        action_space: ActionSpace = None,
        deterministic: bool = False,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(team_pool=team_pool, **kwargs)

        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.deterministic = deterministic

        self.action_space = action_space or ActionSpace()
        self.action_mask = ActionMask(self.action_space)
        self.preprocessor = FeaturePreprocessor()

    def _build_observation_dict(self, battle: AbstractBattle) -> Dict[str, Any]:
        """Build observation dict matching training data format.

        This creates a dict with the same fields as the Observation dataclass
        used during training.
        """
        # Active Pokemon (own)
        active = battle.active_pokemon
        active_species_id = 0
        active_hp = 1.0
        active_status = 0

        if active:
            active_hp = active.current_hp_fraction if active.current_hp_fraction else 1.0
            active_status = STATUS_ENCODING.get(active.status.name if active.status else None, 0)

        # Own team state (6 Pokemon max)
        team_hp: List[float] = []
        team_status: List[int] = []
        team_fainted: List[bool] = []

        team_list = list(battle.team.values())
        for i in range(6):
            if i < len(team_list):
                mon = team_list[i]
                team_hp.append(mon.current_hp_fraction if mon.current_hp_fraction else 0.0)
                team_status.append(STATUS_ENCODING.get(mon.status.name if mon.status else None, 0))
                team_fainted.append(mon.fainted)
            else:
                team_hp.append(0.0)
                team_status.append(0)
                team_fainted.append(True)

        # Opponent state
        opp_active = battle.opponent_active_pokemon
        opp_active_species_id = 0
        opp_active_hp = 1.0

        if opp_active:
            opp_active_hp = opp_active.current_hp_fraction if opp_active.current_hp_fraction else 1.0

        # Revealed opponent Pokemon
        opp_team_revealed: List[int] = [0] * 6
        opp_team_list = list(battle.opponent_team.values())
        for i, mon in enumerate(opp_team_list[:6]):
            opp_team_revealed[i] = 1  # Just mark as revealed

        # Weather
        weather_id = 0
        if battle.weather:
            for w_key in battle.weather.keys():
                weather_id = WEATHER_ENCODING.get(str(w_key).lower(), 0)
                break

        # Terrain
        terrain_id = 0
        if battle.fields:
            for f_key in battle.fields.keys():
                terrain_id = TERRAIN_ENCODING.get(str(f_key).lower(), 0)
                break

        # Hazards (own side)
        side = battle.side_conditions
        own_stealth_rock = any("stealthrock" in str(k).lower() for k in side)
        own_spikes = sum(1 for k in side if "spikes" in str(k).lower() and "toxic" not in str(k).lower())
        own_toxic_spikes = sum(1 for k in side if "toxicspikes" in str(k).lower())
        own_sticky_web = any("stickyweb" in str(k).lower() for k in side)

        # Hazards (opponent side)
        opp_side = battle.opponent_side_conditions
        opp_stealth_rock = any("stealthrock" in str(k).lower() for k in opp_side)
        opp_spikes = sum(1 for k in opp_side if "spikes" in str(k).lower() and "toxic" not in str(k).lower())
        opp_toxic_spikes = sum(1 for k in opp_side if "toxicspikes" in str(k).lower())
        opp_sticky_web = any("stickyweb" in str(k).lower() for k in opp_side)

        # Team ID
        team_id = 0
        if hasattr(self, '_obs_builder') and hasattr(self._obs_builder, '_team_id'):
            team_id = self._obs_builder._team_id or 0

        return {
            "turn": battle.turn,
            "active_species_id": active_species_id,
            "active_hp": active_hp,
            "active_status": active_status,
            "team_hp": team_hp,
            "team_status": team_status,
            "team_fainted": team_fainted,
            "opp_active_species_id": opp_active_species_id,
            "opp_active_hp": opp_active_hp,
            "opp_team_revealed": opp_team_revealed,
            "weather_id": weather_id,
            "terrain_id": terrain_id,
            "own_stealth_rock": own_stealth_rock,
            "own_spikes": own_spikes,
            "own_toxic_spikes": own_toxic_spikes,
            "own_sticky_web": own_sticky_web,
            "opp_stealth_rock": opp_stealth_rock,
            "opp_spikes": opp_spikes,
            "opp_toxic_spikes": opp_toxic_spikes,
            "opp_sticky_web": opp_sticky_web,
            "team_id": team_id,
        }

    def choose_move(self, battle: AbstractBattle):
        """Choose move using the neural network policy."""
        with torch.no_grad():
            # Get observation as dict (matching training format)
            obs_dict = self._build_observation_dict(battle)

            # Preprocess for model
            obs_tensors_raw = self.preprocessor.preprocess(obs_dict)
            obs_tensors = {
                k: v.unsqueeze(0).to(self.device)
                for k, v in obs_tensors_raw.items()
            }

            # Get action mask
            mask = self.action_mask.get_mask_tensor(battle, self.device)
            mask = mask.unsqueeze(0)

            # Get action from policy
            action_probs, _ = self.policy(obs_tensors, mask)

            if self.deterministic:
                action_idx = action_probs.argmax(dim=-1).item()
            else:
                action_idx = torch.multinomial(action_probs, 1).item()

        # Convert to battle order
        return self._action_to_order(action_idx, battle)

    def _action_to_order(self, action_idx: int, battle: AbstractBattle):
        """Convert action index to poke-env order."""
        action_type, target_idx = self.action_space.decode_action(action_idx)

        if action_type == "move":
            if target_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[target_idx])
        else:
            if target_idx < len(battle.available_switches):
                return self.create_order(battle.available_switches[target_idx])

        # Fallback
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])

        return self.choose_default_move()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        team_pool: TeamPool,
        **kwargs
    ) -> "NeuralNetworkAgent":
        """Load agent from checkpoint."""
        from ..models.config import EncoderConfig
        from ..models.factory import create_policy

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create policy
        config = EncoderConfig()
        policy = create_policy("mlp", encoder_config=config)
        policy.load_state_dict(checkpoint["model_state_dict"])

        return cls(policy=policy, team_pool=team_pool, **kwargs)


class DecoderAgent(TeamAwareAgent):
    """Agent using the 200M parameter decoder-only transformer.

    Features:
    - Battle history buffer for sequence-based inference
    - KV cache for efficient multi-turn generation
    - Support for both deterministic and stochastic action selection
    """

    def __init__(
        self,
        policy: nn.Module,
        team_pool: TeamPool,
        action_space: ActionSpace = None,
        deterministic: bool = False,
        device: str = "cpu",
        max_history: int = 50,
        use_kv_cache: bool = True,
        **kwargs
    ):
        super().__init__(team_pool=team_pool, **kwargs)

        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.deterministic = deterministic
        self.max_history = max_history
        self.use_kv_cache = use_kv_cache

        self.action_space = action_space or ActionSpace()
        self.action_mask = ActionMask(self.action_space)
        self.preprocessor = FeaturePreprocessor()

        # Battle history buffer (stores preprocessed observations)
        self.history_buffer: deque = deque(maxlen=max_history)

        # KV cache for efficient inference
        self.kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        # Track current battle for cache invalidation
        self._current_battle_id: Optional[str] = None

    def _reset_for_new_battle(self, battle: AbstractBattle) -> None:
        """Reset history and cache for a new battle."""
        battle_id = battle.battle_tag
        if battle_id != self._current_battle_id:
            self.history_buffer.clear()
            self.kv_cache = None
            self._current_battle_id = battle_id

    def _build_observation_dict(self, battle: AbstractBattle) -> Dict[str, Any]:
        """Build observation dict matching training data format."""
        # Active Pokemon (own)
        active = battle.active_pokemon
        active_hp = 1.0
        active_status = 0

        if active:
            active_hp = active.current_hp_fraction if active.current_hp_fraction else 1.0
            active_status = STATUS_ENCODING.get(active.status.name if active.status else None, 0)

        # Own team state (6 Pokemon max)
        team_hp: List[float] = []
        team_status: List[int] = []
        team_fainted: List[bool] = []

        team_list = list(battle.team.values())
        for i in range(6):
            if i < len(team_list):
                mon = team_list[i]
                team_hp.append(mon.current_hp_fraction if mon.current_hp_fraction else 0.0)
                team_status.append(STATUS_ENCODING.get(mon.status.name if mon.status else None, 0))
                team_fainted.append(mon.fainted)
            else:
                team_hp.append(0.0)
                team_status.append(0)
                team_fainted.append(True)

        # Opponent state
        opp_active = battle.opponent_active_pokemon
        opp_active_hp = 1.0

        if opp_active:
            opp_active_hp = opp_active.current_hp_fraction if opp_active.current_hp_fraction else 1.0

        # Revealed opponent Pokemon
        opp_team_revealed: List[int] = [0] * 6
        opp_team_list = list(battle.opponent_team.values())
        for i, mon in enumerate(opp_team_list[:6]):
            opp_team_revealed[i] = 1

        # Weather
        weather_id = 0
        if battle.weather:
            for w_key in battle.weather.keys():
                weather_id = WEATHER_ENCODING.get(str(w_key).lower(), 0)
                break

        # Terrain
        terrain_id = 0
        if battle.fields:
            for f_key in battle.fields.keys():
                terrain_id = TERRAIN_ENCODING.get(str(f_key).lower(), 0)
                break

        # Hazards (own side)
        side = battle.side_conditions
        own_stealth_rock = any("stealthrock" in str(k).lower() for k in side)
        own_spikes = sum(1 for k in side if "spikes" in str(k).lower() and "toxic" not in str(k).lower())
        own_toxic_spikes = sum(1 for k in side if "toxicspikes" in str(k).lower())
        own_sticky_web = any("stickyweb" in str(k).lower() for k in side)

        # Hazards (opponent side)
        opp_side = battle.opponent_side_conditions
        opp_stealth_rock = any("stealthrock" in str(k).lower() for k in opp_side)
        opp_spikes = sum(1 for k in opp_side if "spikes" in str(k).lower() and "toxic" not in str(k).lower())
        opp_toxic_spikes = sum(1 for k in opp_side if "toxicspikes" in str(k).lower())
        opp_sticky_web = any("stickyweb" in str(k).lower() for k in opp_side)

        # Team ID
        team_id = 0
        if hasattr(self, '_obs_builder') and hasattr(self._obs_builder, '_team_id'):
            team_id = self._obs_builder._team_id or 0

        return {
            "turn": battle.turn,
            "active_species_id": 0,
            "active_hp": active_hp,
            "active_status": active_status,
            "team_hp": team_hp,
            "team_status": team_status,
            "team_fainted": team_fainted,
            "opp_active_species_id": 0,
            "opp_active_hp": opp_active_hp,
            "opp_team_revealed": opp_team_revealed,
            "weather_id": weather_id,
            "terrain_id": terrain_id,
            "own_stealth_rock": own_stealth_rock,
            "own_spikes": own_spikes,
            "own_toxic_spikes": own_toxic_spikes,
            "own_sticky_web": own_sticky_web,
            "opp_stealth_rock": opp_stealth_rock,
            "opp_spikes": opp_spikes,
            "opp_toxic_spikes": opp_toxic_spikes,
            "opp_sticky_web": opp_sticky_web,
            "team_id": team_id,
        }

    def _build_sequence_input(self, current_obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Build sequence input from history buffer and current observation.

        Returns tensors with shape [1, seq_len, ...] for the policy.
        """
        # Add current observation to history
        self.history_buffer.append(current_obs)

        # Stack all observations in history
        observations = list(self.history_buffer)
        seq_len = len(observations)

        # Stack each feature across the sequence
        stacked = {}
        for key in observations[0].keys():
            tensors = [obs[key] for obs in observations]
            stacked[key] = torch.stack(tensors, dim=0)  # [seq_len, ...]

        # Add batch dimension
        batched = {k: v.unsqueeze(0) for k, v in stacked.items()}  # [1, seq_len, ...]

        return batched

    def choose_move(self, battle: AbstractBattle):
        """Choose move using the decoder transformer with history context."""
        # Check if this is a new battle
        self._reset_for_new_battle(battle)

        with torch.no_grad():
            # Build current observation
            obs_dict = self._build_observation_dict(battle)
            obs_tensors = self.preprocessor.preprocess(obs_dict)
            obs_tensors = {k: v.to(self.device) for k, v in obs_tensors.items()}

            # Build sequence input with history
            seq_input = self._build_sequence_input(obs_tensors)

            # Get action mask for current turn
            mask = self.action_mask.get_mask_tensor(battle, self.device)
            mask = mask.unsqueeze(0)  # [1, num_actions]

            # Forward pass with optional KV cache
            if self.use_kv_cache and self.kv_cache is not None:
                # Use cached KV and only process new token
                action_probs, value, new_cache = self.policy(
                    seq_input,
                    action_mask=mask,
                    kv_cache=self.kv_cache,
                    use_cache=True,
                )
                self.kv_cache = new_cache
            else:
                # Full forward pass, optionally building cache
                action_probs, value, new_cache = self.policy(
                    seq_input,
                    action_mask=mask,
                    kv_cache=None,
                    use_cache=self.use_kv_cache,
                )
                if self.use_kv_cache:
                    self.kv_cache = new_cache

            # Select action
            if self.deterministic:
                action_idx = action_probs.argmax(dim=-1).item()
            else:
                action_idx = torch.multinomial(action_probs.squeeze(0), 1).item()

        return self._action_to_order(action_idx, battle)

    def _action_to_order(self, action_idx: int, battle: AbstractBattle):
        """Convert action index to poke-env order."""
        action_type, target_idx = self.action_space.decode_action(action_idx)

        if action_type == "move":
            if target_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[target_idx])
        else:
            if target_idx < len(battle.available_switches):
                return self.create_order(battle.available_switches[target_idx])

        # Fallback
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])

        return self.choose_default_move()

    def get_value_estimate(self, battle: AbstractBattle) -> float:
        """Get the model's value estimate for the current position.

        Returns a value in [-1, 1] where:
        - +1: Confident win
        - -1: Confident loss
        - 0: Uncertain/even
        """
        with torch.no_grad():
            obs_dict = self._build_observation_dict(battle)
            obs_tensors = self.preprocessor.preprocess(obs_dict)
            obs_tensors = {k: v.to(self.device) for k, v in obs_tensors.items()}

            # Build sequence (without adding to history)
            observations = list(self.history_buffer) + [obs_tensors]
            stacked = {}
            for key in observations[0].keys():
                tensors = [obs[key] for obs in observations]
                stacked[key] = torch.stack(tensors, dim=0).unsqueeze(0)

            _, value, _ = self.policy(stacked, action_mask=None, use_cache=False)
            return value.item()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        team_pool: TeamPool,
        config_size: str = "large",
        **kwargs
    ) -> "DecoderAgent":
        """Load decoder agent from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            team_pool: Team pool for team selection
            config_size: Model size ("small", "medium", "large")
            **kwargs: Additional arguments passed to agent
        """
        from ..models.config import SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG
        from ..models.decoder_policy import DecoderPolicy

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Select config
        if config_size == "small":
            config = SMALL_CONFIG
        elif config_size == "medium":
            config = MEDIUM_CONFIG
        else:
            config = LARGE_CONFIG

        # Create and load policy
        policy = DecoderPolicy(config)
        policy.load_state_dict(checkpoint["model_state_dict"])

        return cls(policy=policy, team_pool=team_pool, **kwargs)
