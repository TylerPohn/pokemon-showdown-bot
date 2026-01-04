"""Preprocessing utilities for model input."""
import torch
import numpy as np
from typing import Dict, Any

class FeaturePreprocessor:
    """Convert raw observations to model-ready tensors."""

    def __init__(self, config: dict = None):
        self.config = config or {}

    def preprocess(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess a single observation.

        Args:
            observation: Raw observation dict

        Returns:
            Dict of tensors ready for model input
        """
        return {
            "team_id": self._to_tensor(observation.get("team_id", 0), dtype=torch.long),
            "weather": self._to_tensor(observation.get("weather_id", 0), dtype=torch.long),
            "terrain": self._to_tensor(observation.get("terrain_id", 0), dtype=torch.long),
            "pokemon_features": self._encode_pokemon(observation),
            "hazards": self._encode_hazards(observation),
        }

    def preprocess_batch(self, observations: list) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of observations."""
        processed = [self.preprocess(obs) for obs in observations]
        return {
            key: torch.stack([p[key] for p in processed])
            for key in processed[0].keys()
        }

    def _to_tensor(self, value, dtype=torch.float32) -> torch.Tensor:
        """Convert value to tensor."""
        if isinstance(value, torch.Tensor):
            return value.to(dtype)
        return torch.tensor(value, dtype=dtype)

    def _encode_pokemon(self, obs: dict) -> torch.Tensor:
        """Encode all Pokemon features."""
        features = []

        # Own team (6 Pokemon)
        for i in range(6):
            hp = obs.get(f"team_hp", [1.0] * 6)[i] if i < len(obs.get("team_hp", [])) else 0.0
            fainted = obs.get(f"team_fainted", [False] * 6)[i] if i < len(obs.get("team_fainted", [])) else False

            pokemon_feat = [
                hp,
                float(fainted),
                # Add more features as needed
            ]
            # Pad to fixed size
            pokemon_feat.extend([0.0] * (32 - len(pokemon_feat)))
            features.extend(pokemon_feat[:32])

        # Opponent team (6 Pokemon)
        for i in range(6):
            opp_hp = obs.get(f"opp_team_revealed", [0] * 6)
            revealed = i < len(opp_hp) and opp_hp[i] > 0

            pokemon_feat = [
                obs.get("opp_active_hp", 1.0) if i == 0 else 1.0,
                float(revealed),
            ]
            pokemon_feat.extend([0.0] * (32 - len(pokemon_feat)))
            features.extend(pokemon_feat[:32])

        return torch.tensor(features, dtype=torch.float32)

    def _encode_hazards(self, obs: dict) -> torch.Tensor:
        """Encode hazard state."""
        return torch.tensor([
            float(obs.get("own_stealth_rock", False)),
            float(obs.get("own_spikes", 0)) / 3,
            float(obs.get("own_toxic_spikes", 0)) / 2,
            float(obs.get("own_sticky_web", False)),
            float(obs.get("opp_stealth_rock", False)),
            float(obs.get("opp_spikes", 0)) / 3,
            float(obs.get("opp_toxic_spikes", 0)) / 2,
            float(obs.get("opp_sticky_web", False)),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Padding
        ], dtype=torch.float32)
