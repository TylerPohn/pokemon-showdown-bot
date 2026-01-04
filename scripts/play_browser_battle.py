#!/usr/bin/env python
"""Run a battle between trained agent and random agent for browser viewing."""
import asyncio
import torch
from pathlib import Path

from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer

from poke.models.config import EncoderConfig
from poke.models.factory import create_policy
from poke.models.action_space import ActionSpace
from poke.agents.base import BaseAgent
from poke_env.battle import AbstractBattle


class TrainedAgent(BaseAgent):
    """Agent using trained BC policy."""

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(**kwargs)

        # Load model
        self.device = "cpu"
        config = EncoderConfig()
        self.policy = create_policy("mlp", encoder_config=config)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()

        self.action_space = ActionSpace()

    def choose_move(self, battle: AbstractBattle):
        """Choose move using trained policy."""
        # For simplicity, use a basic observation
        # In production, would use full preprocessing

        with torch.no_grad():
            # Create simple state features
            features = self._extract_features(battle)
            state = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            # Get action mask
            mask = self._get_action_mask(battle)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

            # Get policy output
            action_probs, _ = self.policy({"state": state}, mask_tensor)
            action_idx = action_probs.argmax(dim=-1).item()

        return self._action_to_order(action_idx, battle)

    def _extract_features(self, battle: AbstractBattle):
        """Extract simple features from battle state."""
        features = []

        # Active pokemon HP
        if battle.active_pokemon:
            features.append(battle.active_pokemon.current_hp_fraction)
        else:
            features.append(0.0)

        # Opponent HP
        if battle.opponent_active_pokemon:
            features.append(battle.opponent_active_pokemon.current_hp_fraction)
        else:
            features.append(1.0)

        # Team HP
        for mon in list(battle.team.values())[:6]:
            features.append(mon.current_hp_fraction)
        while len(features) < 8:
            features.append(0.0)

        # Pad to expected size (simplified)
        while len(features) < 128:
            features.append(0.0)

        return features[:128]

    def _get_action_mask(self, battle: AbstractBattle):
        """Get action mask for legal moves."""
        mask = [False] * self.action_space.total_actions

        for i, move in enumerate(battle.available_moves):
            if i < 4:
                mask[i] = True

        for i, pokemon in enumerate(battle.available_switches):
            if i < 5:
                mask[4 + i] = True

        # Ensure at least one action is legal
        if not any(mask):
            mask[0] = True

        return mask

    def _action_to_order(self, action_idx: int, battle: AbstractBattle):
        """Convert action index to battle order."""
        if action_idx < 4:
            # Move
            if action_idx < len(battle.available_moves):
                return self.create_order(battle.available_moves[action_idx])
        else:
            # Switch
            switch_idx = action_idx - 4
            if switch_idx < len(battle.available_switches):
                return self.create_order(battle.available_switches[switch_idx])

        # Fallback
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_default_move()


async def main():
    print("\n" + "="*60)
    print("POKEMON SHOWDOWN BATTLE - TRAINED AI vs RANDOM")
    print("="*60)
    print("\nOpen your browser to: http://localhost:8000")
    print("Click 'Watch a battle' to see the match!\n")

    # Create players
    trained = TrainedAgent(
        checkpoint_path="models/bc/final.pt",
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    random_player = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    print("Starting battle: TrainedAgent vs RandomPlayer...")
    print("(This may take a moment to connect)\n")

    # Run battle
    await trained.battle_against(random_player, n_battles=1)

    # Results
    print("\n" + "="*60)
    print("BATTLE COMPLETE!")
    print("="*60)
    print(f"TrainedAgent wins: {trained.n_won_battles}")
    print(f"RandomPlayer wins: {random_player.n_won_battles}")
    print(f"Winner: {'TrainedAgent' if trained.n_won_battles > 0 else 'RandomPlayer'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
