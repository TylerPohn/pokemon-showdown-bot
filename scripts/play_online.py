#!/usr/bin/env python
"""Play on the public Pokemon Showdown server with BC agent."""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from poke_env import AccountConfiguration, ShowdownServerConfiguration

from poke.agents.nn_agent import NeuralNetworkAgent
from poke.teams.loader import get_default_pool


def load_credentials():
    """Load credentials from environment."""
    load_dotenv()
    username = os.getenv("SHOWDOWN_USERNAME")
    password = os.getenv("SHOWDOWN_PASSWORD")

    if not username or not password:
        raise ValueError(
            "SHOWDOWN_USERNAME and SHOWDOWN_PASSWORD must be set in .env file"
        )

    return username, password


async def main():
    print("\n" + "="*60)
    print("   BC AGENT ON POKEMON SHOWDOWN")
    print("="*60)

    # Load credentials
    username, password = load_credentials()
    print(f"\nLogging in as: {username}")

    # Load team pool
    pool = get_default_pool()
    print(f"Loaded {len(pool)} teams")

    # Load BC agent
    checkpoint_path = Path("models/bc/final.pt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading BC model from {checkpoint_path}...")

    player = NeuralNetworkAgent.from_checkpoint(
        str(checkpoint_path),
        team_pool=pool,
        deterministic=False,  # Use stochastic for variety
        account_configuration=AccountConfiguration(username, password),
        server_configuration=ShowdownServerConfiguration,
        battle_format="gen9ou",
        max_concurrent_battles=1,
    )

    print("\nBC Agent loaded and connected!")
    print("-"*60)
    print("\nTo battle this agent:")
    print(f"  1. Open https://play.pokemonshowdown.com")
    print(f"  2. Log in with a DIFFERENT account")
    print(f"  3. Click 'Find a user' and search for: {username}")
    print(f"  4. Click 'Challenge' -> select '[Gen 9] OU' -> Challenge!")
    print("\nOr the agent will search for ladder battles.")
    print("\nPress Ctrl+C to stop.\n")
    print("-"*60)

    # Accept challenges
    print(f"\n[{username}] Waiting for challenges...")
    await player.accept_challenges(None, n_challenges=10)

    print(f"\nCompleted {player.n_finished_battles} battles")
    print(f"Wins: {player.n_won_battles}")
    if player.n_finished_battles > 0:
        print(f"Win rate: {player.win_rate:.1%}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped.")
