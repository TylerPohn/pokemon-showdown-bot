#!/usr/bin/env python
"""Demo battle you can watch in browser."""
import asyncio
from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

async def main():
    print("\n" + "="*60)
    print("   POKEMON SHOWDOWN BATTLE DEMO")
    print("="*60)
    print("\n1. Open your browser to: http://localhost:8000")
    print("2. Look for 'Battles' in the right sidebar")
    print("3. Click on the battle to watch it live!")
    print("\n" + "-"*60)

    # Create players with names you can find in the lobby
    smart_player = SimpleHeuristicsPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    random_player = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    print(f"\nBattle: {smart_player.username} vs {random_player.username}")
    print("Starting in 3 seconds... (open browser now!)\n")
    await asyncio.sleep(3)

    # Run 3 battles
    print("Running 3 battles...\n")
    await smart_player.battle_against(random_player, n_battles=3)

    # Results
    print("\n" + "="*60)
    print("   RESULTS")
    print("="*60)
    print(f"  SmartHeuristics: {smart_player.n_won_battles} wins")
    print(f"  RandomPlayer:    {random_player.n_won_battles} wins")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
