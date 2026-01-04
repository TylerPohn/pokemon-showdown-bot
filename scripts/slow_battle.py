#!/usr/bin/env python
"""Slow battle demo you can watch in browser."""
import asyncio
from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle


class SlowPlayer(SimpleHeuristicsPlayer):
    """Player that waits before each move so you can watch."""

    async def _handle_battle_request(self, battle: AbstractBattle, from_teampreview_request: bool = False):
        # Wait 2 seconds before each move
        await asyncio.sleep(2)
        return await super()._handle_battle_request(battle, from_teampreview_request)


class SlowRandom(RandomPlayer):
    """Random player that waits before each move."""

    async def _handle_battle_request(self, battle: AbstractBattle, from_teampreview_request: bool = False):
        await asyncio.sleep(2)
        return await super()._handle_battle_request(battle, from_teampreview_request)


async def main():
    print("\n" + "="*60)
    print("   SLOW BATTLE DEMO - WATCH IN BROWSER!")
    print("="*60)
    print("\n>>> Open http://localhost:8000 NOW <<<")
    print("\nLook for the battle in the right sidebar under 'Battles'")
    print("Each move takes 2 seconds so you can follow along!\n")
    print("-"*60)

    smart = SlowPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    random_p = SlowRandom(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1,
    )

    print(f"\nStarting: {smart.username} vs {random_p.username}")
    print("Battle will appear in ~3 seconds...\n")

    await smart.battle_against(random_p, n_battles=1)

    print("\n" + "="*60)
    print(f"  Winner: {'Smart AI' if smart.n_won_battles > 0 else 'Random'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
