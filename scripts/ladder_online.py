#!/usr/bin/env python
"""Climb the ladder on public Pokemon Showdown."""
import asyncio
import getpass
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer


async def main():
    print("\n" + "="*60)
    print("   LADDER ON POKEMON SHOWDOWN")
    print("="*60)

    print("\nThis will queue for RANKED matches on the public ladder!")
    print("Your bot will play against real players.\n")

    username = input("Showdown username: ").strip()
    password = getpass.getpass("Password: ")

    n_games = input("How many games to play? [5]: ").strip()
    n_games = int(n_games) if n_games else 5

    print(f"\nConnecting as '{username}'...")
    print(f"Will play {n_games} ladder games.")
    print("\nYou can watch at: https://play.pokemonshowdown.com")
    print(f"Search for user '{username}' to spectate!\n")
    print("-"*60)

    player = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(username, password),
        server_configuration=ShowdownServerConfiguration,
        battle_format="gen9randombattle",
        max_concurrent_battles=1,
    )

    print(f"\n[{username}] Starting ladder climb...")

    # Play ladder games
    await player.ladder(n_games)

    print("\n" + "="*60)
    print("   RESULTS")
    print("="*60)
    print(f"  Games played: {player.n_finished_battles}")
    print(f"  Wins: {player.n_won_battles}")
    print(f"  Losses: {player.n_lost_battles}")
    print(f"  Win rate: {player.win_rate:.1%}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped.")
