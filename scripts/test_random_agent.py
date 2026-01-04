#!/usr/bin/env python
"""Test random agent in battle."""
import argparse
import asyncio

from poke_env.player import RandomPlayer

from poke.agents.random_agent import RandomAgent, PureRandomAgent
from poke.teams.loader import get_default_pool
from poke.config import BattleConfig

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = BattleConfig()

    # Our random agent
    agent = PureRandomAgent(
        seed=args.seed,
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )

    # Opponent (poke-env's random player)
    opponent = RandomPlayer(
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )

    print(f"Running {args.battles} battles...")
    await agent.battle_against(opponent, n_battles=args.battles)

    wins = agent.n_won_battles
    print(f"\nResults:")
    print(f"  Wins: {wins}/{args.battles} ({100*wins/args.battles:.1f}%)")
    print(f"  Expected: ~50% (random vs random)")

if __name__ == "__main__":
    asyncio.run(main())
