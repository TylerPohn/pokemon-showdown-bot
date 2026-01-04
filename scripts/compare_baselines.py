#!/usr/bin/env python
"""Compare baseline agents."""
import argparse
import asyncio
from collections import defaultdict

from poke.agents.random_agent import PureRandomAgent
from poke.agents.heuristic_agent import MaxDamageAgent
from poke.teams.loader import get_default_pool
from poke.config import BattleConfig

async def run_matchup(agent1, agent2, n_battles: int) -> dict:
    """Run battles between two agents."""
    await agent1.battle_against(agent2, n_battles=n_battles)
    return {
        "agent1_wins": agent1.n_won_battles,
        "agent2_wins": agent2.n_won_battles,
        "total": n_battles,
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=100)
    args = parser.parse_args()

    config = BattleConfig()
    pool = get_default_pool()

    results = {}

    # Random vs Random
    print("Random vs Random...")
    r1 = PureRandomAgent(seed=42, battle_format=config.battle_format)
    r2 = PureRandomAgent(seed=123, battle_format=config.battle_format)
    results["random_vs_random"] = await run_matchup(r1, r2, args.battles)

    # MaxDamage vs Random
    print("MaxDamage vs Random...")
    md = MaxDamageAgent(team_pool=pool, battle_format=config.battle_format)
    r3 = PureRandomAgent(seed=456, battle_format=config.battle_format)
    results["maxdamage_vs_random"] = await run_matchup(md, r3, args.battles)

    # MaxDamage vs MaxDamage
    print("MaxDamage vs MaxDamage...")
    md1 = MaxDamageAgent(team_pool=pool, battle_format=config.battle_format)
    md2 = MaxDamageAgent(team_pool=pool, battle_format=config.battle_format)
    results["maxdamage_vs_maxdamage"] = await run_matchup(md1, md2, args.battles)

    print("\n=== Results ===")
    for name, result in results.items():
        wr = result["agent1_wins"] / result["total"] * 100
        print(f"{name}: {result['agent1_wins']}/{result['total']} ({wr:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
