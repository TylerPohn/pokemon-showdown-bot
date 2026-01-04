#!/usr/bin/env python
"""Test BC agent in battle - quick smoke test for model integration."""
import argparse
import asyncio
import logging
from pathlib import Path

from poke_env.player import RandomPlayer

from poke.agents.nn_agent import NeuralNetworkAgent
from poke.agents.random_agent import PureRandomAgent
from poke.teams.loader import get_default_pool, TeamPool
from poke.config import BattleConfig


async def run_single_battle(agent, opponent, verbose: bool = False):
    """Run a single battle and return result."""
    await agent.battle_against(opponent, n_battles=1)

    # Get the last battle
    if agent.battles:
        battle_id = list(agent.battles.keys())[-1]
        battle = agent.battles[battle_id]
        won = battle.won
        if verbose:
            print(f"  Battle result: {'Won' if won else 'Lost'}")
            print(f"  Turns: {battle.turn}")
        return won
    return False


async def main():
    parser = argparse.ArgumentParser(description="Test BC agent integration")
    parser.add_argument(
        "--checkpoint",
        default="models/bc/final.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=5,
        help="Number of battles to run"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic action selection"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed battle info"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    config = BattleConfig()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return 1

    print(f"Loading BC agent from {checkpoint_path}...")

    # Load team pool
    try:
        pool = get_default_pool()
        print(f"Loaded {len(pool)} teams")
    except FileNotFoundError as e:
        print(f"Warning: Could not load team pool: {e}")
        print("Creating minimal team pool for testing...")
        # Create a minimal team for testing
        from poke.teams.models import Team
        from poke.teams.parser import TeamParser

        sample_team = """Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fighting
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Make It Rain
- Shadow Ball
- Focus Blast
- Nasty Plot

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Rapid Spin
- Knock Off
- Ice Spinner

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Dark
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance

Dragapult @ Choice Specs
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- U-turn
- Thunderbolt

Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Fairy
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Salt Cure
- Recover
- Stealth Rock
- Body Press

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
- Future Sight
- Sludge Bomb
- Slack Off
- Thunder Wave
"""
        parser = TeamParser()
        team = parser.parse_team(sample_team, name="test_team")
        team.format = "gen9ou"
        pool = TeamPool([team])
        print(f"Created test pool with {len(pool)} team(s)")

    # Create BC agent
    try:
        bc_agent = NeuralNetworkAgent.from_checkpoint(
            str(checkpoint_path),
            team_pool=pool,
            deterministic=args.deterministic,
            battle_format=config.battle_format,
            server_configuration=config.server_configuration,
        )
        print("BC agent loaded successfully!")
    except Exception as e:
        print(f"Error loading BC agent: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create opponent - use RandomAgent with same team pool
    from poke.agents.random_agent import RandomAgent
    opponent = RandomAgent(
        team_pool=pool,
        seed=42,
        battle_format=config.battle_format,
        server_configuration=config.server_configuration,
    )

    print(f"\nRunning {args.battles} battles vs RandomAgent...")
    print("-" * 40)

    wins = 0
    for i in range(args.battles):
        print(f"Battle {i+1}/{args.battles}...", end=" ", flush=True)
        try:
            won = await run_single_battle(bc_agent, opponent, verbose=args.verbose)
            wins += int(won)
            print("Won" if won else "Lost")
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("-" * 40)
    print(f"\nResults:")
    print(f"  Wins: {wins}/{args.battles} ({100*wins/args.battles:.1f}%)")
    print(f"  Expected vs Random: >80% for a well-trained model")

    # Quick sanity check
    if wins >= args.battles * 0.6:
        print("\n[OK] BC agent appears to be working correctly!")
        return 0
    else:
        print("\n[WARNING] BC agent winrate seems low. May need debugging.")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
