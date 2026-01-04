#!/usr/bin/env python
"""Evaluate trained agents."""
import argparse
import asyncio
import logging
from pathlib import Path

from poke.agents.random_agent import PureRandomAgent
from poke.agents.heuristic_agent import MaxDamageAgent
from poke.agents.nn_agent import NeuralNetworkAgent
from poke.teams.loader import get_default_pool
from poke.evaluation.runner import BattleRunner, EvaluationReport
from poke.evaluation.metrics import compute_elo_ratings, compute_agent_metrics
from poke.config import BattleConfig

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--battles", type=int, default=100, help="Battles per matchup")
    parser.add_argument("--output", default="evaluation_report.json")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = BattleConfig()
    pool = get_default_pool()

    # Create agents
    agents = [
        PureRandomAgent(
            battle_format=config.battle_format,
            server_configuration=config.server_configuration,
        ),
        MaxDamageAgent(
            team_pool=pool,
            battle_format=config.battle_format,
            server_configuration=config.server_configuration,
        ),
        NeuralNetworkAgent.from_checkpoint(
            args.checkpoint,
            team_pool=pool,
            deterministic=args.deterministic,
            battle_format=config.battle_format,
            server_configuration=config.server_configuration,
        ),
    ]

    # Run tournament
    runner = BattleRunner(battle_format=config.battle_format)
    results = await runner.run_tournament(agents, n_battles_per_matchup=args.battles)

    # Create report
    report = EvaluationReport.from_results(results)

    # Compute Elo
    elo_ratings = compute_elo_ratings([m.to_dict() for m in report.matchups])
    agent_metrics = compute_agent_metrics(report.agent_stats, elo_ratings)

    # Print summary
    print("\n=== Evaluation Results ===\n")
    for name, metrics in sorted(agent_metrics.items(), key=lambda x: -x[1].elo_rating):
        print(f"{name}:")
        print(f"  Winrate: {metrics.winrate:.1%} ({metrics.winrate_ci_low:.1%}-{metrics.winrate_ci_high:.1%})")
        print(f"  Elo: {metrics.elo_rating:.0f}")
        print(f"  Record: {metrics.total_wins}W-{metrics.total_losses}L")
        print()

    # Save report
    report.save(Path(args.output))
    print(f"Report saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
