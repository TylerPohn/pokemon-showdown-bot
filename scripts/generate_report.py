#!/usr/bin/env python
"""Generate evaluation report from battle logs."""
import argparse
from pathlib import Path
import json

from poke.evaluation.aggregator import MetricsAggregator
from poke.evaluation.visualization import (
    create_winrate_chart,
    create_action_distribution_pie,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True, help="Directory with battle logs")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--agent", required=True, help="Agent name to analyze")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and aggregate metrics
    aggregator = MetricsAggregator()
    aggregator.load_from_directory(logs_dir)

    metrics = aggregator.compute_metrics(args.agent)

    # Print summary
    print(f"\n=== Metrics for {args.agent} ===\n")
    print(f"Total battles: {metrics.total_battles}")
    print(f"Average game length: {metrics.avg_game_length:.1f} turns")
    print(f"Switch frequency: {metrics.switch_frequency:.1%}")
    print(f"\nAction distribution:")
    for action, pct in metrics.action_distribution.items():
        print(f"  {action}: {pct:.1%}")

    print(f"\nWinrate by game length:")
    for length, wr in metrics.win_by_turns.items():
        print(f"  {length}: {wr:.1%}")

    # Top moves
    print(f"\nTop 10 moves:")
    for move, count in aggregator.get_move_ranking(10):
        print(f"  {move}: {count}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "total_battles": metrics.total_battles,
            "avg_game_length": metrics.avg_game_length,
            "action_distribution": metrics.action_distribution,
            "switch_frequency": metrics.switch_frequency,
            "win_by_turns": metrics.win_by_turns,
            "top_moves": aggregator.get_move_ranking(20),
        }, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")

    # Generate visualizations
    create_action_distribution_pie(
        metrics.action_distribution,
        output_dir / "action_distribution.png"
    )

if __name__ == "__main__":
    main()
