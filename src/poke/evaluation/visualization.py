"""Visualization utilities for evaluation metrics."""
from typing import Dict, List, Optional
from pathlib import Path

def create_winrate_chart(
    agent_names: List[str],
    winrates: List[float],
    ci_lows: List[float],
    ci_highs: List[float],
    output_path: Optional[Path] = None,
) -> None:
    """Create bar chart of winrates with confidence intervals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(agent_names))
    yerr = [[w - l for w, l in zip(winrates, ci_lows)],
            [h - w for w, h in zip(winrates, ci_highs)]]

    bars = ax.bar(x, winrates, yerr=yerr, capsize=5, color='steelblue', alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='50% baseline')

    ax.set_ylabel('Winrate')
    ax.set_title('Agent Winrates vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def create_action_distribution_pie(
    action_dist: Dict[str, float],
    output_path: Optional[Path] = None,
) -> None:
    """Create pie chart of action distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    labels = list(action_dist.keys())
    sizes = list(action_dist.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title('Action Distribution')

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def create_elo_progression(
    agents: List[str],
    elo_history: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> None:
    """Create line chart of Elo rating progression."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for agent in agents:
        if agent in elo_history:
            ax.plot(elo_history[agent], label=agent)

    ax.set_xlabel('Games')
    ax.set_ylabel('Elo Rating')
    ax.set_title('Elo Rating Progression')
    ax.legend()
    ax.axhline(y=1500, color='gray', linestyle='--', alpha=0.5)

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()
