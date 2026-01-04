COMPLETED

# PR-012: Max Damage Heuristic Agent

## Dependencies
- PR-003 (poke-env Integration)
- PR-010 (TeamID Observation Integration)

## Overview
Implement a heuristic agent that selects the move expected to deal the most damage. This provides a stronger baseline than random.

## Tech Choices
- **Damage Calculation:** poke-env built-in damage estimation
- **Switch Logic:** Switch when out of attacking moves or likely KO'd
- **Priority:** Moves > Switches (simple heuristic)

## Tasks

### 1. Implement max damage agent
Create `src/poke/agents/heuristic_agent.py`:
```python
"""Heuristic-based baseline agents."""
from typing import Optional, List, Tuple

from poke_env.environment import AbstractBattle, Move, Pokemon
from poke_env.environment.move_category import MoveCategory

from .team_aware import TeamAwareAgent
from ..teams.loader import TeamPool

class MaxDamageAgent(TeamAwareAgent):
    """Agent that selects the highest damage move.

    Simple heuristic:
    1. Pick move with highest expected damage
    2. Switch if no damaging moves or active is at low HP
    """

    def __init__(
        self,
        team_pool: TeamPool,
        switch_threshold: float = 0.25,  # Switch if HP below this
        **kwargs
    ):
        super().__init__(team_pool=team_pool, **kwargs)
        self.switch_threshold = switch_threshold

    def choose_move(self, battle: AbstractBattle):
        """Choose the highest damage move available.

        Args:
            battle: Current battle state

        Returns:
            BattleOrder for highest damage action
        """
        # Check if we should switch (low HP, no good moves)
        if self._should_switch(battle):
            switch = self._best_switch(battle)
            if switch:
                return self.create_order(switch)

        # Find best damaging move
        best_move = self._best_move(battle)
        if best_move:
            return self.create_order(best_move)

        # Fall back to any available move
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])

        # Fall back to switch
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])

        return self.choose_default_move()

    def _should_switch(self, battle: AbstractBattle) -> bool:
        """Determine if we should switch out."""
        active = battle.active_pokemon
        if not active:
            return False

        # Low HP
        if active.current_hp_fraction < self.switch_threshold:
            return True

        # No damaging moves left
        has_damaging = any(
            move.base_power > 0
            for move in battle.available_moves
        )
        if not has_damaging and battle.available_switches:
            return True

        return False

    def _best_move(self, battle: AbstractBattle) -> Optional[Move]:
        """Find the move with highest expected damage."""
        if not battle.available_moves:
            return None

        opponent = battle.opponent_active_pokemon
        if not opponent:
            # Pick highest base power if no opponent info
            return max(
                battle.available_moves,
                key=lambda m: m.base_power or 0
            )

        # Score each move
        scored_moves: List[Tuple[float, Move]] = []
        for move in battle.available_moves:
            score = self._score_move(move, battle.active_pokemon, opponent, battle)
            scored_moves.append((score, move))

        if not scored_moves:
            return None

        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return scored_moves[0][1]

    def _score_move(
        self,
        move: Move,
        user: Pokemon,
        target: Pokemon,
        battle: AbstractBattle
    ) -> float:
        """Score a move based on expected damage.

        Uses simplified damage estimation.
        """
        if move.base_power == 0:
            # Status moves get low priority
            return 0.1

        # Type effectiveness
        type_mult = target.damage_multiplier(move)

        # STAB bonus
        stab = 1.5 if move.type in user.types else 1.0

        # Base damage estimate (simplified)
        if move.category == MoveCategory.PHYSICAL:
            attack_stat = user.stats.get("atk", 100)
            defense_stat = target.stats.get("def", 100) if target.stats else 100
        else:
            attack_stat = user.stats.get("spa", 100)
            defense_stat = target.stats.get("spd", 100) if target.stats else 100

        # Avoid division by zero
        defense_stat = max(defense_stat, 1)

        damage = (move.base_power * type_mult * stab * attack_stat) / defense_stat

        # Priority bonus
        if move.priority > 0:
            damage *= 1.1

        # Accuracy penalty
        accuracy = move.accuracy / 100 if move.accuracy else 1.0
        damage *= accuracy

        return damage

    def _best_switch(self, battle: AbstractBattle) -> Optional[Pokemon]:
        """Find the best Pokemon to switch to."""
        if not battle.available_switches:
            return None

        opponent = battle.opponent_active_pokemon
        if not opponent:
            # Pick healthiest
            return max(
                battle.available_switches,
                key=lambda p: p.current_hp_fraction
            )

        # Score each switch option
        scored_switches: List[Tuple[float, Pokemon]] = []
        for pokemon in battle.available_switches:
            score = self._score_switch(pokemon, opponent)
            scored_switches.append((score, pokemon))

        scored_switches.sort(reverse=True, key=lambda x: x[0])
        return scored_switches[0][1]

    def _score_switch(self, pokemon: Pokemon, opponent: Pokemon) -> float:
        """Score a switch option."""
        score = pokemon.current_hp_fraction * 100

        # Type advantage bonus
        for move_id in pokemon.moves:
            move = pokemon.moves[move_id]
            type_mult = opponent.damage_multiplier(move)
            if type_mult > 1:
                score += 20

        # Resistance bonus (simplified)
        for opp_type in opponent.types:
            if opp_type:
                # This is a simplification
                score += 5

        return score

    def teampreview(self, battle: AbstractBattle):
        """Use default team order."""
        return "/team 123456"


class SmartHeuristicAgent(MaxDamageAgent):
    """Enhanced heuristic agent with better decision making."""

    def _should_switch(self, battle: AbstractBattle) -> bool:
        """More nuanced switch decision."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if not active:
            return False

        # Check for guaranteed KO from opponent
        if opponent and active.current_hp_fraction < 0.3:
            # Estimate if we're faster and can KO
            if not self._can_ko(active, opponent, battle):
                return True

        return super()._should_switch(battle)

    def _can_ko(
        self,
        attacker: Pokemon,
        defender: Pokemon,
        battle: AbstractBattle
    ) -> bool:
        """Estimate if we can KO the opponent."""
        for move in battle.available_moves:
            if move.base_power > 0:
                type_mult = defender.damage_multiplier(move)
                if type_mult >= 2 and defender.current_hp_fraction < 0.5:
                    return True
        return False
```

### 2. Create comparison script
Create `scripts/compare_baselines.py`:
```python
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
```

### 3. Write unit tests
Create `tests/agents/test_heuristic_agent.py`:
```python
"""Tests for heuristic agent."""
import pytest
from unittest.mock import Mock, MagicMock

from poke.agents.heuristic_agent import MaxDamageAgent

@pytest.fixture
def mock_battle():
    battle = Mock()

    # Active Pokemon
    active = Mock()
    active.current_hp_fraction = 1.0
    active.types = ("Electric",)
    active.stats = {"atk": 100, "spa": 120}
    battle.active_pokemon = active

    # Opponent
    opponent = Mock()
    opponent.current_hp_fraction = 0.8
    opponent.types = ("Water",)
    opponent.stats = {"def": 80, "spd": 90}
    opponent.damage_multiplier = lambda m: 2.0 if m.type == "Electric" else 1.0
    battle.opponent_active_pokemon = opponent

    # Moves
    thunderbolt = Mock()
    thunderbolt.id = "thunderbolt"
    thunderbolt.base_power = 90
    thunderbolt.type = "Electric"
    thunderbolt.category = Mock()
    thunderbolt.category.name = "SPECIAL"
    thunderbolt.priority = 0
    thunderbolt.accuracy = 100

    tackle = Mock()
    tackle.id = "tackle"
    tackle.base_power = 40
    tackle.type = "Normal"
    tackle.category = Mock()
    tackle.category.name = "PHYSICAL"
    tackle.priority = 0
    tackle.accuracy = 100

    battle.available_moves = [thunderbolt, tackle]
    battle.available_switches = []

    return battle

def test_picks_higher_damage_move(mock_battle):
    # Create a minimal pool
    from poke.teams.loader import TeamPool
    pool = Mock(spec=TeamPool)
    pool.sample = Mock(return_value=Mock())
    pool.get_ids = Mock(return_value=["team1"])

    agent = MaxDamageAgent.__new__(MaxDamageAgent)
    agent.switch_threshold = 0.25
    agent.create_order = lambda x: x

    move = agent._best_move(mock_battle)

    assert move.id == "thunderbolt"

def test_should_switch_at_low_hp(mock_battle):
    mock_battle.active_pokemon.current_hp_fraction = 0.1

    switch_target = Mock()
    switch_target.current_hp_fraction = 0.8
    mock_battle.available_switches = [switch_target]

    agent = MaxDamageAgent.__new__(MaxDamageAgent)
    agent.switch_threshold = 0.25

    assert agent._should_switch(mock_battle) is True
```

### 4. Add to agent registry
Update `src/poke/agents/__init__.py`:
```python
"""Agent implementations."""
from .base import BaseAgent
from .random_agent import RandomAgent, PureRandomAgent
from .heuristic_agent import MaxDamageAgent, SmartHeuristicAgent

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "PureRandomAgent",
    "MaxDamageAgent",
    "SmartHeuristicAgent",
]

BASELINE_AGENTS = {
    "random": PureRandomAgent,
    "random_team": RandomAgent,
    "maxdamage": MaxDamageAgent,
    "smart": SmartHeuristicAgent,
}
```

## Acceptance Criteria
- [ ] Agent picks highest damage move correctly
- [ ] Type effectiveness considered in damage calculation
- [ ] Switches when at low HP or no damaging moves
- [ ] Significantly beats random agent (>60% winrate)
- [ ] Handles edge cases (no moves, no switches)

## Expected Performance
- vs Random: ~70-80% winrate
- vs Itself: ~50% winrate

## Notes
- This is still a simple heuristic, not optimal play
- Doesn't consider future turns, setup moves, or prediction
- Good enough as a baseline for RL agent comparison

## Estimated Complexity
Medium - Damage calculation with type effectiveness
