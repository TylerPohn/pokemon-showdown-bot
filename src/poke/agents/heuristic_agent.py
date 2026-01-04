"""Heuristic-based baseline agents."""
from typing import Optional, List, Tuple

from poke_env.battle import AbstractBattle, Move, Pokemon, MoveCategory

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
