"""Utilities for inspecting battle state."""
from poke_env.battle import AbstractBattle, Pokemon

def summarize_battle(battle: AbstractBattle) -> dict:
    """Create a summary of the current battle state."""
    return {
        "turn": battle.turn,
        "player": battle.player_username,
        "active": battle.active_pokemon.species if battle.active_pokemon else None,
        "team_hp": [p.current_hp_fraction for p in battle.team.values()],
        "opponent_pokemon_seen": len(battle.opponent_team),
        "weather": str(battle.weather) if battle.weather else None,
        "fields": [str(f) for f in battle.fields],
        "available_moves": len(battle.available_moves),
        "available_switches": len(battle.available_switches),
    }

def format_pokemon_status(pokemon: Pokemon) -> str:
    """Format a Pokemon's status for logging."""
    status = f"{pokemon.species} ({pokemon.current_hp_fraction*100:.0f}%)"
    if pokemon.status:
        status += f" [{pokemon.status.name}]"
    return status
