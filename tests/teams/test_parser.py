"""Tests for team parser."""
import pytest
from poke.teams.parser import TeamParser

SAMPLE_POKEMON = """
Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Hidden Power Ice
"""

SAMPLE_TEAM = """
Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Hidden Power Ice

Charizard @ Heavy-Duty Boots
Ability: Solar Power
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Flamethrower
- Dragon Pulse
- Roost
- Focus Blast
"""

def test_parse_pokemon():
    parser = TeamParser()
    mon = parser.parse_pokemon(SAMPLE_POKEMON)

    assert mon.species == "Pikachu"
    assert mon.item == "Light Ball"
    assert mon.ability == "Static"
    assert mon.nature == "Timid"
    assert len(mon.moves) == 4
    assert "Thunderbolt" in mon.moves

def test_parse_evs():
    parser = TeamParser()
    mon = parser.parse_pokemon(SAMPLE_POKEMON)

    assert mon.evs.get("SpA") == 252
    assert mon.evs.get("Spe") == 252
    assert mon.evs.get("SpD") == 4

def test_parse_team():
    parser = TeamParser()
    team = parser.parse_team(SAMPLE_TEAM, name="Test Team")

    assert len(team) == 2
    assert team.pokemon[0].species == "Pikachu"
    assert team.pokemon[1].species == "Charizard"

def test_team_id_generation():
    parser = TeamParser()
    team1 = parser.parse_team(SAMPLE_TEAM)
    team2 = parser.parse_team(SAMPLE_TEAM)

    # Same content should produce same ID
    assert team1.team_id == team2.team_id

def test_roundtrip():
    parser = TeamParser()
    team = parser.parse_team(SAMPLE_TEAM)
    output = team.to_showdown()

    # Parse the output again
    team2 = parser.parse_team(output)
    assert len(team2) == len(team)
