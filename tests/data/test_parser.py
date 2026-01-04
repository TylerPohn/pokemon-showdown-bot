"""Tests for replay parser."""
import pytest
from poke.data.parser import BattleLogParser
from poke.data.models import ActionType

SAMPLE_LOG = """
|j|Player1
|j|Player2
|player|p1|Player1|1
|player|p2|Player2|2
|turn|1
|switch|p1a: Pikachu|Pikachu, L50
|switch|p2a: Charizard|Charizard, L50
|move|p1a: Pikachu|Thunderbolt|p2a: Charizard
|-damage|p2a: Charizard|75/100
|turn|2
|move|p2a: Charizard|Flamethrower|p1a: Pikachu
|-damage|p1a: Pikachu|0/100
|faint|p1a: Pikachu
|win|Player2
"""

def test_parse_basic_battle():
    parser = BattleLogParser()
    replay = {
        "id": "test-123",
        "format": "gen9ou",
        "p1": "Player1",
        "p2": "Player2",
        "log": SAMPLE_LOG,
    }

    result = parser.parse(replay)

    assert result is not None
    assert result.replay_id == "test-123"
    assert result.winner == "Player2"
    assert len(result.turns) >= 1

def test_parse_switch_action():
    parser = BattleLogParser()
    # Create a log where a mid-battle switch happens
    switch_log = """
|turn|1
|switch|p1a: Pikachu|Pikachu, L50
|switch|p2a: Charizard|Charizard, L50
|move|p1a: Pikachu|Thunderbolt|p2a: Charizard
|-damage|p2a: Charizard|50/100
|turn|2
|switch|p1a: Raichu|Raichu, L50
|move|p2a: Charizard|Flamethrower|p1a: Raichu
|turn|3
|win|Player1
"""
    replay = {"id": "test", "log": switch_log}

    result = parser.parse(replay)

    # Second turn should have switch action for p1
    turn2 = result.turns[1]
    assert turn2.p1_action.action_type == ActionType.SWITCH
    assert turn2.p1_action.target == "Raichu"

def test_parse_move_action():
    parser = BattleLogParser()
    replay = {"id": "test", "log": SAMPLE_LOG}

    result = parser.parse(replay)

    # Second turn should have move actions
    if len(result.turns) > 1:
        turn2 = result.turns[1]
        assert turn2.p1_action.action_type == ActionType.MOVE
        assert turn2.p1_action.target == "Thunderbolt"
