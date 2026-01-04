"""Pytest configuration and shared fixtures."""

import pytest

from poke.utils.showdown_server import ShowdownServer


@pytest.fixture
def sample_team():
    """Return a sample Pokemon team for testing."""
    return """
Pikachu @ Light Ball
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Volt Switch
- Grass Knot
- Hidden Power Ice
"""


@pytest.fixture(scope="session")
def showdown_server():
    """Provide a running Showdown server for tests."""
    server = ShowdownServer()
    server.start()
    yield server
    server.stop()
