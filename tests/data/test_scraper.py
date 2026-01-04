"""Tests for replay scraper."""
import pytest
from unittest.mock import Mock, patch

from poke.data.scraper import ReplayScraper, ScraperConfig

def test_config_defaults():
    config = ScraperConfig()
    assert config.format == "gen9ou"
    assert config.requests_per_second == 1.0

@patch("poke.data.scraper.requests.Session")
def test_search_replays(mock_session):
    mock_response = Mock()
    mock_response.json.return_value = [{"id": "gen9ou-123"}]
    mock_session.return_value.get.return_value = mock_response

    config = ScraperConfig()
    scraper = ReplayScraper(config)
    results = scraper.search_replays(page=1)

    assert len(results) == 1
    assert results[0]["id"] == "gen9ou-123"

def test_rate_limiting():
    """Verify rate limiting doesn't exceed configured rate."""
    import time
    config = ScraperConfig(requests_per_second=10.0)  # Fast for testing
    scraper = ReplayScraper(config)

    start = time.time()
    for _ in range(5):
        scraper._rate_limit()
    elapsed = time.time() - start

    # Should take at least 0.4 seconds (4 gaps at 0.1s each)
    assert elapsed >= 0.4
