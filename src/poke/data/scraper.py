"""Scraper for Pokemon Showdown replays."""
import json
import time
import logging
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

REPLAY_SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_DETAIL_URL = "https://replay.pokemonshowdown.com/{replay_id}.json"

@dataclass
class ScraperConfig:
    """Configuration for replay scraper."""
    format: str = "gen9ou"
    output_dir: str = "data/raw/replays"
    requests_per_second: float = 1.0
    min_rating: Optional[int] = None  # Filter by minimum rating

class ReplayScraper:
    """Scraper for Pokemon Showdown replays."""

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        sleep_time = (1.0 / self.config.requests_per_second) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _load_checkpoint(self) -> set[str]:
        """Load IDs of already-scraped replays."""
        checkpoint_file = self.output_dir / ".checkpoint"
        if checkpoint_file.exists():
            return set(checkpoint_file.read_text().splitlines())
        return set()

    def _save_checkpoint(self, replay_ids: set[str]) -> None:
        """Save checkpoint of scraped IDs."""
        checkpoint_file = self.output_dir / ".checkpoint"
        checkpoint_file.write_text("\n".join(replay_ids))

    def search_replays(self, page: int = 1) -> list[dict]:
        """Search for replays matching the format.

        Args:
            page: Page number (1-indexed)

        Returns:
            List of replay metadata dicts
        """
        self._rate_limit()

        params = {
            "format": self.config.format,
            "page": page,
        }

        response = self.session.get(REPLAY_SEARCH_URL, params=params)
        response.raise_for_status()

        return response.json()

    def get_replay(self, replay_id: str) -> dict:
        """Fetch full replay data.

        Args:
            replay_id: The replay ID (e.g., "gen9ou-12345678")

        Returns:
            Full replay data including log
        """
        self._rate_limit()

        url = REPLAY_DETAIL_URL.format(replay_id=replay_id)
        response = self.session.get(url)
        response.raise_for_status()

        return response.json()

    def scrape_replays(self, max_replays: int = 10000) -> Iterator[dict]:
        """Scrape replays up to max count.

        Yields:
            Replay data dicts
        """
        page = 1
        count = 0

        while count < max_replays:
            logger.info(f"Fetching page {page}...")
            results = self.search_replays(page=page)

            if not results:
                logger.info("No more replays found")
                break

            for meta in results:
                if count >= max_replays:
                    break

                # Apply rating filter if configured
                if self.config.min_rating:
                    rating = meta.get("rating", 0)
                    if rating < self.config.min_rating:
                        continue

                try:
                    replay = self.get_replay(meta["id"])
                    count += 1
                    yield replay

                    if count % 100 == 0:
                        logger.info(f"Scraped {count} replays")

                except Exception as e:
                    logger.warning(f"Failed to fetch {meta['id']}: {e}")

            page += 1

    def save_replays(self, max_replays: int = 10000) -> Path:
        """Scrape and save replays to JSONL file.

        Returns:
            Path to output file
        """
        output_file = self.output_dir / f"{self.config.format}_replays.jsonl"

        with open(output_file, "w") as f:
            for replay in self.scrape_replays(max_replays):
                f.write(json.dumps(replay) + "\n")

        logger.info(f"Saved replays to {output_file}")
        return output_file
