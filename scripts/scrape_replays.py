#!/usr/bin/env python
"""CLI for scraping Pokemon Showdown replays."""
import argparse
import logging

from poke.data.scraper import ReplayScraper, ScraperConfig

def main():
    parser = argparse.ArgumentParser(description="Scrape Pokemon Showdown replays")
    parser.add_argument("--format", default="gen9ou", help="Battle format")
    parser.add_argument("--max-replays", type=int, default=10000, help="Max replays to scrape")
    parser.add_argument("--min-rating", type=int, help="Minimum rating filter")
    parser.add_argument("--output-dir", default="data/raw/replays", help="Output directory")
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ScraperConfig(
        format=args.format,
        output_dir=args.output_dir,
        requests_per_second=args.rate,
        min_rating=args.min_rating,
    )

    scraper = ReplayScraper(config)
    output_file = scraper.save_replays(max_replays=args.max_replays)
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
