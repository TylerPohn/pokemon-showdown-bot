#!/usr/bin/env python
"""Parse raw replays into structured format."""
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

from poke.data.parser import BattleLogParser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output", help="Output JSONL file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".parsed.jsonl")

    battle_parser = BattleLogParser()
    success = 0
    failed = 0

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in tqdm(f_in, desc="Parsing"):
            replay = json.loads(line)
            parsed = battle_parser.parse(replay)

            if parsed:
                f_out.write(parsed.model_dump_json() + "\n")
                success += 1
            else:
                failed += 1

    print(f"Parsed {success} battles, {failed} failed")

if __name__ == "__main__":
    main()
