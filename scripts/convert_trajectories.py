#!/usr/bin/env python
"""Convert parsed battles to training trajectories."""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from poke.data.models import ParsedBattle
from poke.data.converter import TrajectoryConverter
from poke.data.encoders import SpeciesEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parsed battles JSONL")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--reward-shaping", action="store_true", help="Enable reward shaping")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: build species encoder
    print("Building species encoder...")
    species_encoder = SpeciesEncoder()
    with open(input_path) as f:
        for line in f:
            battle = ParsedBattle.model_validate_json(line)
            for turn in battle.turns:
                for p in turn.state_before.player1.team:
                    species_encoder.add(p.species)
                for p in turn.state_before.player2.team:
                    species_encoder.add(p.species)

    species_encoder.save(output_dir / "species_encoder.json")
    print(f"Found {len(species_encoder)} unique species")

    # Second pass: convert trajectories
    converter = TrajectoryConverter(
        species_encoder=species_encoder,
        reward_shaping=args.reward_shaping,
    )

    output_file = output_dir / "trajectories.jsonl"
    count = 0

    with open(input_path) as f_in, open(output_file, "w") as f_out:
        for line in tqdm(f_in, desc="Converting"):
            battle = ParsedBattle.model_validate_json(line)
            for trajectory in converter.convert(battle):
                # Serialize trajectory
                data = {
                    "replay_id": trajectory.replay_id,
                    "player": trajectory.player,
                    "total_reward": trajectory.total_reward,
                    "steps": [
                        {
                            "observation": step.observation.__dict__,
                            "action_type": step.action_type,
                            "action_target": step.action_target,
                            "reward": step.reward,
                            "done": step.done,
                        }
                        for step in trajectory.steps
                    ],
                }
                f_out.write(json.dumps(data) + "\n")
                count += 1

    print(f"Converted {count} trajectories to {output_file}")

if __name__ == "__main__":
    main()
