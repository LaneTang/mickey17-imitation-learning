"""Create one reproducible, whole-episode training/validation split.

Run after merging. This writes a new experiment.json and never edits the
dataset. BC and ACT both read the same file.
"""

import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/mikey17_patch4_all")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        parser.error("Use 0 < validation fraction < 0.5")
    root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing split: {output}")
    source_map = json.loads((root / "source_episode_map.json").read_text())
    info = json.loads((root / "meta" / "info.json").read_text())
    if set(source_map) != {str(i) for i in range(info["total_episodes"])}:
        raise ValueError("Source map does not match merged episode count")
    episodes = list(range(info["total_episodes"]))
    if len(episodes) < 2:
        raise ValueError("At least two episodes are needed for a train/validation split")
    randomizer = random.Random(args.seed)
    randomizer.shuffle(episodes)
    count = min(len(episodes) - 1, max(1, math.ceil(len(episodes) * args.validation_fraction)))
    validation = sorted(episodes[:count])
    train = sorted(episodes[count:])
    config = {"dataset_root": str(root), "repo_id": args.repo_id,
              "train_episodes": train, "validation_episodes": validation,
              "split_seed": args.seed, "split_method": "whole episodes, seeded random",
              "source_map": str(root / "source_episode_map.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved split: {output} ({len(train)} train, {len(validation)} validation)")


if __name__ == "__main__":
    main()
