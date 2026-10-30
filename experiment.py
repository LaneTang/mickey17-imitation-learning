"""Shared experiment settings for the Mac training scripts."""

import json
from pathlib import Path


def load_experiment(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    required = {"dataset_root", "repo_id", "train_episodes", "validation_episodes"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing experiment fields: {sorted(missing)}")
    if not config["dataset_root"]:
        raise ValueError("Set dataset_root to the new merged dataset first")
    root = Path(config["dataset_root"]).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Merged dataset not found: {root}")
    if not isinstance(config["repo_id"], str) or not config["repo_id"]:
        raise ValueError("repo_id must be a nonempty string")
    train = config["train_episodes"]
    validation = config["validation_episodes"]
    if not train or not validation:
        raise ValueError("Both train_episodes and validation_episodes must be set")
    if any(type(i) is not int or i < 0 for i in train + validation):
        raise ValueError("Episode IDs must be nonnegative integers")
    if len(set(train + validation)) != len(train + validation):
        raise ValueError("Training and validation episodes overlap or repeat")
    info = json.loads((root / "meta" / "info.json").read_text())
    if info.get("total_tasks") != 1:
        raise ValueError("This ACT split wrapper expects one task; review task labels before training")
    total = info["total_episodes"]
    if set(train + validation) != set(range(total)):
        raise ValueError(f"The split must contain every merged episode ID 0..{total - 1} exactly once")
    config["dataset_root"] = str(root)
    config["config_path"] = str(path)
    config["total_episodes"] = total
    config["total_frames"] = info["total_frames"]
    return config
