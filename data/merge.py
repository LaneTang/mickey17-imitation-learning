"""Merge all structurally valid one-episode datasets into a new dataset.

Preview is the default. Raw demonstrations are read and never modified.
"""

import argparse
import json
from pathlib import Path

from audit import inspect_episode


def source_episodes(raw_root: Path, expected_count: int | None) -> list[tuple[Path, int]]:
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    paths = sorted((path for path in raw_root.iterdir() if path.is_dir()),
                   key=lambda path: path.name)
    if not paths:
        raise ValueError("No episode directories found")
    expected_names = [f"{i:03d}_demonstration" for i in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("Expected consecutive numbered episode directories starting at 001")
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} episodes; found {len(paths)}")
    sources = []
    for path in paths:
        check = inspect_episode(path)
        if check["problems"]:
            raise ValueError(f"Episode {path.name} failed structural audit: {check['problems']}")
        sources.append((path, check["frames"]))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/mikey17_patch4_all")
    parser.add_argument("--expected-episodes", type=int,
                        help="Require this many raw episode folders before merging")
    parser.add_argument("--merge", action="store_true", help="Create the new merged dataset")
    args = parser.parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_name(output.name + ".partial").exists():
        parser.error(f"Output or partial output already exists: {output}")
    if output == raw_root or raw_root in output.parents:
        parser.error("Merged output must be outside the raw recording directory")
    sources = source_episodes(raw_root, args.expected_episodes)
    expected_frames = sum(frames for _, frames in sources)
    print(f"All {len(sources)} episodes passed structural audit; {expected_frames} frames")
    print(f"Output: {output}")
    for new_id, (path, frames) in enumerate(sources):
        print(f"  {new_id:03d} <- {path.name} ({frames} frames)")
    if not args.merge:
        print("PREVIEW ONLY. Add --merge to create a new dataset.")
        return

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets

    partial = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets = [LeRobotDataset(f"local/mikey17_source_{i:03d}", root=path)
                for i, (path, _) in enumerate(sources)]
    merged = merge_datasets(datasets, args.repo_id, output_dir=partial)
    info = merged.meta.info
    if (info.total_episodes != len(sources) or
            info.total_frames != expected_frames):
        raise RuntimeError(f"Merged counts differ; inspect partial output: {partial}")
    source_map = {
        str(i): {"source_dir": str(path), "frames": frames}
        for i, (path, frames) in enumerate(sources)
    }
    (partial / "source_episode_map.json").write_text(json.dumps(source_map, indent=2) + "\n")
    partial.rename(output)
    print(f"MERGED: {output}")


if __name__ == "__main__":
    main()
