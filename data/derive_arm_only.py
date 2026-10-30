"""Create a separate six-joint LeRobot dataset for arm-only ACT training.

Copies the merged dataset to a new directory, then slices action and
observation.state and their metadata from nine values to the first six.
The source dataset is never changed. Preview is the default; --create writes.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

JOINTS = ("arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
          "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos")
FEATURES = ("action", "observation.state")


def check_source(root: Path) -> dict:
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    for feature in FEATURES:
        spec = info["features"][feature]
        if spec["shape"] != [9] or spec["names"][:6] != list(JOINTS):
            raise ValueError(f"Unexpected {feature} layout: {spec}")
    stats = json.loads((root / "meta/stats.json").read_text())
    if any(value != 0 for key in ("min", "max") for value in stats["action"][key][6:]):
        raise ValueError("Base commands are not all zero; arm-only ACT would discard real commands")
    if not list((root / "data").rglob("*.parquet")):
        raise ValueError("No frame Parquet files found")
    if not list((root / "videos").rglob("*.mp4")):
        raise ValueError("No wrist videos found")
    return info


def slice_parquet(path: Path, names: tuple[str, ...], expected_rows: int | None = None) -> int:
    table = pq.read_table(path)
    if expected_rows is not None and table.num_rows != expected_rows:
        raise ValueError(f"Unexpected frame count in {path}: {table.num_rows}")
    for name in names:
        field_index = table.schema.get_field_index(name)
        if field_index < 0:
            raise KeyError(f"{name} missing from {path}")
        field = table.schema.field(field_index)
        values = table.column(field_index).to_pylist()
        if any(len(value) != 9 for value in values):
            raise ValueError(f"Expected nine values for {name} in {path}")
        sliced = pa.array((value[:6] for value in values), type=field.type)
        table = table.set_column(field_index, field, sliced)
    temporary = path.with_suffix(path.suffix + ".new")
    pq.write_table(table, temporary)
    temporary.replace(path)
    return table.num_rows


def create_arm_only(source: Path, output: Path, info: dict, repo_id: str) -> None:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output or partial output already exists: {output}")
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the source dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Independent copies, including the videos, make this dataset portable.
    shutil.copytree(source, partial)
    frame_count = 0
    for path in sorted((partial / "data").rglob("*.parquet")):
        frame_count += slice_parquet(path, FEATURES)
    if frame_count != info["total_frames"]:
        raise ValueError(f"Copied {frame_count} frames, expected {info['total_frames']}; inspect {partial}")
    for path in sorted((partial / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        vector_names = tuple(name for name in table.column_names
                             if any(name.startswith(f"stats/{feature}/") for feature in FEATURES)
                             and not name.endswith("/count"))
        slice_parquet(path, vector_names)
    info_path = partial / "meta/info.json"
    derived_info = json.loads(info_path.read_text())
    for feature in FEATURES:
        derived_info["features"][feature]["shape"] = [6]
        derived_info["features"][feature]["names"] = list(JOINTS)
    info_path.write_text(json.dumps(derived_info, indent=2) + "\n")
    stats_path = partial / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    for feature in FEATURES:
        for stat, values in stats[feature].items():
            if stat != "count":
                if len(values) != 9:
                    raise ValueError(f"Unexpected {feature}/{stat} statistic length")
                stats[feature][stat] = values[:6]
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    source_info = (source / "meta/info.json").read_bytes()
    (partial / "arm_only_source.json").write_text(json.dumps({
        "source_root": str(source), "source_info_sha256": hashlib.sha256(source_info).hexdigest(),
        "repo_id": repo_id, "joint_order": JOINTS, "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"], "base_command_values": [0, 0, 0],
    }, indent=2) + "\n")

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=partial, episodes=[0, info["total_episodes"] - 1],
                             video_backend="pyav")
    for frame in (0, len(dataset) - 1):
        row = dataset[frame]
        if tuple(row["action"].shape) != (6,) or tuple(row["observation.state"].shape) != (6,):
            raise ValueError(f"Derived sample has wrong dimensions at frame {frame}")
        if tuple(row["observation.images.wrist"].shape) != (3, 480, 640):
            raise ValueError(f"Derived wrist image has wrong shape at frame {frame}")
    partial.rename(output)
    print(f"ARM-ONLY DATASET: {output}")
    print(f"Episodes: {info['total_episodes']}; frames: {frame_count}; action/state: 6/6")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default="local/mikey17_patch4_arm_only")
    parser.add_argument("--create", action="store_true", help="Create the derived copy")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    info = check_source(source)
    print(f"Source: {source} ({info['total_episodes']} episodes, {info['total_frames']} frames)")
    print(f"Output: {output}; source remains untouched")
    if not args.create:
        print("PREVIEW ONLY. Add --create to write the arm-only dataset.")
        return
    create_arm_only(source, output, info, args.repo_id)


if __name__ == "__main__":
    main()
