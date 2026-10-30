"""Read-only structural audit of raw one-episode LeRobot datasets."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

def inspect_episode(path: Path) -> dict:
    problems = []
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    parquet = sorted((path / "data").rglob("*.parquet"))
    videos = sorted((path / "videos" / "observation.images.wrist").rglob("*.mp4"))
    if not parquet:
        problems.append("missing parquet")
    if len(videos) != 1:
        problems.append(f"expected one wrist video; found {len(videos)}")
    frames = 0
    gripper_min = gripper_max = ""
    if parquet:
        tables = [pq.read_table(file, columns=["timestamp", "action"]) for file in parquet]
        frames = sum(table.num_rows for table in tables)
        if frames != info.get("total_frames"):
            problems.append(f"metadata says {info.get('total_frames')} frames; parquet has {frames}")
        if frames:
            times = np.concatenate([np.asarray(table["timestamp"].to_numpy(), dtype=float)
                                    for table in tables])
            actions = np.concatenate([np.asarray(table["action"].to_pylist(), dtype=float)
                                      for table in tables])
            if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
                problems.append("non-increasing or non-finite timestamps")
            if not np.isfinite(actions).all():
                problems.append("non-finite motor commands")
            names = info["features"]["action"]["names"]
            gripper = actions[:, names.index("arm_gripper.pos")]
            gripper_min, gripper_max = round(float(gripper.min()), 2), round(float(gripper.max()), 2)
    if info.get("total_episodes") != 1:
        problems.append("source must have exactly one episode")
    if info.get("fps") != 30:
        problems.append(f"expected 30 fps; got {info.get('fps')}")
    video_frames = 0
    video_decodable = False
    if len(videos) == 1:
        capture = cv2.VideoCapture(str(videos[0]))
        try:
            if capture.isOpened():
                video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                first_ok, _ = capture.read()
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, video_frames - 1))
                last_ok, _ = capture.read()
                video_decodable = bool(first_ok and last_ok)
        finally:
            capture.release()
        if not video_decodable:
            problems.append("wrist video first/last frame could not be decoded")
        if abs(video_frames - frames) > 1:
            problems.append(f"parquet has {frames} frames; video reports {video_frames}")
    return {"source_dir": path.name,
            "frames": frames, "fps": info.get("fps"), "video_frames": video_frames,
            "video_decodable": video_decodable, "gripper_min": gripper_min,
            "gripper_max": gripper_max, "problems": "; ".join(problems)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    args = parser.parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    if not raw_root.is_dir():
        parser.error(f"No such raw root: {raw_root}")
    episodes = sorted(path for path in raw_root.iterdir()
                      if path.is_dir() and (path / "meta" / "info.json").is_file())
    if not episodes:
        parser.error("No one-episode LeRobot folders found")
    rows = []
    for episode in episodes:
        try:
            row = inspect_episode(episode)
        except Exception as exc:
            row = {"source_dir": episode.name, "frames": "", "video_frames": "",
                   "problems": f"audit error: {exc}"}
        rows.append(row)
        print(f"{row['source_dir']}: frames={row['frames']} video={row['video_frames']} "
              f"issues={row['problems'] or 'none'}")
    problems = sum(bool(row["problems"]) for row in rows)
    print(f"Audited {len(rows)} episodes; {problems} structural issues")


if __name__ == "__main__":
    main()
