"""Derive ACT rollout software limits from recorded commands and tracking.

This reads the merged arm-only dataset and optional leader-controlled logs. It
never connects to the robot. The resulting limits describe observed operation,
not physical joint limits or collision safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


JOINTS = (
    "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
    "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def round_up(values: np.ndarray) -> list[float]:
    return [round(float(value), 1) for value in np.ceil(values * 10) / 10]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=root / "datasets/patch4_158_merged_arm_only")
    parser.add_argument("--shadow", type=Path, action="append", default=[],
                        help="Leader-controlled cycles.csv or control.csv; repeat for each run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_file = args.dataset / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_file,
                          columns=["action", "observation.state", "episode_index", "timestamp"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    episode = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    if action.shape != state.shape or action.shape[1:] != (6,):
        raise ValueError("Expected six commanded actions and six measured states")
    dt = np.diff(timestamp)
    same_episode = (episode[1:] == episode[:-1]) & (dt >= 0.025) & (dt <= 0.06)
    action_step = np.abs(np.diff(action, axis=0)[same_episode])
    state_step = np.abs(np.diff(state, axis=0)[same_episode])
    target_gap = np.abs(action - state)
    if len(action_step) < 1000 or not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Dataset has too few valid finite 30 Hz transitions")

    shadow_step_max = np.zeros(6, dtype=np.float32)
    shadow_gap_max = np.zeros(6, dtype=np.float32)
    shadow_sources = []
    for path in args.shadow:
        import csv
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for previous, current in zip(rows, rows[1:]):
            cycle_dt = float(current["elapsed_s"]) - float(previous["elapsed_s"])
            if not 0.025 <= cycle_dt <= 0.06:
                continue
            sent = np.asarray([float(current[f"sent.{joint}"]) for joint in JOINTS],
                              dtype=np.float32)
            prior = np.asarray([float(previous[f"sent.{joint}"]) for joint in JOINTS],
                               dtype=np.float32)
            observed = np.asarray([float(current[f"observed.{joint}"]) for joint in JOINTS],
                                  dtype=np.float32)
            if not (np.isfinite(sent).all() and np.isfinite(observed).all()):
                raise ValueError(f"Nonfinite shadow row in {path}")
            shadow_step_max = np.maximum(shadow_step_max, np.abs(sent - prior))
            shadow_gap_max = np.maximum(shadow_gap_max, np.abs(sent - observed))
        shadow_sources.append({"path": str(path), "sha256": sha256(path),
                               "cycles": len(rows)})

    action_p995 = np.percentile(action_step, 99.5, axis=0)
    state_p995 = np.percentile(state_step, 99.5, axis=0)
    gap_p995 = np.percentile(target_gap, 99.5, axis=0)
    max_step = np.maximum(action_p995 * 1.15, shadow_step_max * 1.10)
    target_lead = np.maximum(gap_p995 * 1.15, shadow_gap_max * 1.10)
    # An expert target must never trigger the raw-proposal stop. This is an
    # out-of-demonstration backstop; max_step and target_lead meter what moves.
    proposal_gap = np.maximum(target_gap.max(axis=0) * 1.10, target_lead + 2.0)
    data_low = np.minimum(action.min(axis=0), state.min(axis=0)) - 1.0
    data_high = np.maximum(action.max(axis=0), state.max(axis=0)) + 1.0
    data_low[5] = 0.0  # Recorded gripper commands are nonnegative.
    result = {
        "format_version": 1,
        "description": "Empirical software limits, not physical or collision limits",
        "dataset": str(args.dataset), "dataset_parquet_sha256": sha256(data_file),
        "shadow_sources": shadow_sources,
        "fps": 30, "joint_order": JOINTS,
        "moving_state_delta_mean": [round(float(np.mean(state_step[(action_step[:, i] > 0.1) |
                                                           (state_step[:, i] > 0.1), i])), 3)
                                    for i in range(6)],
        "state_delta_p995": np.round(state_p995, 3).tolist(),
        "action_delta_p995": np.round(action_p995, 3).tolist(),
        "target_gap_p995": np.round(gap_p995, 3).tolist(),
        "target_gap_max": np.round(target_gap.max(axis=0), 3).tolist(),
        "shadow_max_action_delta": np.round(shadow_step_max, 3).tolist(),
        "shadow_max_target_gap": np.round(shadow_gap_max, 3).tolist(),
        "max_step": round_up(max_step),
        "max_target_lead": round_up(target_lead),
        "max_proposal_gap": round_up(proposal_gap),
        "data_low": [round(float(value), 1) for value in np.floor(data_low * 10) / 10],
        "data_high": round_up(data_high),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Measured-state change per 30 Hz cycle, moving frames:", result["moving_state_delta_mean"])
    print("99.5th percentile measured change:", result["state_delta_p995"])
    print("ACT max step:", result["max_step"])
    print("ACT max target lead:", result["max_target_lead"])
    print("ACT proposal stop gap:", result["max_proposal_gap"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
