"""Compare abrupt ACT chunk switching with gradual overlap on a shadow log.

This reads previously recorded leader-controlled observations. It predicts
what the two action schedulers would have proposed; it does not connect to or
command a robot, and cannot establish autonomous task success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deployment"))
from act_overlap_queue import OverlapActionQueue
from act_timing_rehearsal import parse_chunks, read_log, rehearse


def replay_old(rows: list[dict], start: np.ndarray, joints: list[str]) -> np.ndarray:
    previous = start.copy()
    proposals = []
    for row in rows:
        if row["step"]:
            previous = np.asarray([float(row[f"proposal.{joint}"]) for joint in joints],
                                  dtype=np.float32)
        proposals.append(previous.copy())
    return np.asarray(proposals)


def replay_overlap(control: list[dict], chunks, start: np.ndarray,
                   chunk_size: int, new_weight: float,
                   blend_cycles: int) -> tuple[np.ndarray, list[int], int]:
    queue = OverlapActionQueue(chunk_size, len(start), new_weight, blend_cycles)
    previous = start.copy()
    proposals = []
    contributors = []
    ready = merged = 0
    for cycle, row in enumerate(control):
        now_s = float(row["elapsed_s"])
        while ready < len(chunks) and chunks[ready].completed_s <= now_s:
            chunk = chunks[ready]
            update = queue.add(chunk.actions, chunk.source_cycle, cycle)
            merged += update.merged
            ready += 1
        proposal, count = queue.pop(cycle, previous)
        proposals.append(proposal)
        contributors.append(count)
        previous = proposal
    return np.asarray(proposals), contributors, merged


def summary_steps(actions: np.ndarray, first_cycles: int) -> dict[str, float | int]:
    changes = np.abs(np.diff(actions, axis=0)).max(axis=1)
    return {
        "largest_one_cycle_target_change": float(changes.max()),
        "one_cycle_changes_over_8": int(np.count_nonzero(changes > 8)),
        "one_cycle_changes_over_15": int(np.count_nonzero(changes > 15)),
        "first_two_seconds_largest_change": float(changes[:first_cycles].max()),
    }


def plot_comparison(times: np.ndarray, old: np.ndarray, new: np.ndarray,
                    joints: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    old_change = np.r_[0, np.abs(np.diff(old, axis=0)).max(axis=1)]
    new_change = np.r_[0, np.abs(np.diff(new, axis=0)).max(axis=1)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    axes[0].plot(times, old_change, label="Abrupt switch", color="#d94e41", alpha=.75)
    axes[0].plot(times, new_change, label="Gradual overlap", color="#1878a5", alpha=.85)
    axes[0].axhline(8, color="gray", linestyle="--", linewidth=1)
    axes[0].set(ylabel="Largest raw target change\nin one control cycle",
                title="ACT scheduling rehearsal on leader-controlled shadow observations")
    axes[0].legend()
    for axis, joint in zip(axes[1:], ("arm_shoulder_lift.pos", "arm_gripper.pos"), strict=True):
        index = joints.index(joint)
        axis.plot(times, old[:, index], label="Abrupt switch", color="#d94e41", alpha=.75)
        axis.plot(times, new[:, index], label="Gradual overlap", color="#1878a5", alpha=.85)
        axis.set(ylabel=joint.removeprefix("arm_").replace(".pos", "") + " target")
    for axis in axes:
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Seconds")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True,
                        help="Full-chunk ACT shadow folder or .tar.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--new-weight", type=float, default=.7)
    parser.add_argument("--blend-cycles", type=int, default=6)
    args = parser.parse_args()
    if not math.isfinite(args.new_weight) or not 0 < args.new_weight <= 1:
        parser.error("--new-weight must be in (0, 1]")
    if args.blend_cycles < 1:
        parser.error("--blend-cycles must be positive")
    manifest, control, predictions = read_log(args.log.expanduser())
    if int(manifest.get("action_offset_frames", 0)):
        raise ValueError("This legacy comparison assumes zero-offset ACT; "
                         "use act_teacher_forced_trace.py for shifted checkpoints")
    if manifest.get("policy") != "act" or manifest.get("policy_commands_sent") is not False:
        raise ValueError("Expected a leader-controlled ACT shadow log")
    chunk_size = int(manifest["checkpoint_chunk_size"])
    if int(manifest.get("prediction_steps_logged", 0)) != chunk_size:
        raise ValueError("Expected every ACT action from --full-chunk shadow")
    joints = list(manifest["joint_order"])
    fps = float(manifest["fps"])
    chunks = parse_chunks(predictions, joints, chunk_size)
    old_rows, _ = rehearse(control, chunks, fps, joints)
    start = np.asarray([float(control[0][f"observed.{joint}"]) for joint in joints],
                       dtype=np.float32)
    old = replay_old(old_rows, start, joints)
    new, contributors, merged = replay_overlap(control, chunks, start, chunk_size,
                                                args.new_weight, args.blend_cycles)
    first_cycles = min(round(2 * fps), len(control) - 1)
    report = {
        "source_log": str(args.log), "counterfactual": True,
        "policy_commands_sent": False, "control_cycles": len(control),
        "chunks": len(chunks), "merged_future_actions": merged,
        "queue_empty_cycles": int(sum(n == 0 for n in contributors)),
        "new_weight": args.new_weight, "blend_cycles": args.blend_cycles,
        "abrupt": summary_steps(old, first_cycles),
        "gradual_overlap": summary_steps(new, first_cycles),
    }
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "cycles.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["cycle", "elapsed_s", "contributors"]
                        + [f"abrupt.{joint}" for joint in joints]
                        + [f"overlap.{joint}" for joint in joints])
        for cycle, row in enumerate(control):
            writer.writerow([cycle, row["elapsed_s"], contributors[cycle]]
                            + old[cycle].tolist() + new[cycle].tolist())
    times = np.asarray([float(row["elapsed_s"]) for row in control])
    plot_comparison(times, old, new, joints, output / "comparison.png")
    print(json.dumps(report, indent=2))
    print("Review saved:", output)
    print("Leader-driven observations cannot prove autonomous behavior.")


if __name__ == "__main__":
    main()
