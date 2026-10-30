"""Review ACT chunk timing from a leader-controlled shadow log.

This is an offline counterfactual: the recorded observations came from leader
teleoperation, not from ACT controlling the follower. No robot is connected and
no command is sent. Pass a shadow run directory or its .tar.gz archive.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Chunk:
    source_cycle: int
    source_s: float
    completed_s: float
    actions: np.ndarray


def read_log(path: Path) -> tuple[dict, list[dict], list[dict]]:
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text())
        with (path / "control.csv").open(newline="") as file:
            control = list(csv.DictReader(file))
        with (path / "predictions.csv").open(newline="") as file:
            predictions = list(csv.DictReader(file))
        return manifest, control, predictions
    if not path.is_file() or not tarfile.is_tarfile(path):
        raise ValueError("--log must be a shadow run directory or its .tar.gz archive")
    with tarfile.open(path) as archive:
        def member(name: str) -> bytes:
            matches = [item for item in archive.getmembers()
                       if item.isfile() and item.name.split("/")[-1] == name]
            if len(matches) != 1:
                raise ValueError(f"Archive must contain exactly one {name}")
            file = archive.extractfile(matches[0])
            if file is None:
                raise ValueError(f"Cannot read {name}")
            return file.read()

        manifest = json.loads(member("manifest.json"))
        control = list(csv.DictReader(io.StringIO(member("control.csv").decode())))
        predictions = list(csv.DictReader(io.StringIO(member("predictions.csv").decode())))
    return manifest, control, predictions


def parse_chunks(rows: list[dict], joints: list[str], expected_steps: int) -> list[Chunk]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["source_cycle"]), []).append(row)
    chunks = []
    for cycle, group in grouped.items():
        group.sort(key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in group]
        if steps != list(range(expected_steps)):
            raise ValueError(f"Chunk {cycle} has steps {steps}; expected 0..{expected_steps-1}")
        source = float(group[0]["source_s"])
        completed = float(group[0]["completed_s"])
        if completed < source or any(abs(float(row["source_s"]) - source) > 0.001 or
                                     abs(float(row["completed_s"]) - completed) > 0.001
                                     for row in group):
            raise ValueError(f"Inconsistent timestamps in chunk {cycle}")
        actions = np.asarray([[float(row[f"policy.{joint}"]) for joint in joints]
                              for row in group], dtype=np.float64)
        if not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite action in chunk {cycle}")
        chunks.append(Chunk(cycle, source, completed, actions))
    return sorted(chunks, key=lambda chunk: chunk.completed_s)


def selected_step(chunk: Chunk, now_s: float, fps: float,
                  boot_chunk: Chunk | None, boot_s: float | None) -> int:
    if chunk is boot_chunk:
        # The follower is held still before the first inference completes.
        return max(0, math.floor((now_s - boot_s) * fps + 1e-9))
    # Later chunks are indexed by the age of the observation that produced them.
    return max(0, math.ceil((now_s - chunk.source_s) * fps - 1e-9))


def rehearse(control: list[dict], chunks: list[Chunk], fps: float,
             joints: list[str]) -> tuple[list[dict], list[dict]]:
    if not control or not chunks:
        raise ValueError("The log needs control cycles and ACT chunks")
    times = [float(row["elapsed_s"]) for row in control]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("Control timestamps must increase")
    output: list[dict] = []
    handoffs: list[dict] = []
    ready_count = 0
    boot_chunk = None
    boot_s = None
    previous_chunk = None
    previous_target = None
    for cycle, (now_s, _) in enumerate(zip(times, control)):
        while ready_count < len(chunks) and chunks[ready_count].completed_s <= now_s:
            if boot_chunk is None:
                boot_chunk = chunks[ready_count]
                boot_s = now_s
            ready_count += 1
        chosen = None
        index = None
        for candidate in reversed(chunks[:ready_count]):
            candidate_index = selected_step(candidate, now_s, fps, boot_chunk, boot_s)
            if candidate_index < len(candidate.actions):
                chosen, index = candidate, candidate_index
                break
        if chosen is None:
            output.append({"cycle": cycle, "elapsed_s": f"{now_s:.5f}",
                           "status": "NO_READY_ACTION", "chunk_source_cycle": "",
                           "step": "", "observation_age_ms": "",
                           "handoff_max_gap": "", "handoff_mean_gap": ""})
            continue
        target = chosen.actions[index]
        switched = previous_chunk is not None and chosen is not previous_chunk
        max_gap = float(np.max(np.abs(target - previous_target))) if switched else 0.0
        mean_gap = float(np.mean(np.abs(target - previous_target))) if switched else 0.0
        if switched:
            handoffs.append({"elapsed_s": now_s, "from_cycle": previous_chunk.source_cycle,
                             "to_cycle": chosen.source_cycle, "new_step": index,
                             "max_gap": max_gap, "mean_gap": mean_gap})
        output.append({"cycle": cycle, "elapsed_s": f"{now_s:.5f}",
                       "status": "BOOT" if chosen is boot_chunk else "TIMED",
                       "chunk_source_cycle": chosen.source_cycle, "step": index,
                       "observation_age_ms": f"{(now_s-chosen.source_s)*1000:.2f}",
                       "handoff_max_gap": f"{max_gap:.4f}" if switched else "",
                       "handoff_mean_gap": f"{mean_gap:.4f}" if switched else "",
                       **{f"proposal.{joint}": f"{value:.5f}"
                          for joint, value in zip(joints, target, strict=True)}})
        previous_chunk, previous_target = chosen, target
    return output, handoffs


def plot_rehearsal(rows: list[dict], handoffs: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    times = [float(row["elapsed_s"]) for row in rows if row["step"] != ""]
    steps = [int(row["step"]) for row in rows if row["step"] != ""]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   constrained_layout=True)
    ax1.plot(times, steps, ".", markersize=2.5)
    ax1.set(ylabel="Selected action index", title="ACT shadow timing rehearsal (no policy motion)")
    ax1.grid(alpha=0.2)
    if handoffs:
        ax2.stem([row["elapsed_s"] for row in handoffs],
                 [row["max_gap"] for row in handoffs], basefmt=" ")
    ax2.set(xlabel="Seconds from start", ylabel="Largest joint target change\nat a chunk handoff")
    ax2.grid(alpha=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True,
                        help="ACT shadow run directory or downloaded .tar.gz archive")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for derived CSV, JSON, and plot; original log is untouched")
    args = parser.parse_args()
    manifest, control, predictions = read_log(args.log.expanduser())
    if int(manifest.get("action_offset_frames", 0)):
        raise ValueError("This legacy timing rehearsal assumes zero-offset ACT; "
                         "use act_teacher_forced_trace.py for shifted checkpoints")
    if manifest.get("policy") != "act" or manifest.get("policy_commands_sent") is not False:
        raise ValueError("Expected a leader-controlled ACT shadow log")
    steps = int(manifest["checkpoint_chunk_size"])
    if int(manifest.get("prediction_steps_logged", manifest["runtime_action_window"])) < steps:
        raise ValueError("This log contains only the short window; rerun shadow_act.py --full-chunk")
    joints = manifest["joint_order"]
    if len(joints) != 6:
        raise ValueError("Expected six arm/gripper joints")
    fps = float(manifest["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid control rate")
    chunks = parse_chunks(predictions, joints, steps)
    rows, handoffs = rehearse(control, chunks, fps, joints)
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "timing_cycles.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["cycle", "elapsed_s", "status",
                                                "chunk_source_cycle", "step",
                                                "observation_age_ms", "handoff_max_gap",
                                                "handoff_mean_gap"]
                                + [f"proposal.{joint}" for joint in joints])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "handoffs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["elapsed_s", "from_cycle", "to_cycle",
                                                "new_step", "max_gap", "mean_gap"])
        writer.writeheader()
        writer.writerows(handoffs)
    ready_future = [sum(chunk.source_s + step / fps >= chunk.completed_s
                        for step in range(steps)) for chunk in chunks]
    summary = {
        "log": str(args.log), "counterfactual": True, "policy_commands_sent": False,
        "chunks": len(chunks), "full_chunk_steps": steps,
        "inference_delay_ms_median": statistics.median(
            (chunk.completed_s - chunk.source_s) * 1000 for chunk in chunks),
        "future_steps_at_completion_median": statistics.median(ready_future),
        "no_ready_action_cycles": sum(row["status"] == "NO_READY_ACTION" for row in rows),
        "handoffs": len(handoffs),
        "largest_handoff_joint_gap": max((row["max_gap"] for row in handoffs), default=0),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_rehearsal(rows, handoffs, output / "timing.png")
    print(json.dumps(summary, indent=2))
    print("Review saved:", output)
    print("These proposals used leader-driven observations; this cannot prove autonomous behavior.")


if __name__ == "__main__":
    main()
