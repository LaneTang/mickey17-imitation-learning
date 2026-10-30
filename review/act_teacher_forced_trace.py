"""Replay recorded observations through the deployed ACT overlap schedule.

This is an offline, teacher-forced diagnostic: the robot always follows the
recorded expert trajectory, while ACT predictions are computed from recorded
images and joint positions. It never connects to the robot or sends commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))

from experiment import load_experiment
from act_overlap_queue import OverlapActionQueue
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)


def select_device(value: str) -> str:
    if value == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable on this Mac")
    return value


def load_policy(checkpoint: Path, device: str):
    config = ACTConfig.from_pretrained(checkpoint)
    expected = {"observation.state", "observation.images.wrist"}
    if set(config.input_features) != expected:
        raise ValueError("This diagnostic expects wrist image and joint state inputs")
    if tuple(config.input_features["observation.state"].shape) != (6,):
        raise ValueError("This diagnostic expects an arm-only six-position checkpoint")
    if tuple(config.output_features["action"].shape) != (6,):
        raise ValueError("This diagnostic expects six-position ACT actions")
    config.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, config, preprocessor, postprocessor


def predict_chunk(policy, preprocessor, postprocessor, row: dict) -> np.ndarray:
    image = row["observation.images.wrist"].unsqueeze(0).float() / 255.0
    state = row["observation.state"].unsqueeze(0)
    inputs = {"observation.images.wrist": image, "observation.state": state}
    with torch.inference_mode():
        prediction = postprocessor(policy.predict_action_chunk(preprocessor(inputs)))
    return prediction[0].detach().cpu().numpy().astype(np.float32)


def episode_trace(dataset: LeRobotDataset, positions: np.ndarray, states: np.ndarray,
                  commands: np.ndarray, policy, preprocessor, postprocessor,
                  chunk_size: int, replan_frames: int, delay_frames: int,
                  action_offset_frames: int = 0) -> dict:
    length = len(positions)
    queue = OverlapActionQueue(chunk_size,
                               action_offset_frames=action_offset_frames)
    proposed = np.full((length, 6), np.nan, dtype=np.float32)
    contributors = np.zeros(length, dtype=np.int32)
    source = np.full(length, -1, dtype=np.int32)
    newly_merged = np.zeros(length, dtype=np.int32)
    pending = {}

    # Model observations are taken every replan_frames. A prediction becomes
    # available delay_frames later, matching the NUC CPU timing approximately.
    for cycle in range(0, length, replan_frames):
        global_index = int(positions[cycle])
        pending[cycle + delay_frames] = (cycle, predict_chunk(
            policy, preprocessor, postprocessor, dataset[global_index]))

    for cycle in range(length):
        if cycle in pending:
            source_cycle, actions = pending[cycle]
            update = queue.add(actions, source_cycle, cycle)
            source[cycle] = source_cycle
            newly_merged[cycle] = update.merged
        action, count = queue.pop(cycle, states[cycle])
        if count:
            proposed[cycle] = action
            contributors[cycle] = count

    valid = np.isfinite(proposed).all(axis=1)
    error = proposed[valid] - commands[valid]
    lead = proposed[valid] - states[valid]
    expert_lead = commands[valid] - states[valid]
    # The gap counts are diagnostic comparisons, not hardware safety limits.
    summary = {
        "frames": int(length), "frames_with_ACT_prediction": int(valid.sum()),
        "overlapped_frames": int((contributors > 1).sum()),
        "six_joint_mae_to_expert_command": float(np.abs(error).mean()),
        "per_joint_mae_to_expert_command": {
            joint: float(value) for joint, value in zip(JOINTS, np.abs(error).mean(axis=0))
        },
        "shoulder_lift": {
            "ACT_minus_observed_max_abs": float(np.abs(lead[:, 1]).max()),
            "expert_command_minus_observed_max_abs": float(np.abs(expert_lead[:, 1]).max()),
            "ACT_minus_observed_frames_over_15": int((np.abs(lead[:, 1]) > 15).sum()),
            "expert_command_minus_observed_frames_over_15": int((np.abs(expert_lead[:, 1]) > 15).sum()),
        },
    }
    return {"states": states, "commands": commands, "proposed": proposed,
            "contributors": contributors, "source": source,
            "newly_merged": newly_merged, "summary": summary}


def plot_trace(episode: int, fps: float, trace: dict, path: Path) -> None:
    t = np.arange(len(trace["states"])) / fps
    state, expert, act = trace["states"], trace["commands"], trace["proposed"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, layout="constrained")
    for ax, joint, title in ((axes[0, 0], 1, "Shoulder lift position"),
                             (axes[0, 1], 5, "Gripper position")):
        ax.plot(t, state[:, joint], label="measured joint", color="#4672b4", lw=1.7)
        ax.plot(t, expert[:, joint], label="expert command", color="#222a36", lw=1.7)
        ax.plot(t, act[:, joint], label="ACT proposal", color="#dd6a58", lw=1.5)
        ax.set(title=title, ylabel="encoder position units")
        ax.grid(alpha=.25)
        ax.legend(loc="best")
    for ax, joint, title in ((axes[1, 0], 1, "Shoulder target minus measured"),
                             (axes[1, 1], 5, "Gripper target minus measured")):
        ax.plot(t, expert[:, joint] - state[:, joint], label="expert lead",
                color="#222a36", lw=1.5)
        ax.plot(t, act[:, joint] - state[:, joint], label="ACT lead",
                color="#dd6a58", lw=1.5)
        ax.axhline(0, color="#b4bdca", lw=.8)
        if joint == 1:
            ax.axhline(15, color="#bd905a", lw=1, ls="--", label="live stop threshold")
            ax.axhline(-15, color="#bd905a", lw=1, ls="--")
        ax.set(title=title, xlabel="seconds from episode start", ylabel="target − measured")
        ax.grid(alpha=.25)
        ax.legend(loc="best")
    fig.suptitle(f"Recorded episode {episode:03d} · offline ACT overlap replay",
                 fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_cycles(path: Path, fps: float, trace: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "elapsed_s", "contributors", "new_chunk_source_frame",
                         "newly_merged"]
                        + [f"observed.{joint}" for joint in JOINTS]
                        + [f"expert_command.{joint}" for joint in JOINTS]
                        + [f"ACT_proposal.{joint}" for joint in JOINTS])
        for frame in range(len(trace["states"])):
            writer.writerow([frame, f"{frame / fps:.5f}", int(trace["contributors"][frame]),
                             int(trace["source"][frame]), int(trace["newly_merged"][frame])]
                            + trace["states"][frame].tolist()
                            + trace["commands"][frame].tolist()
                            + trace["proposed"][frame].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "experiment.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" /
                        "act_arm_only_20261011_051140/checkpoints/004000/pretrained_model")
    parser.add_argument("--episodes", type=int, nargs="+", default=[6, 63])
    parser.add_argument("--replan-frames", type=int, default=12)
    parser.add_argument("--delay-frames", type=int, default=12,
                        help="Approximate NUC inference latency at 30 Hz")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "reviews" /
                        "act_teacher_forced_patch4_004000")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite an existing review: {args.output}")
    if (args.replan_frames < 1 or args.delay_frames < 0 or
            len(set(args.episodes)) != len(args.episodes)):
        parser.error("Replan/delay frames or episode selection is invalid")
    experiment = load_experiment(args.experiment)
    if any(episode not in experiment["validation_episodes"] for episode in args.episodes):
        parser.error("Choose distinct held-out episode IDs from experiment.json")
    checkpoint = args.checkpoint.expanduser().resolve()
    device = select_device(args.device)
    policy, config, preprocessor, postprocessor = load_policy(checkpoint, device)
    if args.replan_frames >= config.chunk_size:
        parser.error("Replan interval must be shorter than the ACT chunk")
    print(f"Checkpoint action-label offset: {config.action_offset_frames} frames")
    derived_root = ROOT / "datasets/patch4_merged_arm_only"
    source_manifest = json.loads((derived_root / "arm_only_source.json").read_text())
    if Path(source_manifest["source_root"]).resolve() != Path(experiment["dataset_root"]).resolve():
        raise ValueError("Derived ACT dataset does not match the experiment source")
    metadata = LeRobotDatasetMetadata(source_manifest["repo_id"], root=derived_root)
    delta = resolve_delta_timestamps(config, metadata, {})
    dataset = LeRobotDataset(source_manifest["repo_id"], root=derived_root,
                            episodes=args.episodes, delta_timestamps=delta,
                            video_backend="pyav", return_uint8=True)
    rows = dataset.hf_dataset
    episode_ids = np.asarray(rows["episode_index"], dtype=np.int64)
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    commands = np.asarray(rows["action"], dtype=np.float32)
    if states.shape[1:] != (6,) or commands.shape[1:] != (6,):
        raise ValueError("ACT dataset joints are not the expected six positions")
    fps = float(metadata.fps)
    summaries = {}
    args.output.mkdir(parents=True)
    for episode in args.episodes:
        positions = np.flatnonzero(episode_ids == episode)
        if not len(positions):
            raise ValueError(f"Held-out episode {episode} is missing")
        trace = episode_trace(dataset, positions, states[positions], commands[positions],
                              policy, preprocessor, postprocessor, config.chunk_size,
                              args.replan_frames, args.delay_frames,
                              config.action_offset_frames)
        plot_trace(episode, fps, trace, args.output / f"episode_{episode:03d}.png")
        save_cycles(args.output / f"episode_{episode:03d}.csv", fps, trace)
        summaries[str(episode)] = trace["summary"]
        shoulder = trace["summary"]["shoulder_lift"]
        print(f"Episode {episode}: ACT/expert MAE {trace['summary']['six_joint_mae_to_expert_command']:.2f}; "
              f"shoulder |ACT−state|>15 on {shoulder['ACT_minus_observed_frames_over_15']} frames, "
              f"expert on {shoulder['expert_command_minus_observed_frames_over_15']} frames",
              flush=True)
    (args.output / "summary.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "episodes": args.episodes,
        "replan_frames": args.replan_frames, "delay_frames": args.delay_frames,
        "action_offset_frames": config.action_offset_frames,
        "fps": fps, "mode": "teacher forced; recorded observations; no robot commands",
        "limitations": ["Fixed inference delay is an approximation of NUC timing.",
                        "First ACT chunk begins when ready after a recorded-pose hold.",
                        "Recorded states do not respond to ACT proposals."],
        "results": summaries,
    }, indent=2) + "\n")
    print(f"Saved read-only ACT diagnostic: {args.output}", flush=True)


if __name__ == "__main__":
    main()
