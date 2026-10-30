"""Time one ACT chunk on the Mac using a saved wrist frame and joint state.

No robot connection, motor commands, training, or checkpoint changes.
Timing includes frame-to-tensor conversion, LeRobot preprocessing, model
inference, postprocessing, and conversion of the full chunk to NumPy. Camera
capture and any Mac-to-NUC network transfer are outside this benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment"))

from act_teacher_forced_trace import load_policy, select_device
from policy_io import wrist_tensor
from lerobot.datasets import LeRobotDataset


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" /
                        "act_arm_only_20261011_051140/checkpoints/004000/pretrained_model")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "datasets/patch4_merged_arm_only")
    parser.add_argument("--episode", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("Warmup must be nonnegative and samples must be positive")

    device = select_device(args.device)
    dataset_root = args.dataset.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = json.loads((dataset_root / "arm_only_source.json").read_text())
    dataset = LeRobotDataset(manifest["repo_id"], root=dataset_root,
                            episodes=[args.episode], video_backend="pyav",
                            return_uint8=True)
    sample = dataset[0]
    frame = np.ascontiguousarray(
        sample["observation.images.wrist"].permute(1, 2, 0).numpy())
    state = sample["observation.state"].numpy().astype(np.float32, copy=True)
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8 or state.shape != (6,):
        raise ValueError("Saved observation does not match the deployed wrist/state shape")
    policy, config, preprocessor, postprocessor = load_policy(checkpoint, device)

    timings = []
    with torch.inference_mode():
        for index in range(args.warmup + args.samples):
            synchronize(device)
            started = time.perf_counter()
            inputs = {"observation.state": torch.from_numpy(state.copy()),
                      "observation.images.wrist": wrist_tensor(frame)}
            output = postprocessor(policy.predict_action_chunk(preprocessor(inputs)))
            actions = torch.as_tensor(output).detach().cpu().numpy()
            synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if actions.shape != (1, config.chunk_size, 6) or not np.isfinite(actions).all():
                raise ValueError(f"Unexpected ACT output: {actions.shape}")
            if index >= args.warmup:
                timings.append(elapsed_ms)

    print(f"Device: {device}; ACT chunk: {config.chunk_size}; "
          f"warmups: {args.warmup}; measured runs: {args.samples}")
    print(f"Mac single-observation inference: median {statistics.median(timings):.1f} ms; "
          f"p95 {np.percentile(timings, 95):.1f} ms; "
          f"slowest {max(timings):.1f} ms")
    print("Includes tensor conversion, preprocessing, inference, and postprocessing. "
          "Excludes camera capture and Tailscale transfer.")


if __name__ == "__main__":
    main()
