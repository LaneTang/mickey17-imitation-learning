"""Profile ACT inference stages on saved NUC data, with no robot connection.

Reads one wrist snapshot and its recorded start pose from a prior rollout.
Never connects to LeKiwi, sends commands, trains, or edits the checkpoint.
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
from PIL import Image

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))

from pipeline_settings import load_settings
from policy_io import wrist_tensor
from preflight import load_act, sha256_file

STAGES = ("image + state tensors", "preprocessor", "ACT model", "postprocessor", "to NumPy")


def timing_line(label: str, samples: list[float]) -> str:
    return (f"{label:23} median {statistics.median(samples):7.1f} ms  "
            f"p95 {np.percentile(samples, 95):7.1f} ms  "
            f"max {max(samples):7.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Saved rollout directory with manifest.json and wrist_*.png")
    parser.add_argument("--checkpoint", type=Path,
                        default=PIPELINE / "checkpoints" / "act_004000")
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--cpu-threads", type=int,
                        help="Override CPU threads for this offline benchmark only")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("Warmup must be nonnegative and samples must be positive")

    run = args.run.expanduser().resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    frame_files = sorted(run.glob("wrist_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No wrist snapshots in {run}")
    frame = np.array(Image.open(frame_files[0]).convert("RGB"), copy=True)
    state = np.asarray(manifest["start_state"], dtype=np.float32)
    if frame.shape != (480, 640, 3) or state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Saved wrist frame or start pose has the wrong shape")

    settings, _, _ = load_settings(args.config)
    cpu_threads = (args.cpu_threads if args.cpu_threads is not None
                   else settings["act"]["runtime"]["cpu_threads"])
    if cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(cpu_threads)
    checkpoint = args.checkpoint.expanduser().resolve()
    digest = sha256_file(checkpoint / "model.safetensors")
    policy, config, preprocessor, postprocessor = load_act(checkpoint)
    print(f"NUC CPU: {cpu_threads} PyTorch threads; ACT chunk {config.chunk_size}")
    print(f"Checkpoint SHA-256: {digest}")
    print(f"Saved input: {frame_files[0]} | start state: {state.round(2).tolist()}")

    stage_samples = {name: [] for name in STAGES}
    total_samples = []
    with torch.inference_mode():
        for attempt in range(args.warmup + args.samples):
            total_started = time.perf_counter()
            mark = total_started
            inputs = {"observation.state": torch.from_numpy(state.copy()),
                      "observation.images.wrist": wrist_tensor(frame)}
            after_tensors = time.perf_counter()
            processed = preprocessor(inputs)
            after_pre = time.perf_counter()
            prediction = policy.predict_action_chunk(processed)
            after_model = time.perf_counter()
            output = postprocessor(prediction)
            after_post = time.perf_counter()
            actions = torch.as_tensor(output).detach().cpu().numpy()
            after_numpy = time.perf_counter()
            if actions.shape != (1, config.chunk_size, 6) or not np.isfinite(actions).all():
                raise ValueError(f"ACT returned an invalid chunk: {actions.shape}")
            if attempt >= args.warmup:
                marks = (mark, after_tensors, after_pre, after_model, after_post, after_numpy)
                for name, earlier, later in zip(STAGES, marks[:-1], marks[1:]):
                    stage_samples[name].append((later - earlier) * 1000)
                total_samples.append((after_numpy - total_started) * 1000)

    print(f"Measured {args.samples} full chunks after {args.warmup} warmups")
    for name in STAGES:
        print(timing_line(name, stage_samples[name]))
    print(timing_line("TOTAL", total_samples))
    print("Snapshot loading, camera capture, and robot/network communication are excluded.")


if __name__ == "__main__":
    main()
