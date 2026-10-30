"""Measure live observation-to-ACT-chunk latency without commanding Mikey17.

Run on the NUC with the camera-enabled LeKiwi host running and no other
client controlling the robot. This script only calls get_observation(); it
never calls send_action().
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))

from pipeline_settings import load_settings
from policy_io import read_observation, wrist_tensor
from preflight import load_act, sha256_file

STAGES = (
    "observation fetch",
    "image + state tensors",
    "preprocessor",
    "ACT model",
    "postprocessor",
    "to NumPy",
)


def timing_line(label: str, samples: list[float]) -> str:
    return (f"{label:23} median {statistics.median(samples):7.1f} ms  "
            f"p95 {np.percentile(samples, 95):7.1f} ms  "
            f"max {max(samples):7.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PIPELINE / "checkpoints" / "act_004000")
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--cpu-threads", type=int,
                        help="Override ACT CPU threads for this profile")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--period-ms", type=float, default=400.0,
                        help="Time between observation requests; default is 12 frames at 30 Hz")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1 or not math.isfinite(args.period_ms) or args.period_ms <= 0:
        parser.error("Warmup must be nonnegative; samples and period must be positive")

    settings, _, _ = load_settings(args.config)
    cpu_threads = (args.cpu_threads if args.cpu_threads is not None
                   else settings["act"]["runtime"]["cpu_threads"])
    if cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(cpu_threads)

    checkpoint = args.checkpoint.expanduser().resolve()
    digest = sha256_file(checkpoint / "model.safetensors")
    policy, config, preprocessor, postprocessor = load_act(checkpoint)

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip, id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera_path,
                                               width=640, height=480, fps=30)},
    ))
    stage_samples = {name: [] for name in STAGES}
    total_samples: list[float] = []
    try:
        robot.connect()
        print(f"ACT chunk {config.chunk_size}; {cpu_threads} PyTorch CPU threads; "
              f"request period {args.period_ms:g} ms")
        print(f"Checkpoint SHA-256: {digest}")
        print("READ-ONLY PROFILE: observation requests and predictions only; no robot commands.")
        with torch.inference_mode():
            for attempt in range(args.warmup + args.samples):
                started = time.perf_counter()
                state, frame = read_observation(robot.get_observation())
                after_fetch = time.perf_counter()
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
                ready = time.perf_counter()
                if actions.shape != (1, config.chunk_size, 6) or not np.isfinite(actions).all():
                    raise ValueError(f"ACT returned an invalid chunk: {actions.shape}")
                if attempt >= args.warmup:
                    marks = (started, after_fetch, after_tensors, after_pre,
                             after_model, after_post, ready)
                    for name, earlier, later in zip(STAGES, marks[:-1], marks[1:]):
                        stage_samples[name].append((later - earlier) * 1000)
                    total_samples.append((ready - started) * 1000)
                remaining = args.period_ms / 1000 - (time.perf_counter() - started)
                if remaining > 0 and attempt < args.warmup + args.samples - 1:
                    time.sleep(remaining)
    finally:
        if robot.is_connected:
            robot.disconnect()

    print(f"Measured {args.samples} live chunks after {args.warmup} warmups")
    for name in STAGES:
        print(timing_line(name, stage_samples[name]))
    print(timing_line("TOTAL observation-to-ready", total_samples))
    p95_frames = math.ceil(float(np.percentile(total_samples, 95)) * 30 / 1000)
    print(f"At 30 Hz, p95 total spans about {p95_frames} frame periods (rounded up).")
    print("This measures request-to-ready latency; camera age before the request and "
          "motor/host command delay are not included.")


if __name__ == "__main__":
    main()
