"""First, supervised ACT arm-only trial with asynchronous 30-action chunks.

Preview sends no command. --arm additionally requires an interactive RUN.
The first inference holds the measured start pose. Later chunks use their
time-aligned future action while the prior chunk supplies interim targets.
This is a short trial, not collision detection or a general safety controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_settings import apply_act_window, load_settings
from policy_io import ARM_KEYS, full_lekiwi_action, read_observation
from preflight import check_default_pose, load_act, sha256_file
from shadow_act import Prediction, predict_chunk

MAX_SECONDS = 2.0
START_TOLERANCE = 3.0
MAX_STEP = np.asarray([1, 1, 1, 1, 1, 2], dtype=np.float32)
MAX_TARGET_LEAD = np.asarray([4, 4, 4, 4, 4, 6], dtype=np.float32)
MAX_TOTAL_DISPLACEMENT = np.asarray([12] * 6, dtype=np.float32)
MAX_PROPOSAL_GAP = np.asarray([15, 15, 15, 15, 15, 35], dtype=np.float32)
MAX_HANDOFF_GAP = np.asarray([8, 8, 8, 8, 8, 15], dtype=np.float32)
# Command range from the fourth-patch demonstrations. These are data bounds,
# not physical joint limits. A small raw-prediction margin permits metering.
DATA_LOW = np.asarray([-33.5, -102.9, -62.5, -4.7, -69.3, 0], dtype=np.float32)
DATA_HIGH = np.asarray([25.6, 53.4, 59.1, 101.9, 7.4, 83.8], dtype=np.float32)


def proposed_step(chunk: Prediction, now_s: float, fps: float,
                  boot_chunk: Prediction | None, boot_s: float | None) -> int:
    if chunk is boot_chunk:
        return max(0, math.floor((now_s - boot_s) * fps + 1e-9))
    return max(0, math.ceil((now_s - chunk.source_s) * fps - 1e-9))


def meter(proposal: np.ndarray, state: np.ndarray, start: np.ndarray,
          previous: np.ndarray) -> np.ndarray:
    proposal = np.asarray(proposal, dtype=np.float32)
    if proposal.shape != (6,) or not np.isfinite(proposal).all():
        raise RuntimeError("ACT proposal is not six finite joint positions")
    if np.any(state < DATA_LOW - 5) or np.any(state > DATA_HIGH + 5):
        raise RuntimeError("Measured pose left the demonstration envelope")
    if np.any(proposal < DATA_LOW - 5) or np.any(proposal > DATA_HIGH + 5):
        raise RuntimeError("ACT proposal left the demonstration envelope")
    if np.any(np.abs(proposal - state) > MAX_PROPOSAL_GAP):
        raise RuntimeError("ACT proposal is too far from the measured pose")
    if np.any(np.abs(state - start) > MAX_TOTAL_DISPLACEMENT + 2):
        raise RuntimeError("Measured pose left the first-trial travel window")
    low = np.maximum.reduce((previous - MAX_STEP, state - MAX_TARGET_LEAD,
                             start - MAX_TOTAL_DISPLACEMENT, DATA_LOW))
    high = np.minimum.reduce((previous + MAX_STEP, state + MAX_TARGET_LEAD,
                              start + MAX_TOTAL_DISPLACEMENT, DATA_HIGH))
    if np.any(low > high):
        raise RuntimeError("Measured pose and bounded ACT target cannot be reconciled")
    return np.clip(proposal, low, high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--default-pose", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--arm", action="store_true", help="Permit policy motion after typing RUN")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "rollout_act")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= MAX_SECONDS:
        parser.error(f"First ACT trial must be > 0 and <= {MAX_SECONDS:g} seconds")
    if args.arm and not sys.stdin.isatty():
        parser.error("ACT motion requires an interactive terminal")
    if len(args.expected_sha256) != 64 or any(c not in "0123456789abcdef"
                                                   for c in args.expected_sha256.lower()):
        parser.error("--expected-sha256 must be a 64-character hex digest")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
    from lerobot.utils.robot_utils import precise_sleep

    settings, settings_path, settings_sha256 = load_settings(args.config)
    fps = float(settings["deployment"]["control_hz"])
    torch.set_num_threads(settings["act"]["runtime"]["cpu_threads"])
    checkpoint = args.checkpoint.expanduser().resolve()
    actual_sha256 = sha256_file(checkpoint / "model.safetensors")
    if actual_sha256.lower() != args.expected_sha256.lower():
        raise RuntimeError("ACT checkpoint digest differs from preflight")
    policy, config, preprocessor, postprocessor = load_act(checkpoint)
    window = apply_act_window(policy, config, settings)
    if config.chunk_size <= window:
        raise RuntimeError("ACT chunk must be longer than the 12-step replanning window")
    print(f"ACT chunk {config.chunk_size}; replan every {window} cycles; "
          f"{fps:g} Hz; checkpoint SHA-256 {actual_sha256}", flush=True)

    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip, id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera_path,
                                               width=640, height=480, fps=30)},
    ))
    last_state = None
    sent_any = False
    log_dir = None
    log_file = None
    frames = []
    pending: Future | None = None
    cycles = overruns = clipped_cycles = handoffs = 0
    inference_ms = []
    stop_reason = "preview only"
    try:
        robot.connect()
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, START_TOLERANCE)
        preview_started = time.perf_counter()
        preview = predict_chunk(policy, preprocessor, postprocessor, state.copy(), frame.copy(),
                                0, 0.0, preview_started, config.chunk_size, config.chunk_size)
        first = preview.actions[0]
        metered = meter(first, state, state, state)
        print("Observed start:       ", np.round(state, 2).tolist(), flush=True)
        print("First ACT proposal:   ", np.round(first, 2).tolist(), flush=True)
        print("First metered target: ", np.round(metered, 2).tolist(), flush=True)
        print("First command delta:  ", np.round(metered - state, 2).tolist(), flush=True)
        print(f"Full-chunk inference: {preview.inference_ms:.1f} ms; "
              f"chunk horizon {config.chunk_size / fps:.2f} s", flush=True)
        print(f"Limits: {MAX_SECONDS:g}s max; 3-unit start tolerance; base zero; "
              "1 arm/2 gripper unit per cycle; 12-unit total displacement.", flush=True)
        if not args.arm:
            print("PREVIEW ONLY: no action sent. Add --arm for supervised motion.", flush=True)
            return

        print("Clear the arm workspace. Stay at the power switch; Ctrl-C also stops the loop.",
              flush=True)
        if input("Type RUN to allow the short ACT-controlled trial: ").strip() != "RUN":
            print("Cancelled; no command sent.", flush=True)
            return
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, START_TOLERANCE)
        start_state = state.copy()
        root = args.log_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / datetime.now(timezone.utc).strftime("act_rollout_%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(exist_ok=False)
        manifest_path = log_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "policy": "act", "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha256, "default_pose": str(args.default_pose.resolve()),
            "settings_path": str(settings_path), "settings_sha256": settings_sha256,
            "fps": fps, "chunk_size": config.chunk_size, "replan_window": window,
            "seconds_requested": args.seconds, "start_state": start_state.tolist(),
            "base_command": {"x.vel": 0, "y.vel": 0, "theta.vel": 0},
            "status": "started",
        }, indent=2) + "\n")
        log_file = (log_dir / "cycles.csv").open("w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["cycle", "elapsed_s", "mode", "chunk_source_cycle", "step",
                         "inference_ms", "cycle_ms", "clipped", "handoff_max_gap"]
                        + [f"observed.{key}" for key in ARM_KEYS]
                        + [f"proposed.{key}" for key in ARM_KEYS]
                        + [f"requested.{key}" for key in ARM_KEYS])
        log_file.flush()
        print(f"ACT MOTION: {args.seconds:g}s maximum. Log: {log_dir}", flush=True)

        started = time.perf_counter()
        previous_target = start_state.copy()
        previous_proposal = None
        active: Prediction | None = None
        boot_chunk: Prediction | None = None
        boot_s: float | None = None
        last_request_s = -float("inf")
        last_snapshot_s = -float("inf")
        stop_reason = "time limit"
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-rollout") as worker:
            while time.perf_counter() - started < args.seconds:
                cycle_started = time.perf_counter()
                state, frame = read_observation(robot.get_observation())
                last_state = state
                now_s = time.perf_counter() - started
                if pending is not None and pending.done():
                    result = pending.result()
                    pending = None
                    if boot_chunk is None:
                        boot_chunk, boot_s = result, now_s
                    active = result
                    inference_ms.append(result.inference_ms)
                if pending is None and now_s - last_request_s >= window / fps:
                    pending = worker.submit(predict_chunk, policy, preprocessor, postprocessor,
                                            state.copy(), frame.copy(), cycles, now_s,
                                            started, config.chunk_size, config.chunk_size)
                    last_request_s = now_s
                if active is None:
                    mode, step, proposal = "HOLD_FIRST_INFERENCE", -1, start_state.copy()
                    source_cycle = -1
                else:
                    step = proposed_step(active, now_s, fps, boot_chunk, boot_s)
                    if step >= config.chunk_size:
                        raise RuntimeError("No current ACT prediction remains; stopping")
                    proposal = active.actions[step].astype(np.float32)
                    source_cycle = active.source_cycle
                    mode = "BOOT_CHUNK" if active is boot_chunk else "TIMED_CHUNK"
                switch = previous_proposal is not None and source_cycle != previous_source_cycle
                handoff_gap = float(np.max(np.abs(proposal - previous_proposal))) if switch else 0.0
                if switch:
                    handoffs += 1
                    allowed_gap = (MAX_PROPOSAL_GAP if previous_source_cycle == -1
                                   else MAX_HANDOFF_GAP)
                    if np.any(np.abs(proposal - previous_proposal) > allowed_gap):
                        raise RuntimeError("ACT chunk handoff target jump exceeded limit")
                target = meter(proposal, state, start_state, previous_target)
                clipped = bool(np.any(np.abs(target - proposal) > 1e-4))
                if time.perf_counter() - cycle_started > 0.10:
                    raise RuntimeError("Observation/control cycle exceeded 100 ms")
                echoed = robot.send_action(full_lekiwi_action(target))
                sent_any = True
                if not isinstance(echoed, dict) or any(key not in echoed for key in ARM_KEYS):
                    raise RuntimeError("LeKiwi did not echo the ACT arm request")
                echoed_positions = np.asarray([echoed[key] for key in ARM_KEYS], dtype=np.float32)
                if not np.isfinite(echoed_positions).all() or np.any(
                        np.abs(echoed_positions - target) > 1e-4):
                    raise RuntimeError("LeKiwi echoed a different ACT arm request")
                previous_target = target.copy()
                previous_proposal = proposal.copy()
                previous_source_cycle = source_cycle
                elapsed_s = time.perf_counter() - started
                cycle_ms = (time.perf_counter() - cycle_started) * 1000
                writer.writerow([cycles, f"{elapsed_s:.5f}", mode, source_cycle, step,
                                 f"{inference_ms[-1]:.2f}" if inference_ms else "",
                                 f"{cycle_ms:.2f}", int(clipped), f"{handoff_gap:.3f}"]
                                + state.tolist() + proposal.tolist() + target.tolist())
                log_file.flush()
                if elapsed_s - last_snapshot_s >= 0.5:
                    frames.append((elapsed_s, frame.copy()))
                    last_snapshot_s = elapsed_s
                cycles += 1
                clipped_cycles += int(clipped)
                overruns += cycle_ms > 1000 / fps
                if overruns >= 4:
                    raise RuntimeError("Four control-loop overruns; stopping")
                precise_sleep(max(1 / fps - cycle_ms / 1000, 0))
        median_inference = (f"{statistics.median(inference_ms):.1f}"
                            if inference_ms else "none completed")
        print(f"Finished: {cycles} commands; {clipped_cycles} clipped; "
              f"{handoffs} chunk handoffs; {overruns} overruns; "
              f"median inference {median_inference} ms.", flush=True)
    except KeyboardInterrupt:
        stop_reason = "operator Ctrl-C"
        print("Stopped by operator (Ctrl-C).", flush=True)
    except Exception as error:
        stop_reason = f"error: {error}"
        raise
    finally:
        if robot.is_connected:
            if sent_any and last_state is not None:
                try:
                    hold_state, _ = read_observation(robot.get_observation())
                    robot.send_action(full_lekiwi_action(hold_state))
                    time.sleep(0.15)
                    print("Queued hold-current-position and zero-base command.", flush=True)
                except Exception as error:
                    print(f"Could not queue final hold command: {error}", flush=True)
            robot.disconnect()
        if log_file is not None:
            log_file.close()
        if log_dir is not None:
            from PIL import Image
            for elapsed_s, frame in frames:
                Image.fromarray(frame).save(log_dir / f"wrist_{elapsed_s:07.3f}s.png")
            manifest_path = log_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"status": stop_reason, "cycles": cycles,
                             "clipped_cycles": clipped_cycles, "handoffs": handoffs,
                             "overruns": overruns})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved ACT log and {len(frames)} wrist frames: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
