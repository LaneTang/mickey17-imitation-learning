"""Operator-supervised ACT rollout with raw arm and gripper targets.

This deliberately has no joint clipping, proposal-gap limit, handoff-jump
limit, or total-travel limit. The operator chooses the duration. Preview mode
does not send commands; --arm also requires an interactive RUN. The LeKiwi
base is commanded to zero velocity on every cycle.

ACT inference runs in a worker thread. The measured start pose is held until
the first chunk is ready. Later chunks use the action corresponding to the
age of the observation that produced them. If a chunk runs out before its
replacement is ready, its last raw action is repeated.
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


def selected_step(chunk: Prediction, now_s: float, fps: float,
                  boot_chunk: Prediction | None, boot_s: float | None) -> int:
    if chunk is boot_chunk:
        step = math.floor((now_s - boot_s) * fps + 1e-9)
    else:
        step = math.ceil((now_s - chunk.source_s) * fps - 1e-9)
    return min(max(step, 0), len(chunk.actions) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--default-pose", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--seconds", type=float, required=True,
                        help="Operator-chosen duration; no script-imposed maximum")
    parser.add_argument("--arm", action="store_true",
                        help="Send raw ACT targets after typing RUN; omit for preview")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "rollout_act_unbounded")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be a positive, finite number")
    if args.arm and not sys.stdin.isatty():
        parser.error("Policy motion requires an interactive terminal")
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
        raise RuntimeError("ACT checkpoint digest differs from the selected checkpoint")
    policy, config, preprocessor, postprocessor = load_act(checkpoint)
    window = apply_act_window(policy, config, settings)
    if window >= config.chunk_size:
        raise RuntimeError("The ACT runtime window must be shorter than its chunk")
    print(f"ACT chunk {config.chunk_size}; replan every {window} cycles; "
          f"{fps:g} Hz; checkpoint SHA-256 {actual_sha256}", flush=True)

    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip, id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera_path,
                                               width=640, height=480, fps=30)},
    ))
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-raw-rollout")
    pending: Future | None = None
    sent_any = False
    last_state = None
    log_dir = None
    log_file = None
    frames: list[tuple[float, np.ndarray]] = []
    cycles = overruns = handoffs = stale_cycles = 0
    inference_ms: list[float] = []
    stop_reason = "preview only"
    try:
        robot.connect()
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id,
                           settings["deployment"]["bc_raw_start_tolerance"])
        preview = predict_chunk(policy, preprocessor, postprocessor,
                                state.copy(), frame.copy(), 0, 0.0,
                                time.perf_counter(), config.chunk_size, config.chunk_size)
        first = preview.actions[0].astype(np.float32)
        full_lekiwi_action(first)
        print("Observed start:       ", np.round(state, 2).tolist(), flush=True)
        print("First RAW ACT target: ", np.round(first, 2).tolist(), flush=True)
        print("Raw target minus start:", np.round(first - state, 2).tolist(), flush=True)
        print(f"Full-chunk inference: {preview.inference_ms:.1f} ms; "
              f"chunk horizon {config.chunk_size / fps:.2f} s", flush=True)
        print(f"Requested duration: {args.seconds:g}s; raw arm/gripper targets; base zero. "
              "No joint clipping or handoff-jump limit.", flush=True)
        if not args.arm:
            print("PREVIEW ONLY: no action sent. Add --arm for operator-supervised motion.",
                  flush=True)
            return

        print("Clear the workspace and stay at the power switch; Ctrl-C also stops the loop.",
              flush=True)
        if input("Type RUN to allow RAW ACT-controlled motion: ").strip() != "RUN":
            print("Cancelled; no command sent.", flush=True)
            return

        # The operator may have moved the arm while reading the preview.
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id,
                           settings["deployment"]["bc_raw_start_tolerance"])
        start_state = state.copy()
        root = args.log_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / datetime.now(timezone.utc).strftime("act_raw_%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(exist_ok=False)
        manifest_path = log_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "policy": "act", "action_mode": "unbounded raw arm/gripper targets",
            "checkpoint": str(checkpoint), "checkpoint_sha256": actual_sha256,
            "default_pose": str(args.default_pose.resolve()),
            "settings_path": str(settings_path), "settings_sha256": settings_sha256,
            "fps": fps, "chunk_size": config.chunk_size, "replan_window": window,
            "seconds_requested": args.seconds, "start_state": start_state.tolist(),
            "base_command": {"x.vel": 0, "y.vel": 0, "theta.vel": 0},
            "status": "started",
        }, indent=2) + "\n")
        log_file = (log_dir / "cycles.csv").open("w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["cycle", "elapsed_s", "mode", "chunk_source_cycle", "step",
                         "inference_ms", "cycle_ms", "handoff_max_gap",
                         "handoff_joint", "target_minus_observed_max"]
                        + [f"observed.{key}" for key in ARM_KEYS]
                        + [f"requested.{key}" for key in ARM_KEYS])
        log_file.flush()
        print(f"RAW ACT MOTION: {args.seconds:g}s requested. Log: {log_dir}", flush=True)

        started = time.perf_counter()
        active: Prediction | None = None
        boot_chunk: Prediction | None = None
        boot_s: float | None = None
        previous_target: np.ndarray | None = None
        previous_source_cycle = -1
        last_request_s = -float("inf")
        last_snapshot_s = -float("inf")
        stop_reason = "time limit"
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
                mode, step, source_cycle = "HOLD_FIRST_INFERENCE", -1, -1
                target = start_state.copy()
            else:
                unclamped_step = (math.floor((now_s - boot_s) * fps + 1e-9)
                                  if active is boot_chunk else
                                  math.ceil((now_s - active.source_s) * fps - 1e-9))
                step = selected_step(active, now_s, fps, boot_chunk, boot_s)
                target = active.actions[step].astype(np.float32)
                source_cycle = active.source_cycle
                mode = ("STALE_LAST_ACTION" if unclamped_step >= config.chunk_size else
                        "BOOT_CHUNK" if active is boot_chunk else "TIMED_CHUNK")
                stale_cycles += mode == "STALE_LAST_ACTION"
            command = full_lekiwi_action(target)
            switched = previous_target is not None and source_cycle != previous_source_cycle
            gap_vector = (np.abs(target - previous_target) if switched else
                          np.zeros(6, dtype=np.float32))
            max_gap = float(gap_vector.max())
            gap_joint = ARM_KEYS[int(gap_vector.argmax())] if switched else ""
            if switched:
                handoffs += 1
                print(f"Handoff at {now_s:.2f}s: {gap_joint} raw target jump "
                      f"{max_gap:.2f} position units", flush=True)
            robot.send_action(command)
            sent_any = True
            elapsed_s = time.perf_counter() - started
            cycle_ms = (time.perf_counter() - cycle_started) * 1000
            writer.writerow([cycles, f"{elapsed_s:.5f}", mode, source_cycle, step,
                             f"{inference_ms[-1]:.2f}" if inference_ms else "",
                             f"{cycle_ms:.2f}", f"{max_gap:.3f}", gap_joint,
                             f"{np.max(np.abs(target-state)):.3f}"]
                            + state.tolist() + target.tolist())
            log_file.flush()
            if elapsed_s - last_snapshot_s >= 0.5:
                frames.append((elapsed_s, frame.copy()))
                last_snapshot_s = elapsed_s
            previous_target = target.copy()
            previous_source_cycle = source_cycle
            cycles += 1
            overruns += cycle_ms > 1000 / fps
            precise_sleep(max(1 / fps - cycle_ms / 1000, 0))
        median_inference = (f"{statistics.median(inference_ms):.1f}"
                            if inference_ms else "none completed")
        print(f"Finished: {cycles} raw commands; {handoffs} chunk handoffs; "
              f"{stale_cycles} stale-action cycles; {overruns} loop overruns; "
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
                    try:
                        hold_state, _ = read_observation(robot.get_observation())
                    except Exception:
                        hold_state = last_state
                    robot.send_action(full_lekiwi_action(hold_state))
                    time.sleep(0.15)
                    print("Queued hold-current-position and zero-base command.", flush=True)
                except Exception as error:
                    print(f"Could not queue final hold command: {error}", flush=True)
            robot.disconnect()
        worker.shutdown(wait=False, cancel_futures=True)
        if log_file is not None:
            log_file.close()
        if log_dir is not None:
            from PIL import Image
            for elapsed_s, frame in frames:
                Image.fromarray(frame).save(log_dir / f"wrist_{elapsed_s:07.3f}s.png")
            manifest_path = log_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"status": stop_reason, "cycles": cycles,
                             "handoffs": handoffs, "stale_cycles": stale_cycles,
                             "overruns": overruns})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved raw ACT log and {len(frames)} wrist frames: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
