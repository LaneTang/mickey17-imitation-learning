"""Leader-controlled ACT shadow test for six- or nine-channel checkpoints.

Only SO-101 leader targets and zero base velocity are sent to the follower.
ACT predictions run in a background thread and are logged, never commanded.
The configured action window controls inference cadence and the original
leader-comparison summary. --full-chunk logs all 30 predictions for an offline
timing rehearsal; the checkpoint and robot commands are unchanged.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_settings import apply_act_window, load_settings
from policy_io import (ARM_KEYS, BASE_KEYS, act_state_from_observation,
                       full_lekiwi_action, read_observation, wrist_tensor)
from preflight import check_default_pose, load_act, sha256_file
from shadow_bc import leader_positions, sent_positions


@dataclass
class Prediction:
    source_cycle: int
    source_s: float
    completed_s: float
    inference_ms: float
    actions: np.ndarray  # (logged steps, six arm/gripper joints)
    base_predictions: np.ndarray | None = None  # (logged steps, three base velocities)


def predict_chunk(policy, preprocessor, postprocessor, state: np.ndarray,
                  frame: np.ndarray, cycle: int, source_s: float,
                  start: float, logged_steps: int, chunk_size: int) -> Prediction:
    """Worker uses copied inputs and has no reference to the robot client."""
    began = time.perf_counter()
    inputs = {"observation.state": torch.from_numpy(state),
              "observation.images.wrist": wrist_tensor(frame)}
    with torch.inference_mode():
        raw = postprocessor(policy.predict_action_chunk(preprocessor(inputs)))
    actions = torch.as_tensor(raw).detach().cpu().numpy()
    channels = state.shape[0]
    if actions.shape != (1, chunk_size, channels) or not np.isfinite(actions).all():
        raise ValueError(f"ACT returned an invalid chunk: {actions.shape}")
    return Prediction(cycle, source_s, time.perf_counter() - start,
                      (time.perf_counter() - began) * 1000,
                      actions[0, :logged_steps, :6].copy(),
                      actions[0, :logged_steps, 6:].copy() if channels == 9 else None)


def summarize(controls: list[tuple[float, np.ndarray]], predictions: list[Prediction],
              fps: float, elapsed: float, overruns: int, review_window: int,
              action_offset_frames: int = 0) -> None:
    print(f"Leader control: {len(controls) / max(elapsed, 1e-9):.1f} Hz; "
          f"overruns {overruns}/{len(controls)}; ACT chunks {len(predictions)}")
    if not controls or not predictions:
        print("No aligned ACT comparisons available.")
        return
    inference = [p.inference_ms for p in predictions]
    ready = [(p.completed_s - p.source_s) * 1000 for p in predictions]
    window_ms = review_window * 1000 / fps
    print(f"Inference median {statistics.median(inference):.1f} ms; "
          f"slowest {max(inference):.1f} ms; observation-to-ready median "
          f"{statistics.median(ready):.1f} ms; {window_ms:.1f} ms action window")
    print(f"Chunks ready after their window: {sum(ms >= window_ms for ms in ready)}"
          f"/{len(ready)}")
    future_steps = [sum(p.source_s + (action_offset_frames + step) / fps >= p.completed_s
                        for step in range(len(p.actions))) for p in predictions]
    print(f"Predicted action timestamps still ahead when ready: "
          f"median {statistics.median(future_steps):g}/{len(predictions[0].actions)}")
    times = np.asarray([row[0] for row in controls], dtype=np.float64)
    targets = np.stack([row[1] for row in controls])
    gaps = []
    for prediction in predictions:
        for index, proposal in enumerate(prediction.actions[:review_window]):
            target_s = prediction.source_s + (action_offset_frames + index) / fps
            nearest = int(np.argmin(np.abs(times - target_s)))
            if abs(times[nearest] - target_s) <= 1.5 / fps:
                gaps.append(np.abs(proposal - targets[nearest]))
    if gaps:
        errors = np.stack(gaps)
        print(f"Time-aligned comparisons from first {review_window} steps "
              f"{len(errors)}; six-joint MAE "
              f"{errors.mean():.2f}; gripper MAE {errors[:, 5].mean():.2f}; "
              f"largest joint gap {errors.max():.2f}")
        print("Per-joint MAE: " + ", ".join(
            f"{joint}={mae:.2f}" for joint, mae in zip(ARM_KEYS, errors.mean(0), strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--default-pose", type=Path, required=True)
    parser.add_argument("--seconds", type=float, help="Override deployment.shadow_seconds")
    parser.add_argument("--enable-leader-motion", action="store_true",
                        help="Required to allow leader-controlled follower movement")
    parser.add_argument("--full-chunk", action="store_true",
                        help="Log all ACT chunk actions for offline timing rehearsal; never send them")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-id", default="Mickey17_leader")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "shadow_act")
    args = parser.parse_args()
    settings, settings_path, settings_sha256 = load_settings(args.config)
    deployment = settings["deployment"]
    fps = deployment["control_hz"]
    seconds = args.seconds if args.seconds is not None else deployment["shadow_seconds"]
    if not args.enable_leader_motion:
        parser.error("Add --enable-leader-motion only when ready for supervised follower motion")
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        parser.error("Use 0 < --seconds <= 60")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.robot_utils import precise_sleep

    torch.set_num_threads(settings["act"]["runtime"]["cpu_threads"])
    checkpoint = args.checkpoint.expanduser().resolve()
    policy, checkpoint_config, preprocessor, postprocessor = load_act(checkpoint)
    act_channels = checkpoint_config.input_features["observation.state"].shape[0]
    action_offset_frames = getattr(checkpoint_config, "action_offset_frames", 0)
    saved_window = checkpoint_config.n_action_steps
    window = apply_act_window(policy, checkpoint_config, settings)
    logged_steps = checkpoint_config.chunk_size if args.full_chunk else window
    print(f"ACT chunk {checkpoint_config.chunk_size}; label offset {action_offset_frames} frames; "
          f"checkpoint window {saved_window}; "
          f"runtime window {window}; logged steps {logged_steps}; {fps:g} Hz", flush=True)
    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip, id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera_path,
                                               width=640, height=480, fps=30)},
    ))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
    log_dir = None
    controls: list[tuple[float, np.ndarray]] = []
    predictions: list[Prediction] = []
    frames: list[tuple[float, np.ndarray]] = []
    pending: Future | None = None
    last_state = None
    motion_started = False
    started = None
    control_ended = None
    overruns = 0
    try:
        robot.connect()
        leader.connect()
        observation = robot.get_observation()
        state, _ = read_observation(observation)
        act_state_from_observation(observation, act_channels)
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id,
                           deployment["shadow_start_tolerance"])
        initial_leader = leader_positions(leader)
        gap = np.abs(initial_leader - state)
        print("Initial leader/follower gap:", np.round(gap, 2).tolist(), flush=True)
        if np.any(gap > deployment["shadow_leader_start_tolerance"]):
            raise RuntimeError("Leader and follower start poses differ too much; align first")
        root = args.log_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / datetime.now(timezone.utc).strftime("act_shadow_%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(exist_ok=False)
        (log_dir / "manifest.json").write_text(json.dumps({
            "policy": "act", "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
            "checkpoint_chunk_size": checkpoint_config.chunk_size,
            "act_state_channels": act_channels, "act_action_channels": act_channels,
            "base_predictions_logged": act_channels == 9,
            "base_commands_sent": {key: 0.0 for key in BASE_KEYS},
            "action_offset_frames": action_offset_frames,
            "checkpoint_action_window": saved_window, "runtime_action_window": window,
            "prediction_steps_logged": logged_steps,
            "joint_order": ARM_KEYS, "fps": fps, "seconds_requested": seconds,
            "settings_path": str(settings_path), "settings_sha256": settings_sha256,
            "settings": settings, "policy_commands_sent": False,
        }, indent=2) + "\n")
        print("SHADOW: leader commands the follower; ACT predictions are logged only.", flush=True)
        print(f"Log: {log_dir}", flush=True)
        with ((log_dir / "control.csv").open("w", newline="") as control_file,
              (log_dir / "predictions.csv").open("w", newline="") as prediction_file,
              ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-shadow") as worker):
            control = csv.writer(control_file)
            prediction = csv.writer(prediction_file)
            control.writerow(["cycle", "elapsed_s", "cycle_ms"]
                             + [f"observed.{key}" for key in ARM_KEYS]
                             + [f"leader.{key}" for key in ARM_KEYS]
                             + [f"sent.{key}" for key in ARM_KEYS])
            prediction.writerow(["source_cycle", "source_s", "target_s", "completed_s",
                                 "inference_ms", "ready_delay_ms", "step"]
                                + [f"policy.{key}" for key in ARM_KEYS]
                                + ([f"policy.{key}" for key in BASE_KEYS]
                                   if act_channels == 9 else []))
            started = time.perf_counter()
            last_request_s = -float("inf")
            last_snapshot_s = -float("inf")
            last_report_s = 0.0
            while time.perf_counter() - started < seconds:
                cycle_start = time.perf_counter()
                observation = robot.get_observation()
                state, frame = read_observation(observation)
                act_state = act_state_from_observation(observation, act_channels)
                last_state = state
                target = leader_positions(leader)
                if np.any(np.abs(target[:5] - state[:5]) > deployment["shadow_max_tracking_gap"]):
                    raise RuntimeError("Leader/follower arm tracking gap exceeded limit")
                # The only per-cycle command is the leader target, with zero base.
                accepted = sent_positions(robot.send_action(full_lekiwi_action(target)))
                motion_started = True
                elapsed = time.perf_counter() - started
                cycle = len(controls)
                controls.append((elapsed, target.copy()))
                if pending is not None and pending.done():
                    result = pending.result()
                    predictions.append(result)
                    for step, action in enumerate(result.actions):
                        prediction.writerow([result.source_cycle, f"{result.source_s:.5f}",
                                             f"{result.source_s + (action_offset_frames + step) / fps:.5f}",
                                             f"{result.completed_s:.5f}", f"{result.inference_ms:.3f}",
                                             f"{(result.completed_s-result.source_s)*1000:.3f}", step]
                                            + action.tolist()
                                            + (result.base_predictions[step].tolist()
                                               if result.base_predictions is not None else []))
                    pending = None
                if pending is None and elapsed - last_request_s >= window / fps:
                    pending = worker.submit(predict_chunk, policy, preprocessor, postprocessor,
                                            act_state.copy(), frame.copy(), cycle, elapsed,
                                            started, logged_steps, checkpoint_config.chunk_size)
                    last_request_s = elapsed
                if deployment["shadow_snapshot_every_s"] and (elapsed - last_snapshot_s >=
                        deployment["shadow_snapshot_every_s"]):
                    frames.append((elapsed, frame.copy()))
                    last_snapshot_s = elapsed
                cycle_ms = (time.perf_counter() - cycle_start) * 1000
                control.writerow([cycle, f"{elapsed:.5f}", f"{cycle_ms:.3f}"]
                                 + state.tolist() + target.tolist() + accepted.tolist())
                overruns += cycle_ms > 1000 / fps
                precise_sleep(max(1 / fps - cycle_ms / 1000, 0.0))
                if elapsed - last_report_s >= 5:
                    print(f"{elapsed:.1f}s: {len(controls)} leader cycles, "
                          f"{len(predictions)} ACT chunks, {overruns} overruns", flush=True)
                    last_report_s = elapsed
            control_ended = time.perf_counter()
            if pending is not None:
                result = pending.result()
                predictions.append(result)
                for step, action in enumerate(result.actions):
                    prediction.writerow([result.source_cycle, f"{result.source_s:.5f}",
                                                 f"{result.source_s + (action_offset_frames + step) / fps:.5f}",
                                         f"{result.completed_s:.5f}", f"{result.inference_ms:.3f}",
                                         f"{(result.completed_s-result.source_s)*1000:.3f}", step]
                                        + action.tolist()
                                        + (result.base_predictions[step].tolist()
                                           if result.base_predictions is not None else []))
    except KeyboardInterrupt:
        print("Stopped by operator (Ctrl-C).", flush=True)
    finally:
        if started is not None and control_ended is None:
            control_ended = time.perf_counter()
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            if motion_started and last_state is not None:
                try:
                    robot.send_action(full_lekiwi_action(last_state))
                    print("Queued hold-current-position and zero-base command.", flush=True)
                except Exception as error:
                    print(f"Could not queue final hold command: {error}", flush=True)
            robot.disconnect()
        if log_dir is not None:
            from PIL import Image
            for elapsed, frame in frames:
                Image.fromarray(frame).save(log_dir / f"wrist_{elapsed:07.3f}s.png")
            if started is not None:
                summarize(controls, predictions, fps, control_ended - started, overruns,
                          window, action_offset_frames)
            print(f"Saved logs and {len(frames)} wrist frames: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
