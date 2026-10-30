"""Supervised ACT trial with time-aligned overlapping action chunks.

Preview sends no command. --arm additionally requires an interactive RUN.
The first inference holds the measured start pose. Later predictions for the
same future control cycle are blended. Targets keep rate, measured-position
lead, proposal-gap, and demonstration-envelope limits in both trial modes.
This is not collision detection or a general safety controller.
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
from act_overlap_queue import OverlapActionQueue
from policy_io import (ARM_KEYS, BASE_KEYS, act_state_from_observation,
                       full_lekiwi_action, read_observation)
from preflight import check_default_pose, load_act, sha256_file
from shadow_act import predict_chunk

MAX_SECONDS = 2.0
MAX_FULL_TRIAL_SECONDS = 40.0
START_TOLERANCE = 3.0
LIMITS_FILE = Path(__file__).with_name("empirical_limits_patch4_158.json")
EMPIRICAL_LIMITS = json.loads(LIMITS_FILE.read_text())
if (EMPIRICAL_LIMITS.get("format_version") != 1 or
        tuple(EMPIRICAL_LIMITS.get("joint_order", ())) != ARM_KEYS or
        EMPIRICAL_LIMITS.get("fps") != 30):
    raise ValueError(f"Incompatible ACT empirical limits: {LIMITS_FILE}")


def limit_vector(name: str) -> np.ndarray:
    value = np.asarray(EMPIRICAL_LIMITS[name], dtype=np.float32)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name} in {LIMITS_FILE}")
    return value


MAX_STEP = limit_vector("max_step")
MAX_TARGET_LEAD = limit_vector("max_target_lead")
MAX_TOTAL_DISPLACEMENT = np.asarray([12] * 6, dtype=np.float32)
MAX_PROPOSAL_GAP = limit_vector("max_proposal_gap")
MAX_QUEUE_PROPOSAL_STEP = np.asarray([12, 12, 12, 12, 12, 20], dtype=np.float32)
# Data bounds come from all 158 demonstrations. They are not physical limits.
DATA_LOW = limit_vector("data_low")
DATA_HIGH = limit_vector("data_high")
if (np.any(MAX_STEP <= 0) or np.any(MAX_TARGET_LEAD <= 0) or
        np.any(MAX_PROPOSAL_GAP <= MAX_TARGET_LEAD) or np.any(DATA_LOW >= DATA_HIGH)):
    raise ValueError(f"Invalid ACT empirical limit ordering: {LIMITS_FILE}")


def meter(proposal: np.ndarray, state: np.ndarray, start: np.ndarray,
          previous: np.ndarray, total_displacement: np.ndarray | None = MAX_TOTAL_DISPLACEMENT) -> np.ndarray:
    proposal = np.asarray(proposal, dtype=np.float32)
    if proposal.shape != (6,) or not np.isfinite(proposal).all():
        raise RuntimeError("ACT proposal is not six finite joint positions")
    if np.any(state < DATA_LOW - 5) or np.any(state > DATA_HIGH + 5):
        raise RuntimeError("Measured pose left the demonstration envelope")
    if np.any(proposal < DATA_LOW - 5) or np.any(proposal > DATA_HIGH + 5):
        raise RuntimeError("ACT proposal left the demonstration envelope")
    gaps = np.abs(proposal - state)
    if np.any(gaps > MAX_PROPOSAL_GAP):
        joint = int(np.argmax(gaps / MAX_PROPOSAL_GAP))
        raise RuntimeError(
            f"ACT proposal gap on {ARM_KEYS[joint]} is {gaps[joint]:.2f}; "
            f"empirical stop threshold is {MAX_PROPOSAL_GAP[joint]:.2f}"
        )
    if total_displacement is not None and np.any(np.abs(state - start) > total_displacement + 2):
        raise RuntimeError("Measured pose left the first-trial travel window")
    lower_bounds = [previous - MAX_STEP, state - MAX_TARGET_LEAD, DATA_LOW]
    upper_bounds = [previous + MAX_STEP, state + MAX_TARGET_LEAD, DATA_HIGH]
    if total_displacement is not None:
        lower_bounds.append(start - total_displacement)
        upper_bounds.append(start + total_displacement)
    low = np.maximum.reduce(lower_bounds)
    high = np.minimum.reduce(upper_bounds)
    if np.any(low > high):
        raise RuntimeError("Measured pose and bounded ACT target cannot be reconciled")
    return np.clip(proposal, low, high)


def average_arm_proposal(proposal: np.ndarray, previous: np.ndarray,
                         alpha: float) -> np.ndarray:
    """Average arm targets over time; preserve the gripper command."""
    averaged = np.asarray(proposal, dtype=np.float32).copy()
    averaged[:5] = alpha * averaged[:5] + (1.0 - alpha) * previous[:5]
    return averaged


def save_operator_rating(log_dir: Path, rollout_status: str, cycles: int) -> None:
    """Ask for a human outcome label after motion and save it beside the logs."""
    score = None
    circle_rating = None
    if cycles and sys.stdin.isatty():
        print("Rate this rollout: 0=no phase completed; 1-4=phases completed "
              "in order; 5=all four phases completed within 30 s.", flush=True)
        while True:
            try:
                answer = input("Quality score (0-5, Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nRating skipped.", flush=True)
                break
            if not answer:
                break
            if answer in {"0", "1", "2", "3", "4", "5"}:
                score = int(answer)
                break
            print("Enter a whole number from 0 to 5, or press Enter to skip.", flush=True)
        print("Cube in the circle: 0=100% inside; 1=about 95% inside; 2=about 90% inside; "
              "3=about 66% inside; 4=33% inside; 5=0% inside", flush=True)
        while True:
            try:
                answer = input("Cube-in-circle rating (0-5, Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCircle rating skipped.", flush=True)
                break
            if not answer:
                break
            if answer in {"0", "1", "2", "3", "4", "5"}:
                circle_rating = int(answer)
                break
            print("Enter a whole number from 0 to 5, or press Enter to skip.", flush=True)
    record = {
        "format_version": 2,
        "source": "operator_after_rollout",
        "quality_score": score,
        "score_out_of": 5,
        "completed_phases": min(score, 4) if score is not None else None,
        "fast_full_success": score == 5 if score is not None else None,
        "cube_in_the_circle": {
            "rating": circle_rating,
            "out_of": 5,
            "estimated_inside_percent": {0: 100, 1: 95, 2: 90, 3: 66, 4: 33, 5: 0}.get(circle_rating),
            "method": "operator visual estimate",
        },
        "rollout_status": rollout_status,
        "logged_cycles": cycles,
        "rated_at_utc": (datetime.now(timezone.utc).isoformat()
                         if score is not None or circle_rating is not None else None),
    }
    path = log_dir / "quality_label.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved rollout quality label: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--default-pose", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--arm-filter-alpha", type=float, default=0.5,
                        help="Arm target average: 0.5 weights the new proposal and previous filtered target equally; 1 disables filtering. Gripper is unchanged.")
    parser.add_argument("--full-trial", action="store_true",
                        help="Allow up to 40 seconds without the first-trial travel window; retain step, lead, proposal, and data-envelope limits")
    parser.add_argument("--arm", action="store_true", help="Permit policy motion after typing RUN")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "e2e_rollout_official_act_offset0_cube_deg0")
    args = parser.parse_args()
    max_seconds = MAX_FULL_TRIAL_SECONDS if args.full_trial else MAX_SECONDS
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= max_seconds:
        parser.error(f"ACT trial must be > 0 and <= {max_seconds:g} seconds")
    if not math.isfinite(args.arm_filter_alpha) or not 0 < args.arm_filter_alpha <= 1:
        parser.error("--arm-filter-alpha must be > 0 and <= 1")
    total_displacement = None if args.full_trial else MAX_TOTAL_DISPLACEMENT
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
    act_channels = config.input_features["observation.state"].shape[0]
    window = apply_act_window(policy, config, settings)
    action_offset_frames = getattr(config, "action_offset_frames", 0)
    if config.chunk_size <= window:
        raise RuntimeError("ACT chunk must be longer than its replanning window")
    print(f"ACT chunk {config.chunk_size}; label offset {action_offset_frames} frames; "
          f"replan every {window} cycles; "
          f"{fps:g} Hz; checkpoint SHA-256 {actual_sha256}", flush=True)
    print(f"Empirical controller limits: {LIMITS_FILE}; SHA-256 {sha256_file(LIMITS_FILE)}",
          flush=True)
    print(f"Arm average alpha: {args.arm_filter_alpha:g} (1 disables); gripper unfiltered",
          flush=True)
    print(f"Per-cycle step: {MAX_STEP.tolist()}; target lead: {MAX_TARGET_LEAD.tolist()}; "
          f"proposal stop gap: {MAX_PROPOSAL_GAP.tolist()}", flush=True)

    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip, id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera_path,
                                               width=640, height=480, fps=30)},
    ))
    last_state = None
    sent_any = False
    log_dir = None
    log_file = None
    chunk_file = None
    frames = []
    pending: Future | None = None
    worker: ThreadPoolExecutor | None = None
    cycles = overruns = clipped_cycles = overlap_updates = 0
    inference_ms = []
    stop_reason = "preview only"
    try:
        robot.connect()
        observation = robot.get_observation()
        state, frame = read_observation(observation)
        act_state = act_state_from_observation(observation, act_channels)
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, START_TOLERANCE)
        preview_started = time.perf_counter()
        preview = predict_chunk(policy, preprocessor, postprocessor, act_state.copy(), frame.copy(),
                                0, 0.0, preview_started, config.chunk_size, config.chunk_size)
        first = preview.actions[0]
        meter(first, state, state, state, total_displacement)  # Validate raw proposal.
        filtered_first = average_arm_proposal(first, state, args.arm_filter_alpha)
        metered = meter(filtered_first, state, state, state, total_displacement)
        print("Observed start:       ", np.round(state, 2).tolist(), flush=True)
        print("First ACT proposal:   ", np.round(first, 2).tolist(), flush=True)
        print("First averaged target:", np.round(filtered_first, 2).tolist(), flush=True)
        if preview.base_predictions is not None:
            print("Observed base input:  ",
                  dict(zip(BASE_KEYS, np.round(act_state[6:], 4).tolist(), strict=True)),
                  flush=True)
            print("First ACT base prediction (ignored):",
                  dict(zip(BASE_KEYS, np.round(preview.base_predictions[0], 4).tolist(),
                           strict=True)), flush=True)
        print(f"First proposal is labelled for observation + {action_offset_frames} frames.",
              flush=True)
        print("First metered target: ", np.round(metered, 2).tolist(), flush=True)
        print("First command delta:  ", np.round(metered - state, 2).tolist(), flush=True)
        print(f"Full-chunk inference: {preview.inference_ms:.1f} ms; "
              f"chunk horizon {config.chunk_size / fps:.2f} s", flush=True)
        travel_limit = ("no start-relative travel window" if args.full_trial
                        else "12-unit total displacement")
        print(f"Limits: {max_seconds:g}s max; 3-unit start tolerance; base zero; "
              f"{travel_limit}; empirical step, lead, proposal, and data-envelope "
              "checks remain active.", flush=True)
        if not args.arm:
            print("PREVIEW ONLY: no action sent. Add --arm for supervised motion.", flush=True)
            return

        print("Clear the arm workspace. Stay at the power switch; Ctrl-C also stops the loop.",
              flush=True)
        if input("Type RUN to allow the short ACT-controlled trial: ").strip() != "RUN":
            print("Cancelled; no command sent.", flush=True)
            return
        observation = robot.get_observation()
        state, frame = read_observation(observation)
        act_state_from_observation(observation, act_channels)
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, START_TOLERANCE)
        start_state = state.copy()
        root = args.log_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / datetime.now(timezone.utc).strftime(
            f"act_offset{action_offset_frames}_overlap_%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(exist_ok=False)
        manifest_path = log_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "policy": "act", "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha256, "default_pose": str(args.default_pose.resolve()),
            "act_state_channels": act_channels, "act_action_channels": act_channels,
            "base_predictions_logged": act_channels == 9,
            "settings_path": str(settings_path), "settings_sha256": settings_sha256,
            "controller_limits_path": str(LIMITS_FILE),
            "controller_limits_sha256": sha256_file(LIMITS_FILE),
            "max_step": MAX_STEP.tolist(), "max_target_lead": MAX_TARGET_LEAD.tolist(),
            "max_proposal_gap": MAX_PROPOSAL_GAP.tolist(),
            "fps": fps, "chunk_size": config.chunk_size, "replan_window": window,
            "action_offset_frames": action_offset_frames,
            "queue_new_weight": 0.7, "queue_blend_cycles": 6,
            "arm_filter_alpha": args.arm_filter_alpha,
            "full_trial": args.full_trial,
            "seconds_requested": args.seconds, "start_state": start_state.tolist(),
            "base_command": {"x.vel": 0, "y.vel": 0, "theta.vel": 0},
            "status": "started",
        }, indent=2) + "\n")
        log_file = (log_dir / "cycles.csv").open("w", newline="")
        chunk_file = (log_dir / "chunks.csv").open("w", newline="")
        writer = csv.writer(log_file)
        chunk_writer = csv.writer(chunk_file)
        writer.writerow(["cycle", "elapsed_s", "mode", "new_chunk_source_cycle",
                         "queue_contributors", "queue_depth", "new_chunk_merged",
                         "inference_ms", "cycle_ms", "clipped", "raw_proposal_step",
                         "filtered_proposal_step"]
                        + [f"observed.{key}" for key in ARM_KEYS]
                        + [f"proposed.{key}" for key in ARM_KEYS]
                        + [f"filtered.{key}" for key in ARM_KEYS]
                        + [f"requested.{key}" for key in ARM_KEYS]
                        + [f"sent.{key}" for key in ARM_KEYS])
        chunk_writer.writerow(["source_cycle", "source_s", "completed_s", "inference_ms",
                               "step", "nominal_target_s"]
                              + [f"raw.{key}" for key in ARM_KEYS]
                              + ([f"raw.{key}" for key in BASE_KEYS]
                                 if act_channels == 9 else []))
        log_file.flush()
        chunk_file.flush()
        print(f"OVERLAP ACT MOTION: {args.seconds:g}s maximum. Log: {log_dir}", flush=True)

        started = time.perf_counter()
        previous_target = start_state.copy()
        previous_proposal: np.ndarray | None = None
        previous_filtered = start_state.copy()
        queue = OverlapActionQueue(config.chunk_size,
                                   action_offset_frames=action_offset_frames)
        queue_ready = False
        last_request_s = -float("inf")
        last_snapshot_s = -float("inf")
        stop_reason = "time limit"
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-overlap")
        while time.perf_counter() - started < args.seconds:
            cycle_started = time.perf_counter()
            observation = robot.get_observation()
            state, frame = read_observation(observation)
            act_state = act_state_from_observation(observation, act_channels)
            last_state = state
            now_s = time.perf_counter() - started
            new_source = -1
            new_merged = 0
            if pending is not None and pending.done():
                result = pending.result()
                pending = None
                for step, raw_action in enumerate(result.actions):
                    chunk_writer.writerow([result.source_cycle, f"{result.source_s:.5f}",
                                           f"{result.completed_s:.5f}",
                                           f"{result.inference_ms:.3f}", step,
                                           f"{result.source_s + (action_offset_frames + step) / fps:.5f}"]
                                          + raw_action.tolist()
                                          + (result.base_predictions[step].tolist()
                                             if result.base_predictions is not None else []))
                chunk_file.flush()
                update = queue.add(result.actions, result.source_cycle, cycles)
                queue_ready = True
                new_source = result.source_cycle
                new_merged = update.merged
                overlap_updates += update.merged
                inference_ms.append(result.inference_ms)
            if pending is None and now_s - last_request_s >= window / fps:
                pending = worker.submit(predict_chunk, policy, preprocessor, postprocessor,
                                        act_state.copy(), frame.copy(), cycles, now_s,
                                        started, config.chunk_size, config.chunk_size)
                last_request_s = now_s
            proposal, contributors = queue.pop(cycles, start_state)
            if not queue_ready:
                mode = "HOLD_FIRST_INFERENCE"
            elif contributors == 0 and queue.next_cycle is not None and cycles < queue.next_cycle:
                mode = "HOLD_UNTIL_LABEL_TIME"
            elif contributors == 0:
                raise RuntimeError("ACT action queue ran dry; stopping")
            else:
                mode = "BLENDED" if contributors > 1 else "QUEUED"
            raw_step = (np.abs(proposal - previous_proposal)
                        if previous_proposal is not None else np.zeros(6, dtype=np.float32))
            meter(proposal, state, start_state, previous_target,
                  total_displacement)  # Stop on unsafe raw proposals before averaging.
            filtered = average_arm_proposal(proposal, previous_filtered,
                                            args.arm_filter_alpha)
            filtered_step = np.abs(filtered - previous_filtered)
            if np.any(filtered_step > MAX_QUEUE_PROPOSAL_STEP):
                raise RuntimeError("ACT averaged proposal changed too far in one cycle")
            target = meter(filtered, state, start_state, previous_target,
                           total_displacement)
            clipped = bool(np.any(np.abs(target - filtered) > 1e-4))
            if time.perf_counter() - cycle_started > 0.10:
                raise RuntimeError("Observation/control cycle exceeded 100 ms")
            echoed = robot.send_action(full_lekiwi_action(target))
            sent_any = True
            if not isinstance(echoed, dict) or any(
                    key not in echoed for key in ARM_KEYS + BASE_KEYS):
                raise RuntimeError("LeKiwi did not echo the full ACT request")
            echoed_positions = np.asarray([echoed[key] for key in ARM_KEYS], dtype=np.float32)
            echoed_base = np.asarray([echoed[key] for key in BASE_KEYS], dtype=np.float32)
            if not np.isfinite(echoed_positions).all() or np.any(
                    np.abs(echoed_positions - target) > 1e-4):
                raise RuntimeError("LeKiwi echoed a different ACT arm request")
            if not np.isfinite(echoed_base).all() or np.any(echoed_base != 0):
                raise RuntimeError("LeKiwi echoed a nonzero base request")
            previous_target = target.copy()
            previous_proposal = proposal.copy()
            previous_filtered = filtered.copy()
            elapsed_s = time.perf_counter() - started
            cycle_ms = (time.perf_counter() - cycle_started) * 1000
            writer.writerow([cycles, f"{elapsed_s:.5f}", mode, new_source,
                             contributors, queue.queued_steps, new_merged,
                             f"{inference_ms[-1]:.2f}" if inference_ms else "",
                             f"{cycle_ms:.2f}", int(clipped), f"{raw_step.max():.3f}",
                             f"{filtered_step.max():.3f}"]
                            + state.tolist() + proposal.tolist() + filtered.tolist()
                            + target.tolist()
                            + echoed_positions.tolist())
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
              f"{overlap_updates} overlapping future actions merged; {overruns} overruns; "
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
        if worker is not None:
            worker.shutdown(wait=False, cancel_futures=True)
        if log_file is not None:
            log_file.close()
        if chunk_file is not None:
            chunk_file.close()
        if log_dir is not None:
            from PIL import Image
            for elapsed_s, frame in frames:
                Image.fromarray(frame).save(log_dir / f"wrist_{elapsed_s:07.3f}s.png")
            manifest_path = log_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"status": stop_reason, "cycles": cycles,
                             "clipped_cycles": clipped_cycles,
                             "overlap_updates": overlap_updates,
                             "overruns": overruns})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved ACT log and {len(frames)} wrist frames: {log_dir}", flush=True)
            save_operator_rating(log_dir, stop_reason, cycles)


if __name__ == "__main__":
    main()
