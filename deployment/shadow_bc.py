"""Leader-controlled BC shadow test for Mikey17.

The SO-101 leader's targets, with zero base velocity, are the only commands
sent to LeKiwi. BC proposals are compared and logged, never sent. Running this
script DOES move the follower under leader teleoperation; --enable-leader-motion
is required. Keep the camera-enabled LeKiwi host running in another terminal.
"""

import argparse
import csv
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_settings import load_settings
from policy_io import ARM_KEYS, full_lekiwi_action, read_observation
from preflight import BC_MODES, bc_proposal, check_default_pose, load_bc, sha256_file

def leader_positions(leader) -> np.ndarray:
    action = {f"arm_{key}": value for key, value in leader.get_action().items()}
    if set(action) != set(ARM_KEYS):
        raise ValueError(f"Leader joint names differ from the follower: {sorted(action)}")
    values = np.asarray([action[key] for key in ARM_KEYS], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Leader returned non-finite joint positions")
    return values


def sent_positions(command: dict) -> np.ndarray:
    if not isinstance(command, dict) or any(key not in command for key in ARM_KEYS):
        raise ValueError("LeKiwi did not return the six arm commands it accepted")
    values = np.asarray([command[key] for key in ARM_KEYS], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("LeKiwi returned a non-finite accepted command")
    return values


def report_window(errors: list[np.ndarray], tracking: list[np.ndarray],
                  inference_ms: list[float], cycles: int, overruns: int,
                  began: float, ended: float, horizon: int) -> None:
    if not errors:
        return
    gaps = np.asarray(errors)
    following = np.asarray(tracking)
    print(
        f"{ended - began:.1f}s: {cycles / (ended - began):.1f} Hz; "
        f"overruns {overruns}/{cycles}; "
        f"policy-vs-leader (+{horizon} frames) MAE {gaps.mean():.2f}; "
        f"gripper MAE {gaps[:, 5].mean():.2f}; "
        f"largest gap {gaps.max():.2f}; "
        f"leader-vs-state MAE {following.mean():.2f}; "
        f"inference median {np.median(inference_ms):.1f} ms",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(BC_MODES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--default-pose", type=Path, required=True)
    parser.add_argument("--seconds", type=float, help="Override deployment.shadow_seconds")
    parser.add_argument("--enable-leader-motion", action="store_true",
                        help="Allow the leader arm to command the follower")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--leader-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-id", default="Mickey17_leader")
    parser.add_argument("--start-tolerance", type=float)
    parser.add_argument("--leader-start-tolerance", type=float)
    parser.add_argument("--max-tracking-gap", type=float,
                        help="Stop if any of the five arm joints differs this much from the leader")
    parser.add_argument("--snapshot-every-s", type=float,
                        help="Keep raw wrist frames in memory; write PNGs after motion stops. 0 disables.")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "shadow_bc")
    args = parser.parse_args()
    settings, settings_path, settings_sha256 = load_settings(args.config)
    deployment = settings["deployment"]
    fps = deployment["control_hz"]
    for field, key in (("seconds", "shadow_seconds"),
                       ("start_tolerance", "shadow_start_tolerance"),
                       ("leader_start_tolerance", "shadow_leader_start_tolerance"),
                       ("max_tracking_gap", "shadow_max_tracking_gap"),
                       ("snapshot_every_s", "shadow_snapshot_every_s")):
        if getattr(args, field) is None:
            setattr(args, field, deployment[key])
    if not args.enable_leader_motion:
        parser.error("Add --enable-leader-motion only when ready for supervised follower motion")
    if not 0 < args.seconds <= 60:
        parser.error("Use 0 < --seconds <= 60")
    if any(not np.isfinite(x) or x <= 0 for x in
           (args.start_tolerance, args.leader_start_tolerance, args.max_tracking_gap)):
        parser.error("Pose and tracking tolerances must be positive")
    if not np.isfinite(args.snapshot_every_s) or args.snapshot_every_s < 0:
        parser.error("--snapshot-every-s must be nonnegative")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    from lerobot.utils.robot_utils import precise_sleep

    torch.set_num_threads(4)
    checkpoint = args.checkpoint.expanduser().resolve()
    model, payload, mode, horizon = load_bc(checkpoint, args.policy)
    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip, id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera_path,
                                               width=640, height=480, fps=30)},
    ))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))

    log_dir = None
    log_stream = None
    snapshots: list[tuple[float, np.ndarray]] = []
    motion_started = False
    last_state = None
    started = 0.0
    cycles = overruns = 0
    all_errors: list[np.ndarray] = []
    all_tracking: list[np.ndarray] = []
    all_inference_ms: list[float] = []
    window_errors: list[np.ndarray] = []
    window_tracking: list[np.ndarray] = []
    window_inference_ms: list[float] = []
    window_started = 0.0
    window_cycles = window_overruns = 0
    pending = deque()
    try:
        robot.connect()
        leader.connect()
        last_state, frame = read_observation(robot.get_observation())
        check_default_pose(args.default_pose, last_state, args.robot_id, args.start_tolerance)
        leader_start = leader_positions(leader)
        leader_gap = np.abs(leader_start - last_state)
        print("Leader minus follower start:", np.round(leader_start - last_state, 2).tolist(), flush=True)
        if np.any(leader_gap > args.leader_start_tolerance):
            failed = [ARM_KEYS[i] for i in np.flatnonzero(leader_gap > args.leader_start_tolerance)]
            raise RuntimeError(f"Leader and follower are not aligned at start: {failed}")

        log_root = args.log_root.expanduser().resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        log_dir = log_root / datetime.now(timezone.utc).strftime("bc_shadow_%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(exist_ok=False)
        (log_dir / "manifest.json").write_text(json.dumps({
            "policy": args.policy, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint), "joint_order": ARM_KEYS,
            "horizon_frames": horizon, "fps": fps, "seconds_requested": args.seconds,
            "settings_path": str(settings_path), "settings_sha256": settings_sha256,
            "settings": settings,
            "default_pose": str(args.default_pose),
            "policy_commands_sent": False,
        }, indent=2) + "\n")
        log_stream = (log_dir / "cycles.csv").open("w", newline="")
        writer = csv.writer(log_stream)
        writer.writerow(["cycle", "elapsed_s", "policy_target_cycle", "inference_ms", "cycle_ms"]
                        + [f"observed.{key}" for key in ARM_KEYS]
                        + [f"leader.{key}" for key in ARM_KEYS]
                        + [f"sent.{key}" for key in ARM_KEYS]
                        + [f"policy.{key}" for key in ARM_KEYS])
        print("SHADOW: the leader controls the follower; BC proposals are logged only.", flush=True)
        print(f"Log: {log_dir}", flush=True)
        started = window_started = time.perf_counter()
        last_snapshot_at = -float("inf")
        while time.perf_counter() - started < args.seconds:
            cycle_start = time.perf_counter()
            observation = robot.get_observation()
            state, frame = read_observation(observation)
            last_state = state
            target = leader_positions(leader)
            if np.any(np.abs(target[:5] - state[:5]) > args.max_tracking_gap):
                raise RuntimeError("Leader/follower arm tracking gap exceeded the limit; stopping")

            # This is the ONLY per-cycle robot command. It comes from the
            # leader, not from the network prediction below.
            sent = robot.send_action(full_lekiwi_action(target))
            motion_started = True
            accepted = sent_positions(sent)
            inference_start = time.perf_counter()
            proposed = bc_proposal(model, payload, mode, state, frame)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            if proposed.shape != (6,) or not np.isfinite(proposed).all():
                raise RuntimeError("Policy generated an invalid target")

            pending.append((cycles, proposed.copy(), state.copy()))
            if len(pending) > horizon:
                source_cycle, aligned, source_state = pending.popleft()
                if cycles - source_cycle != horizon:
                    raise AssertionError("Policy/leader time alignment failed")
                gap = np.abs(aligned - accepted)
                follow_gap = np.abs(accepted - source_state)
                all_errors.append(gap)
                all_tracking.append(follow_gap)
                window_errors.append(gap)
                window_tracking.append(follow_gap)

            all_inference_ms.append(inference_ms)
            window_inference_ms.append(inference_ms)
            elapsed = time.perf_counter() - started
            if args.snapshot_every_s and elapsed - last_snapshot_at >= args.snapshot_every_s:
                snapshots.append((elapsed, frame.copy()))
                last_snapshot_at = elapsed
            cycle_ms = (time.perf_counter() - cycle_start) * 1000
            writer.writerow([cycles, f"{elapsed:.5f}", cycles + horizon,
                             f"{inference_ms:.3f}", f"{cycle_ms:.3f}"]
                            + state.tolist() + target.tolist() + accepted.tolist() + proposed.tolist())
            overrun = cycle_ms > 1000 / fps
            cycles += 1
            window_cycles += 1
            overruns += overrun
            window_overruns += overrun
            precise_sleep(max(1 / fps - cycle_ms / 1000, 0.0))
            now = time.perf_counter()
            if now - window_started >= 5 and window_errors:
                report_window(window_errors, window_tracking, window_inference_ms,
                              window_cycles, window_overruns, window_started, now, horizon)
                window_started = now
                window_cycles = window_overruns = 0
                window_errors.clear()
                window_tracking.clear()
                window_inference_ms.clear()
    except KeyboardInterrupt:
        print("Stopped by operator (Ctrl-C).", flush=True)
    finally:
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            if motion_started and last_state is not None:
                try:
                    # Hold the most recently measured pose and stop the base.
                    # This target also does not come from the policy.
                    try:
                        hold_state, _ = read_observation(robot.get_observation())
                    except Exception:
                        hold_state = last_state
                    robot.send_action(full_lekiwi_action(hold_state))
                    print("Queued hold-current-position and zero-base command.", flush=True)
                except Exception as error:
                    print(f"Could not queue final hold command: {error}", flush=True)
            robot.disconnect()
        if log_stream is not None:
            log_stream.close()
        if log_dir is not None:
            from PIL import Image
            for elapsed, raw_frame in snapshots:
                Image.fromarray(raw_frame).save(log_dir / f"wrist_{elapsed:07.3f}s.png")
            if all_errors:
                errors = np.asarray(all_errors)
                print(f"FINISHED: {cycles} cycles; {overruns} overruns; "
                      f"policy-vs-leader MAE {errors.mean():.2f}; "
                      f"gripper MAE {errors[:, 5].mean():.2f}; "
                      f"largest gap {errors.max():.2f}; "
                      f"inference median {np.median(all_inference_ms):.1f} ms", flush=True)
            print(f"Saved CSV and {len(snapshots)} wrist frames: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
