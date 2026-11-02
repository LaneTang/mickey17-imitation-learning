"""Short, supervised arm-only rollout for the patch-4 BC checkpoints.

Policy commands move the follower only with --arm and an interactive RUN
confirmation. Keep the camera-enabled LeKiwi host running and stay at the
robot's power switch. This script is a guarded first trial, not a collision
detector or a mechanical safety controller.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from policy_io import ARM_KEYS, full_lekiwi_action, read_observation
from preflight import BC_MODES, bc_proposal, check_default_pose, load_bc, sha256_file

FPS = 30.0
MAX_SECONDS = 2.0
START_TOLERANCE = 3.0
MAX_ARM_PROPOSAL_GAP = 10.0
MAX_GRIPPER_PROPOSAL_GAP = 35.0
MAX_STEP = np.asarray([1, 1, 1, 1, 1, 2], dtype=np.float32)
MAX_TARGET_LEAD = np.asarray([3, 3, 3, 3, 3, 4], dtype=np.float32)
MAX_TOTAL_DISPLACEMENT = np.asarray([12, 12, 12, 12, 12, 12], dtype=np.float32)

# Patch-4 recorded *command* extrema, rounded outward. These are a data
# envelope, not physical joint limits or collision protection.
DATA_LOW = np.asarray([-33.5, -102.9, -62.5, -4.7, -69.3, 0], dtype=np.float32)
DATA_HIGH = np.asarray([25.6, 53.4, 59.1, 101.9, 7.4, 83.8], dtype=np.float32)


def check_proposal(proposal: np.ndarray, state: np.ndarray,
                   start: np.ndarray, previous_target: np.ndarray) -> np.ndarray:
    """Meter a target from the last request, with a measured-position leash."""
    if proposal.shape != (6,) or not np.isfinite(proposal).all():
        raise RuntimeError("Policy proposal is not six finite joint positions")
    if np.any(state < DATA_LOW - 2) or np.any(state > DATA_HIGH + 2):
        raise RuntimeError("Measured pose left the recorded action envelope")
    if np.any(proposal < DATA_LOW - 2) or np.any(proposal > DATA_HIGH + 2):
        raise RuntimeError("Policy proposal is outside the recorded action envelope")
    if np.max(np.abs(proposal[:5] - state[:5])) > MAX_ARM_PROPOSAL_GAP:
        raise RuntimeError("Policy requested an arm target over 10 position units away")
    if abs(float(proposal[5] - state[5])) > MAX_GRIPPER_PROPOSAL_GAP:
        raise RuntimeError("Policy requested an unexpectedly large gripper change")
    if np.any(np.abs(state - start) > MAX_TOTAL_DISPLACEMENT + 2):
        raise RuntimeError("Measured pose left the first-trial displacement window")

    # A target can advance even while a motor has not moved, but it may never
    # get farther than MAX_TARGET_LEAD from the measured joint position.
    desired = np.clip(proposal, DATA_LOW, DATA_HIGH)
    low = np.maximum.reduce((previous_target - MAX_STEP, state - MAX_TARGET_LEAD,
                             start - MAX_TOTAL_DISPLACEMENT, DATA_LOW))
    high = np.minimum.reduce((previous_target + MAX_STEP, state + MAX_TARGET_LEAD,
                              start + MAX_TOTAL_DISPLACEMENT, DATA_HIGH))
    if np.any(low > high):
        raise RuntimeError("Measured arm position and bounded target cannot be reconciled")
    target = np.clip(desired, low, high)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(BC_MODES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--default-pose", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True,
                        help="Digest from the checkpoint-only preflight")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--arm", action="store_true",
                        help="Permit policy motion after typing RUN; omit for preview")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "rollout_bc")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= MAX_SECONDS:
        parser.error(f"First-trial duration must be > 0 and <= {MAX_SECONDS:g} seconds")
    if args.arm and not sys.stdin.isatty():
        parser.error("Policy motion requires an interactive terminal")
    if len(args.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.expected_sha256.lower()):
        parser.error("--expected-sha256 must be a 64-character hex digest")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
    from lerobot.utils.robot_utils import precise_sleep

    torch.set_num_threads(4)
    checkpoint = args.checkpoint.expanduser().resolve()
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256.lower() != args.expected_sha256.lower():
        raise RuntimeError("Checkpoint digest differs from the selected checkpoint")
    model, payload, mode, horizon = load_bc(checkpoint, args.policy)

    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip,
        id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(
            index_or_path=args.camera_path, width=640, height=480, fps=30)},
    ))
    last_state = None
    motion_started = False
    log_dir = None
    log_file = None
    snapshots = []
    cycles = overruns = clipped_cycles = 0
    started = None
    stop_reason = "no motion"
    try:
        robot.connect()
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, START_TOLERANCE)
        first = bc_proposal(model, payload, mode, state, frame)
        first_target = check_proposal(first, state, state, state)
        print("Observed start:      ", np.round(state, 2).tolist(), flush=True)
        print("First BC proposal:   ", np.round(first, 2).tolist(), flush=True)
        print("First metered target:", np.round(first_target, 2).tolist(), flush=True)
        print("First command delta: ", np.round(first_target - state, 2).tolist(), flush=True)
        print(f"Policy: {args.policy}; checkpoint SHA-256: {actual_sha256}", flush=True)
        print(f"Limits: {MAX_SECONDS:g}s max; 3-unit start tolerance; "
              "1 arm/2 gripper unit per cycle; 3 arm/4 gripper unit target lead; "
              "12-unit total displacement; base zero.", flush=True)
        if not args.arm:
            print("PREVIEW ONLY: no action sent. Add --arm for supervised motion.", flush=True)
            return

        print("Clear the workspace. Stay at the power switch; Ctrl-C also stops the loop.", flush=True)
        if input("Type RUN to allow the short policy-controlled trial: ").strip() != "RUN":
            print("Cancelled; no command sent.", flush=True)
            return

        # Recheck after the human confirmation; the robot may have moved while
        # the prompt was displayed.
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, START_TOLERANCE)
        start_state = state.copy()
        first = bc_proposal(model, payload, mode, state, frame)
        check_proposal(first, state, start_state, start_state)

        root = args.log_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        log_dir = root / datetime.now(timezone.utc).strftime("bc_rollout_%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(exist_ok=False)
        (log_dir / "manifest.json").write_text(json.dumps({
            "policy": args.policy,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha256,
            "default_pose": str(args.default_pose.resolve()),
            "fps_target": FPS,
            "seconds_requested": args.seconds,
            "base_command": {"x.vel": 0, "y.vel": 0, "theta.vel": 0},
            "max_step": MAX_STEP.tolist(),
            "max_target_lead": MAX_TARGET_LEAD.tolist(),
            "max_total_displacement": MAX_TOTAL_DISPLACEMENT.tolist(),
            "start_state": start_state.tolist(),
            "status": "started",
        }, indent=2) + "\n")
        log_file = (log_dir / "cycles.csv").open("w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["cycle", "elapsed_s", "inference_ms", "cycle_ms", "clipped"]
                        + [f"observed.{key}" for key in ARM_KEYS]
                        + [f"proposed.{key}" for key in ARM_KEYS]
                        + [f"requested.{key}" for key in ARM_KEYS])
        log_file.flush()
        print(f"POLICY MOTION: {args.seconds:g}s maximum. Log: {log_dir}", flush=True)
        started = time.perf_counter()
        previous_target = start_state.copy()
        previous_frame = None
        last_snapshot_time = -float("inf")
        stop_reason = "time limit"
        while time.perf_counter() - started < args.seconds:
            cycle_start = time.perf_counter()
            state, frame = read_observation(robot.get_observation())
            last_state = state
            if frame is previous_frame:
                raise RuntimeError("Observation stream reused the previous wrist frame")
            previous_frame = frame
            inference_start = time.perf_counter()
            proposal = bc_proposal(model, payload, mode, state, frame)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            target = check_proposal(proposal, state, start_state, previous_target)
            clipped = bool(np.any(np.abs(target - proposal) > 1e-4))
            if time.perf_counter() - cycle_start > 0.10:
                raise RuntimeError("Observation/inference exceeded 100 ms; no new command sent")
            motion_started = True
            echoed = robot.send_action(full_lekiwi_action(target))
            # LeKiwiClient returns a local echo, not an acknowledgement from
            # the NUC motor controller. Only later observations show tracking.
            if not isinstance(echoed, dict) or any(key not in echoed for key in ARM_KEYS):
                raise RuntimeError("LeKiwi did not echo the requested arm command")
            echoed_positions = np.asarray([echoed[key] for key in ARM_KEYS], dtype=np.float32)
            if not np.isfinite(echoed_positions).all():
                raise RuntimeError("LeKiwi echoed a non-finite command")
            if np.any(np.abs(echoed_positions - target) > 1e-4):
                raise RuntimeError("LeKiwi echoed an arm command different from the request")
            previous_target = target.copy()
            elapsed_s = time.perf_counter() - started
            cycle_ms = (time.perf_counter() - cycle_start) * 1000
            writer.writerow([cycles, f"{elapsed_s:.5f}", f"{inference_ms:.3f}", f"{cycle_ms:.3f}",
                             int(clipped)] + state.tolist() + proposal.tolist() + target.tolist())
            log_file.flush()
            if elapsed_s - last_snapshot_time >= 0.5:
                snapshots.append((elapsed_s, frame.copy()))
                last_snapshot_time = elapsed_s
            cycles += 1
            clipped_cycles += int(clipped)
            overruns += cycle_ms > 1000 / FPS
            if overruns >= 4:
                raise RuntimeError("Four control-loop overruns; stopping")
            precise_sleep(max(1 / FPS - cycle_ms / 1000, 0))
        print(f"Finished: {cycles} commands; {clipped_cycles} clipped; "
              f"{overruns} overruns.", flush=True)
    except KeyboardInterrupt:
        stop_reason = "operator Ctrl-C"
        print("Stopped by operator (Ctrl-C).", flush=True)
    except Exception as error:
        stop_reason = f"error: {error}"
        raise
    finally:
        if robot.is_connected:
            if motion_started and last_state is not None:
                try:
                    # A hold target is not a torque-off or emergency stop.
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
        if log_file is not None:
            log_file.close()
        if log_dir is not None:
            from PIL import Image
            for elapsed_s, frame in snapshots:
                Image.fromarray(frame).save(log_dir / f"wrist_{elapsed_s:07.3f}s.png")
            manifest_path = log_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"status": stop_reason, "cycles": cycles,
                             "clipped_cycles": clipped_cycles, "overruns": overruns})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved CSV and {len(snapshots)} wrist frames: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
