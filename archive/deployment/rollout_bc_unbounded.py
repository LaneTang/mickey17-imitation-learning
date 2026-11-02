"""Operator-supervised BC rollout that sends raw arm/gripper policy targets.

No joint target clipping, proposal-gap limit, target-lead limit, or total-travel
limit is applied here. The operator chooses the duration and must be beside the
robot. The LeKiwi base is commanded to zero velocity on every cycle.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_settings import load_settings
from policy_io import ARM_KEYS, full_lekiwi_action, read_observation
from preflight import BC_MODES, bc_proposal, check_default_pose, load_bc, sha256_file

def next_run_number(root: Path, policy: str) -> int:
    """Continue policy-specific numbering, including older bc_raw_* logs."""
    slug = policy.replace("-", "_")
    numbered_prefix = f"{slug}_No"
    historical_count = 0
    highest_number = 0
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        if folder.name.startswith(numbered_prefix):
            number = folder.name[len(numbered_prefix):].split("_", 1)[0]
            if number.isdecimal():
                highest_number = max(highest_number, int(number))
        manifest = folder / "manifest.json"
        if manifest.is_file():
            try:
                if json.loads(manifest.read_text()).get("policy") == policy:
                    historical_count += 1
            except (OSError, ValueError):
                pass  # An incomplete old log must not prevent a new run.
    return max(historical_count, highest_number) + 1


def raw_action(proposal: np.ndarray) -> dict[str, float]:
    """Validate the six outputs and add a zero-velocity base command."""
    return full_lekiwi_action(proposal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(BC_MODES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--default-pose", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True,
                        help="Duration you choose; there is no script-imposed maximum")
    parser.add_argument("--arm", action="store_true",
                        help="Allow raw policy actions after typing RUN; omit for preview")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    parser.add_argument("--log-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs" / "rollout_bc_unbounded")
    args = parser.parse_args()
    settings, settings_path, settings_sha256 = load_settings(args.config)
    fps = settings["deployment"]["control_hz"]
    start_tolerance = settings["deployment"]["bc_raw_start_tolerance"]
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be a positive, finite number")
    if args.arm and not sys.stdin.isatty():
        parser.error("Policy motion requires an interactive terminal")
    if len(args.expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.expected_sha256.lower()
    ):
        parser.error("--expected-sha256 must be a 64-character hex digest")

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
    from lerobot.utils.robot_utils import precise_sleep

    torch.set_num_threads(4)
    checkpoint = args.checkpoint.expanduser().resolve()
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256.lower() != args.expected_sha256.lower():
        raise RuntimeError("Checkpoint digest differs from the selected checkpoint")
    model, payload, mode, _ = load_bc(checkpoint, args.policy)

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
    cycles = overruns = 0
    stop_reason = "no motion"
    try:
        robot.connect()
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, start_tolerance)
        first = bc_proposal(model, payload, mode, state, frame)
        raw_action(first)
        print("Observed start:       ", np.round(state, 2).tolist(), flush=True)
        print("First RAW arm target: ", np.round(first, 2).tolist(), flush=True)
        print("Raw target minus start:", np.round(first - state, 2).tolist(), flush=True)
        print(f"Policy: {args.policy}; checkpoint SHA-256: {actual_sha256}", flush=True)
        print(f"Requested duration: {args.seconds:g}s; raw arm/gripper targets; base zero.", flush=True)
        if not args.arm:
            print("PREVIEW ONLY: no action sent. Add --arm for supervised motion.", flush=True)
            return

        print("No joint command clipping or travel limits. Stay at the power switch; "
              "Ctrl-C also stops the loop.", flush=True)
        if input("Type RUN to allow raw policy-controlled motion: ").strip() != "RUN":
            print("Cancelled; no command sent.", flush=True)
            return

        # Recheck after the prompt, because the robot could have moved.
        state, frame = read_observation(robot.get_observation())
        last_state = state
        check_default_pose(args.default_pose, state, args.robot_id, start_tolerance)
        first = bc_proposal(model, payload, mode, state, frame)
        raw_action(first)
        print("Actual first RAW target:", np.round(first, 2).tolist(), flush=True)
        print("Actual first delta:     ", np.round(first - state, 2).tolist(), flush=True)

        root = args.log_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_number = next_run_number(root, args.policy)
        slug = args.policy.replace("-", "_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        while True:
            log_dir = root / f"{slug}_No{run_number:03d}_{timestamp}"
            try:
                log_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                run_number += 1
        (log_dir / "manifest.json").write_text(json.dumps({
            "policy": args.policy,
            "run_number": run_number,
            "run_id": log_dir.name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_sha256,
            "default_pose": str(args.default_pose.resolve()),
            "fps_target": fps,
            "settings_path": str(settings_path),
            "settings_sha256": settings_sha256,
            "settings": settings,
            "seconds_requested": args.seconds,
            "action_mode": "unbounded raw policy arm/gripper targets",
            "base_command": {"x.vel": 0, "y.vel": 0, "theta.vel": 0},
            "start_state": state.tolist(),
            "status": "started",
        }, indent=2) + "\n")
        log_file = (log_dir / "cycles.csv").open("w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["cycle", "elapsed_s", "inference_ms", "cycle_ms"]
                        + [f"observed.{key}" for key in ARM_KEYS]
                        + [f"requested.{key}" for key in ARM_KEYS])
        log_file.flush()
        print(f"RAW POLICY MOTION: {args.seconds:g}s requested. Log: {log_dir}", flush=True)
        started = time.perf_counter()
        last_snapshot_time = -float("inf")
        stop_reason = "time limit"
        while time.perf_counter() - started < args.seconds:
            cycle_start = time.perf_counter()
            state, frame = read_observation(robot.get_observation())
            last_state = state
            inference_start = time.perf_counter()
            proposal = bc_proposal(model, payload, mode, state, frame)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            command = raw_action(proposal)
            motion_started = True
            robot.send_action(command)
            elapsed_s = time.perf_counter() - started
            cycle_ms = (time.perf_counter() - cycle_start) * 1000
            writer.writerow([cycles, f"{elapsed_s:.5f}", f"{inference_ms:.3f}",
                             f"{cycle_ms:.3f}"] + state.tolist() + proposal.tolist())
            log_file.flush()
            if elapsed_s - last_snapshot_time >= 0.5:
                snapshots.append((elapsed_s, frame.copy()))
                last_snapshot_time = elapsed_s
            cycles += 1
            overruns += cycle_ms > 1000 / fps
            precise_sleep(max(1 / fps - cycle_ms / 1000, 0))
        print(f"Finished: {cycles} raw commands; {overruns} loop overruns.", flush=True)
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
            manifest.update({"status": stop_reason, "cycles": cycles, "overruns": overruns})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved CSV and {len(snapshots)} wrist frames: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
