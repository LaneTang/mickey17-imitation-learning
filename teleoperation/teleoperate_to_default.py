"""Leader teleoperation with live distance to a saved follower start pose.

Run on the NUC with the camera-enabled LeKiwi host already running. Only the
leader moves the follower; the saved pose is guidance, not a motor command.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from lerobot.utils.robot_utils import precise_sleep

from define_default_position import (
    ARM_KEYS,
    DEFAULT_POSE_FILE,
    FPS,
    STILL_FRAMES,
    enter_pressed,
    make_leader,
    make_robot,
    teleoperate_once,
)


def load_pose(path):
    payload = json.loads(path.read_text())
    if payload.get("format_version") != 1 or payload.get("robot_id") != "Mickey17_awesome_kiwi":
        raise ValueError("Reference pose format or robot ID does not match Mikey 17")
    if payload.get("joint_order") != list(ARM_KEYS):
        raise ValueError("Reference joint names/order do not match the current robot")
    pose = [float(value) for value in payload["joint_positions"]]
    if len(pose) != len(ARM_KEYS) or not all(map(math.isfinite, pose)):
        raise ValueError("Reference contains invalid joint positions")
    return pose


def within_tolerance(current, target, joint_tolerance, gripper_tolerance):
    differences = [now - ref for now, ref in zip(current, target, strict=True)]
    limits = [joint_tolerance] * 5 + [gripper_tolerance]
    return differences, all(abs(gap) <= limit for gap, limit in zip(differences, limits, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-file", type=Path, default=DEFAULT_POSE_FILE)
    parser.add_argument("--joint-tolerance", type=float, default=2.0, help="Degrees for each of the five arm joints")
    parser.add_argument("--gripper-tolerance", type=float, default=2.0, help="Gripper position units")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Run in an interactive terminal so Enter can finish alignment")
    if args.joint_tolerance <= 0 or args.gripper_tolerance <= 0:
        parser.error("Tolerances must be positive")
    if not args.pose_file.is_file():
        parser.error(f"Reference pose missing: {args.pose_file}; run define_default_position.py first")
    target = load_pose(args.pose_file)

    robot, leader = make_robot(), make_leader()
    previous_frame = None
    stale_frames = 0
    aligned_frames = 0
    last_report = -float("inf")
    try:
        robot.connect()
        leader.connect()
        print(f"Reference: {args.pose_file}", flush=True)
        print("Move with the leader. Delta = observed follower position minus reference.", flush=True)
        print("Hold all joints within tolerance for about half a second; then press Enter.", flush=True)
        print("Ctrl-C stops without marking alignment complete.", flush=True)
        while True:
            cycle_start = time.perf_counter()
            current, frame = teleoperate_once(robot, leader)
            stale_frames = stale_frames + 1 if frame is previous_frame else 0
            if stale_frames >= 3:
                raise RuntimeError("Wrist/observation stream stopped updating")
            previous_frame = frame

            differences, aligned = within_tolerance(
                current, target, args.joint_tolerance, args.gripper_tolerance
            )
            aligned_frames = aligned_frames + 1 if aligned else 0
            ready = aligned_frames >= STILL_FRAMES
            now = time.perf_counter()
            if now - last_report >= 1:
                print("READY" if ready else "Aligning", flush=True)
                for name, value, gap in zip(ARM_KEYS, current, differences, strict=True):
                    print(f"  {name:<25} current {value:>7.2f}  delta {gap:>+7.2f}", flush=True)
                last_report = now

            if enter_pressed():
                if ready:
                    print("Aligned to the saved joint pose. Teleoperation stopped.", flush=True)
                    return
                print("Not yet within tolerance for half a second; keep guiding with the leader.", flush=True)

            elapsed = time.perf_counter() - cycle_start
            precise_sleep(max(1 / FPS - elapsed, 0.0))
    except KeyboardInterrupt:
        print("Stopped before alignment was confirmed.", flush=True)
    finally:
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
