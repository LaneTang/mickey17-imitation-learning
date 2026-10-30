"""Teleoperate Mikey 17 and save a measured follower start pose on Enter.

Run on the NUC with the camera-enabled LeKiwi host already running. This script
commands only the SO-101 leader's arm targets and zero base velocity. It does
not move the follower to a stored pose or run a policy.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import select
import sys
import time
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
ARM_KEYS = (
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
)
DEFAULT_POSE_FILE = Path(__file__).with_name("default_pose.json")
STILL_FRAMES = 15  # About half a second at 30 Hz.
STILL_LIMITS = (1.5, 1.5, 1.5, 1.5, 1.5, 2.0)


def make_robot():
    return LeKiwiClient(
        LeKiwiClientConfig(
            remote_ip="127.0.0.1",
            id="Mickey17_awesome_kiwi",
            cameras={
                "wrist": OpenCVCameraConfig(
                    index_or_path="/dev/video0", width=640, height=480, fps=30
                )
            },
        )
    )


def make_leader():
    return SO101Leader(SO101LeaderConfig(port="/dev/ttyACM1", id="Mickey17_leader"))


def observed_pose(observation):
    if "wrist" not in observation:
        raise RuntimeError("Wrist image missing from the LeKiwi host")
    pose = [float(observation[key]) for key in ARM_KEYS]
    if not all(map(math.isfinite, pose)):
        raise RuntimeError("Non-finite follower joint position")
    return pose


def teleoperate_once(robot, leader):
    observation = robot.get_observation()
    pose = observed_pose(observation)
    arm_action = {f"arm_{key}": value for key, value in leader.get_action().items()}
    if set(arm_action) != set(ARM_KEYS):
        raise RuntimeError(f"Unexpected leader joints: {sorted(arm_action)}")
    if not all(math.isfinite(float(value)) for value in arm_action.values()):
        raise RuntimeError("Non-finite leader target")
    robot.send_action({**arm_action, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    return pose, observation["wrist"]


def enter_pressed():
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def pose_is_still(history):
    if len(history) < STILL_FRAMES:
        return False
    return all(
        max(sample[joint] for sample in history) - min(sample[joint] for sample in history)
        <= STILL_LIMITS[joint]
        for joint in range(len(ARM_KEYS))
    )


def save_pose(path, pose, overwrite):
    payload = {
        "format_version": 1,
        "robot_id": "Mickey17_awesome_kiwi",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "observed follower positions during leader teleoperation",
        "joint_order": list(ARM_KEYS),
        "joint_positions": [round(value, 3) for value in pose],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-file", type=Path, default=DEFAULT_POSE_FILE)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing reference pose")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Run in an interactive terminal so Enter can capture the pose")
    if args.pose_file.exists() and not args.overwrite:
        parser.error(f"Reference already exists: {args.pose_file}; use --overwrite to replace it")

    robot, leader = make_robot(), make_leader()
    history = deque(maxlen=STILL_FRAMES)
    previous_frame = None
    stale_frames = 0
    report_at = time.perf_counter()
    pose_report_at = -float("inf")
    cycles = overruns = 0
    try:
        robot.connect()
        leader.connect()
        print("Use the leader to place the follower at your chosen start pose.", flush=True)
        print("Hold it still for about half a second, then press Enter to save; Ctrl-C cancels.", flush=True)
        while True:
            cycle_start = time.perf_counter()
            pose, frame = teleoperate_once(robot, leader)
            stale_frames = stale_frames + 1 if frame is previous_frame else 0
            if stale_frames >= 3:
                raise RuntimeError("Wrist/observation stream stopped updating; no pose saved")
            previous_frame = frame
            history.append(pose)

            now = time.perf_counter()
            if now - pose_report_at >= 1:
                values = ", ".join(
                    f"{name.removeprefix('arm_').removesuffix('.pos')}={value:.1f}"
                    for name, value in zip(ARM_KEYS, pose, strict=True)
                )
                steady = "yes" if pose_is_still(history) else "no"
                print(f"Follower now: {values} | steady 0.5 s: {steady}", flush=True)
                pose_report_at = now

            if enter_pressed():
                if not pose_is_still(history):
                    print("Follower is still moving. Hold it steady and press Enter again.", flush=True)
                else:
                    save_pose(args.pose_file, pose, args.overwrite)
                    print(f"Saved default pose: {args.pose_file}", flush=True)
                    for name, value in zip(ARM_KEYS, pose, strict=True):
                        print(f"  {name}: {value:.2f}", flush=True)
                    return

            elapsed = time.perf_counter() - cycle_start
            overruns += elapsed > 1 / FPS
            precise_sleep(max(1 / FPS - elapsed, 0.0))
            cycles += 1
            now = time.perf_counter()
            if now - report_at >= 5:
                print(f"Teleoperation {cycles / (now - report_at):.1f} Hz; overruns {overruns}/{cycles}", flush=True)
                report_at, cycles, overruns = now, 0, 0
    except KeyboardInterrupt:
        print("Cancelled; no default pose saved.", flush=True)
    finally:
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
