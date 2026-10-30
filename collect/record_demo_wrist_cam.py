"""Record one numbered LeRobot wrist-camera demonstration on Mikey 17.

Run this on the NUC with the camera-enabled LeKiwi host already running.
The leader and follower remain connected to the NUC. No viewer or Hub upload.
Each run gets the next free number in the selected output directory.
"""

import argparse
import re
import select
import sys
import time
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
TASK = "Pick up the object and place it in the marked target area"
OUTPUT_DIR = Path.home() / "Projects" / "lerobot" / "atom's" / "outputs"


def next_dataset_path(output_dir: Path) -> tuple[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(match.group(1))
        for path in output_dir.iterdir()
        if path.is_dir() and (match := re.match(r"^(\d{3})_", path.name))
    ]
    number = max([0, *numbers]) + 1
    while True:
        name = f"{number:03d}_demonstration"
        root = output_dir / name
        if not root.exists():
            return name, root
        number += 1


def teleoperate_arm_once(robot: LeKiwiClient, leader: SO101Leader):
    observation = robot.get_observation()
    if "wrist" not in observation:
        raise RuntimeError("Wrist image missing; stopping teleoperation.")
    arm_action = {f"arm_{key}": value for key, value in leader.get_action().items()}
    action = {**arm_action, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    sent_action = robot.send_action(action)
    return observation, sent_action


def teleoperate_until_enter(robot: LeKiwiClient, leader: SO101Leader) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Run this script in an interactive SSH terminal.")
    print(
        "Live reset mode: use the leader to place the arm at its start pose. "
        "Set the object and target, then press Enter to begin recording.",
        flush=True,
    )
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return
        start = time.perf_counter()
        teleoperate_arm_once(robot, leader)
        precise_sleep(max(1 / FPS - (time.perf_counter() - start), 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Folder for numbered episodes; use a separate folder for each patch")
    parser.add_argument("--task", default=TASK)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    robot = LeKiwiClient(
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
    leader = SO101Leader(SO101LeaderConfig(port="/dev/ttyACM1", id="Mickey17_leader"))

    # Allocate after the user is ready. A prior numbered folder is never overwritten.
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
    }
    dataset = None
    saved = False
    saved_root = None
    try:
        leader.connect()
        robot.connect()
        teleoperate_until_enter(robot, leader)
        name, root = next_dataset_path(args.output_dir.expanduser())

        dataset = LeRobotDataset.create(
            repo_id=f"local/{name}",
            fps=FPS,
            features=features,
            root=root,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=2,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=23),
        )
        print(f"Recording {args.seconds:g} seconds to {root}", flush=True)

        start = time.perf_counter()
        frames = overruns = 0
        observe_s = leader_s = send_s = record_s = 0.0
        while time.perf_counter() - start < args.seconds:
            cycle_start = time.perf_counter()
            observation = robot.get_observation()
            after_observe = time.perf_counter()
            if "wrist" not in observation:
                raise RuntimeError("Wrist image missing; discarding this episode.")

            arm_action = {f"arm_{key}": value for key, value in leader.get_action().items()}
            after_leader = time.perf_counter()
            action = {**arm_action, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            sent_action = robot.send_action(action)
            after_send = time.perf_counter()

            frame = {
                **build_dataset_frame(dataset.features, observation, OBS_STR),
                **build_dataset_frame(dataset.features, sent_action, ACTION),
                "task": args.task,
            }
            dataset.add_frame(frame)
            after_record = time.perf_counter()

            elapsed = after_record - cycle_start
            overruns += elapsed > 1 / FPS
            precise_sleep(max(1 / FPS - elapsed, 0))
            frames += 1
            observe_s += after_observe - cycle_start
            leader_s += after_leader - after_observe
            send_s += after_send - after_leader
            record_s += after_record - after_send

        duration = time.perf_counter() - start
        print(
            f"Captured {frames} frames in {duration:.1f}s ({frames / duration:.1f} Hz); "
            f"overruns {overruns}/{frames}; average observe {observe_s / frames * 1000:.1f}ms, "
            f"leader {leader_s / frames * 1000:.1f}ms, send {send_s / frames * 1000:.1f}ms, "
            f"record {record_s / frames * 1000:.1f}ms",
            flush=True,
        )

        # Stop controlling the robot before the potentially slow video encode.
        leader.disconnect()
        robot.disconnect()
        print("Saving the episode and encoding video; this may take a while...", flush=True)
        dataset.save_episode()
        saved = True
        saved_root = root
        print(f"Saved one local episode: {root}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted. No episode was saved.", flush=True)
    finally:
        if leader.is_connected:
            leader.disconnect()
        if robot.is_connected:
            robot.disconnect()
        if dataset is not None:
            if not saved and dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()

    if saved_root is not None:
        try:
            from plot_episode_commands import analyze_episode

            review = analyze_episode(saved_root)
            print(f"Saved command plot: {review['command_plot']}", flush=True)
            print(f"Saved wrist frame review: {review['wrist_review']}", flush=True)
            print(f"Saved episode summary: {review['summary']}", flush=True)
        except Exception as exc:
            print(f"Episode saved, but motor command plot failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
