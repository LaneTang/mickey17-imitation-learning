"""Read-only checkpoint and optional live-observation preflight for BC or ACT.

Checkpoint-only mode loads and validates a model without connecting to Mikey17.
--live reads one observation from the running LeKiwi host, checks the recorded
default pose, and predicts without sending any command. There is no send_action
call in this file.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE / "train"))
sys.path.insert(0, str(PIPELINE))

from bc_model import VanillaBC
from bc_targets import action_from_prediction
from pipeline_settings import apply_act_window, load_settings
from policy_io import (ARM_KEYS, BASE_KEYS, BASE_ZERO, act_state_from_observation,
                       full_lekiwi_action, read_observation, wrist_tensor)

BC_MODES = {
    "bc-direct": ("direct_same_frame", "direct", 0),
    "bc-residual": ("residual_same_frame", "residual", 0),
    "bc-future-residual": ("future_residual_3_frames", "future-residual", 3),
}
ACT_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json",
             "policy_postprocessor.json")
BC_REPO_IDS = frozenset({"local/mikey17_patch4_all", "local/mikey17_patch4_158_all"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bc(checkpoint: Path, policy_type: str):
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"BC checkpoint missing or empty: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    repo_id = payload.get("experiment", {}).get("repo_id")
    if repo_id not in BC_REPO_IDS:
        raise ValueError(f"BC checkpoint dataset {repo_id!r} is not a supported patch-4 dataset")
    label, mode, horizon = BC_MODES[policy_type]
    if payload.get("label_mode") != label or payload.get("horizon_frames") != horizon:
        raise ValueError(f"Checkpoint label/horizon does not match {policy_type}")
    if tuple(payload.get("joint_order", ())) != ARM_KEYS or payload.get("image_size") != [120, 160]:
        raise ValueError("Unexpected BC joint order or image resolution")
    for key in ("state_mean", "state_std", "target_mean", "target_std"):
        value = payload.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (6,) or not torch.isfinite(value).all():
            raise ValueError(f"Missing or invalid {key}")
    if torch.any(payload["state_std"] <= 0) or torch.any(payload["target_std"] <= 0):
        raise ValueError("BC normalization standard deviations must be positive")
    model = VanillaBC().eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, payload, mode, horizon


def load_act(checkpoint: Path):
    missing = [name for name in ACT_FILES if not (checkpoint / name).is_file()
               or (checkpoint / name).stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"ACT checkpoint incomplete: {missing}")
    saved_config = json.loads((checkpoint / "config.json").read_text())
    saved_offset = saved_config.get("action_offset_frames", 0)
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = ACTConfig.from_pretrained(checkpoint)
    if (type(saved_offset) is not int or saved_offset < 0 or
            getattr(config, "action_offset_frames", 0) != saved_offset or
            config.action_delta_indices[0] != saved_offset):
        raise ValueError("ACT action-label offset is unsupported or inconsistent; "
                         "install the matching LeRobot ACTConfig before using this checkpoint")
    if set(config.input_features) != {"observation.state", "observation.images.wrist"}:
        raise ValueError("ACT input names differ from the wrist-camera pipeline")
    state_shape = tuple(config.input_features["observation.state"].shape)
    if state_shape not in ((6,), (9,)):
        raise ValueError("ACT state must have six arm values or nine LeKiwi values")
    if tuple(config.input_features["observation.images.wrist"].shape) != (3, 480, 640):
        raise ValueError("ACT wrist image feature has the wrong shape")
    if (set(config.output_features) != {"action"} or
            tuple(config.output_features["action"].shape) != state_shape):
        raise ValueError("ACT output width must match its six- or nine-value state")
    if not 1 <= config.n_action_steps <= config.chunk_size:
        raise ValueError("Invalid ACT action window")
    config.device = "cpu"
    model = ACTPolicy.from_pretrained(checkpoint, config=config).to("cpu").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return model, config, preprocessor, postprocessor


def check_default_pose(path: Path, state: np.ndarray, robot_id: str, tolerance: float) -> None:
    saved = json.loads(path.read_text())
    if saved.get("robot_id") != robot_id or tuple(saved.get("joint_order", ())) != ARM_KEYS:
        raise ValueError("Default pose belongs to another robot or joint order")
    reference = np.asarray(saved.get("joint_positions"), dtype=np.float32)
    if reference.shape != (6,) or not np.isfinite(reference).all():
        raise ValueError("Default pose must have six finite joint positions")
    difference = state - reference
    print("Saved default pose:       ", np.round(reference, 2).tolist())
    print("Current minus default:    ", np.round(difference, 2).tolist())
    if np.any(np.abs(difference) > tolerance):
        failed = [ARM_KEYS[i] for i in np.flatnonzero(np.abs(difference) > tolerance)]
        raise ValueError(f"Start pose differs by more than {tolerance:g} on {failed}; "
                         "realign by supervised teleoperation before a shadow test")
    print(f"DEFAULT POSE: PASS (every joint within {tolerance:g} position units)")


def bc_proposal(model, payload, mode: str, state: np.ndarray, frame: np.ndarray) -> np.ndarray:
    image = F.interpolate(wrist_tensor(frame).unsqueeze(0), size=(120, 160),
                          mode="bilinear", align_corners=False)
    measured = torch.from_numpy(state.copy())
    normalized = (measured - payload["state_mean"]) / payload["state_std"]
    with torch.inference_mode():
        output = model(image, normalized.unsqueeze(0)).squeeze(0)
        command = action_from_prediction(output, measured, payload["target_mean"],
                                         payload["target_std"], mode)
    return command.detach().cpu().numpy()


def act_proposal(model, config, preprocessor, postprocessor,
                 state: np.ndarray, frame: np.ndarray) -> np.ndarray:
    inputs = {"observation.state": torch.from_numpy(state.copy()),
              "observation.images.wrist": wrist_tensor(frame)}
    with torch.inference_mode():
        chunk = postprocessor(model.predict_action_chunk(preprocessor(inputs)))
    values = torch.as_tensor(chunk).detach().cpu().numpy()
    channels = config.output_features["action"].shape[0]
    if values.shape != (1, config.chunk_size, channels) or not np.isfinite(values).all():
        raise ValueError(f"ACT chunk has invalid shape or values: {values.shape}")
    print(f"ACT chunk: {config.chunk_size} predictions; queued action window: {config.n_action_steps}")
    print(f"ACT action {config.n_action_steps} arm:  ",
          np.round(values[0, config.n_action_steps - 1, :6], 2).tolist())
    if channels == 9:
        print(f"ACT action {config.n_action_steps} base (logged only):",
              dict(zip(BASE_KEYS, np.round(values[0, config.n_action_steps - 1, 6:], 4), strict=True)))
    return values[0, 0]


def live_observation(args, loaded) -> None:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

    robot = LeKiwiClient(LeKiwiClientConfig(
        remote_ip=args.robot_ip, id=args.robot_id,
        cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera_path,
                                               width=640, height=480, fps=30)},
    ))
    try:
        robot.connect()
        observation = robot.get_observation()
        state, frame = read_observation(observation)
        print("LIVE OBSERVATION: PASS")
        print("Observed arm/gripper:     ", np.round(state, 2).tolist())
        print(f"Wrist frame: {frame.shape}, {frame.dtype}; mean brightness: {frame.mean():.1f}/255")
        check_default_pose(args.default_pose, state, args.robot_id, args.start_tolerance)
        began = time.perf_counter()
        if args.policy == "act":
            act_state = act_state_from_observation(
                observation, loaded[1].input_features["observation.state"].shape[0])
            print("ACT input state:        ", np.round(act_state, 3).tolist())
            predicted = act_proposal(*loaded, act_state, frame)
            proposed = predicted[:6]
            if predicted.shape == (9,):
                print("Predicted base (ignored):",
                      dict(zip(BASE_KEYS, np.round(predicted[6:], 4), strict=True)))
        else:
            proposed = bc_proposal(*loaded[:3], state, frame)
        inference_ms = (time.perf_counter() - began) * 1000
        planned = full_lekiwi_action(proposed)
        print("First proposed arm target:", np.round(proposed, 2).tolist())
        print("Target minus observation: ", np.round(proposed - state, 2).tolist())
        print("Prospective base command:", {key: planned[key] for key in BASE_ZERO})
        print(f"Preprocess + inference + postprocess: {inference_ms:.1f} ms")
        print("NO ACTION SENT. This is a single-observation preflight, not a rollout.")
    finally:
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=(*BC_MODES, "act"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--live", action="store_true", help="Read one observation from the NUC host")
    parser.add_argument("--default-pose", type=Path,
                        help="Required with --live; captured default_pose.json on the NUC")
    parser.add_argument("--start-tolerance", type=float,
                        help="Override deployment.preflight_start_tolerance")
    parser.add_argument("--robot-ip", default="127.0.0.1")
    parser.add_argument("--robot-id", default="Mickey17_awesome_kiwi")
    parser.add_argument("--camera-path", default="/dev/video0")
    args = parser.parse_args()
    settings, settings_path, settings_sha256 = load_settings(args.config)
    if args.start_tolerance is None:
        args.start_tolerance = settings["deployment"]["preflight_start_tolerance"]
    if args.live and args.default_pose is None:
        parser.error("--live requires --default-pose")
    if not np.isfinite(args.start_tolerance) or args.start_tolerance <= 0:
        parser.error("--start-tolerance must be positive")

    checkpoint = args.checkpoint.expanduser().resolve()
    torch.set_num_threads(4)
    if args.policy == "act":
        loaded = load_act(checkpoint)
        weights = checkpoint / "model.safetensors"
        config = loaded[1]
        checkpoint_window = config.n_action_steps
        runtime_window = apply_act_window(loaded[0], config, settings)
        channels = config.input_features["observation.state"].shape[0]
        print(f"ACT features: wrist (3, 480, 640) + {channels} state values → "
              f"{channels} predictions; base commands fixed to zero")
        print(f"ACT chunk: {config.chunk_size}; checkpoint window: {checkpoint_window}; "
              f"runtime window: {runtime_window}; "
              f"label offset: {getattr(config, 'action_offset_frames', 0)} frames")
    else:
        loaded = load_bc(checkpoint, args.policy)
        weights = checkpoint
        print(f"BC features: wrist (3, 120, 160) + six positions → six commands")
        print(f"BC target offset: {loaded[3]} frames")
    print("CHECKPOINT: PASS")
    print(f"Settings: {settings_path}; SHA-256: {settings_sha256}")
    print(f"Policy: {args.policy}; path: {checkpoint}")
    print(f"Weights SHA-256: {sha256_file(weights)}")
    print("Joint order:", ", ".join(ARM_KEYS))
    print("Base command mapping:", BASE_ZERO)
    if args.live:
        live_observation(args, loaded)
    else:
        print("Checkpoint-only mode: no robot connection or motion.")


if __name__ == "__main__":
    main()
