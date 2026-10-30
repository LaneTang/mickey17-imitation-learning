"""Shared arm-only policy input and output layout for Mikey17.

These functions only transform data. They never connect to or command a robot.
"""

from __future__ import annotations

import numpy as np
import torch

ARM_KEYS = (
    "arm_shoulder_pan.pos",
    "arm_shoulder_lift.pos",
    "arm_elbow_flex.pos",
    "arm_wrist_flex.pos",
    "arm_wrist_roll.pos",
    "arm_gripper.pos",
)
BASE_ZERO = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
BASE_KEYS = tuple(BASE_ZERO)
WRIST_SHAPE = (480, 640, 3)


def read_observation(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return six measured positions and the raw HWC uint8 wrist frame."""
    missing = [key for key in ARM_KEYS if key not in observation]
    if missing:
        raise ValueError(f"Missing arm observation fields: {missing}")
    state = np.asarray([observation[key] for key in ARM_KEYS], dtype=np.float32)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Arm observation must contain six finite positions")
    frame = observation.get("wrist")
    if not isinstance(frame, np.ndarray) or frame.shape != WRIST_SHAPE or frame.dtype != np.uint8:
        raise ValueError(f"Wrist frame must be uint8 {WRIST_SHAPE}; got "
                         f"{getattr(frame, 'shape', None)}, {getattr(frame, 'dtype', None)}")
    return state, frame


def act_state_from_observation(observation: dict, channels: int) -> np.ndarray:
    """Build ACT's state in LeKiwi feature order; retain six-channel compatibility."""
    if channels not in (6, 9):
        raise ValueError(f"Unsupported ACT state width: {channels}")
    keys = ARM_KEYS if channels == 6 else ARM_KEYS + BASE_KEYS
    missing = [key for key in keys if key not in observation]
    if missing:
        raise ValueError(f"Missing ACT observation fields: {missing}")
    state = np.asarray([observation[key] for key in keys], dtype=np.float32)
    if state.shape != (channels,) or not np.isfinite(state).all():
        raise ValueError(f"ACT state must contain {channels} finite values")
    return state


def wrist_tensor(frame: np.ndarray) -> torch.Tensor:
    """Match training's RGB/channel order without swapping camera channels."""
    if frame.shape != WRIST_SHAPE or frame.dtype != np.uint8:
        raise ValueError("Unexpected wrist frame")
    return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float().div(255)


def full_lekiwi_action(arm_positions: np.ndarray) -> dict[str, float]:
    """Map six policy targets to the LeKiwi command schema with a stopped base.

    Constructing this dictionary does not send it. Later shadow and guarded
    rollout code can import this single mapping rather than duplicating it.
    """
    values = np.asarray(arm_positions, dtype=np.float32)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Policy target must have six finite joint positions")
    return {**dict(zip(ARM_KEYS, map(float, values), strict=True)), **BASE_ZERO}
