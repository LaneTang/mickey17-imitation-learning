"""Episode-safe labels and action reconstruction for the three BC runs."""

import numpy as np
import torch

MODES = ("direct", "residual", "future-residual")
FUTURE_HORIZON = 3


def horizon_frames(mode: str) -> int:
    if mode not in MODES:
        raise ValueError(f"Unknown BC mode: {mode}")
    return FUTURE_HORIZON if mode == "future-residual" else 0


def frame_indices(episode_ids: np.ndarray, mode: str) -> np.ndarray:
    """Keep only frames whose future command belongs to the same episode."""
    episode_ids = np.asarray(episode_ids)
    horizon = horizon_frames(mode)
    if horizon == 0:
        return np.arange(len(episode_ids), dtype=np.int64)
    current = np.arange(max(0, len(episode_ids) - horizon), dtype=np.int64)
    return current[episode_ids[current] == episode_ids[current + horizon]]


def labels(states: torch.Tensor, actions: torch.Tensor,
           indices: np.ndarray, mode: str) -> torch.Tensor:
    """Actions are recorded leader targets; states are measured follower poses."""
    horizon = horizon_frames(mode)
    command = actions[indices + horizon]
    if mode == "direct":
        return command
    return command - states[indices]


def action_from_prediction(prediction: torch.Tensor, observed_state: torch.Tensor,
                           target_mean: torch.Tensor, target_std: torch.Tensor,
                           mode: str) -> torch.Tensor:
    """Convert a normalized network output into an absolute six-joint command."""
    correction_or_action = target_mean + prediction * target_std
    return correction_or_action if mode == "direct" else observed_state + correction_or_action
