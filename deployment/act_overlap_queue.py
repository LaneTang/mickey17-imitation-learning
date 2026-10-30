"""Time-align and blend overlapping ACT chunks without connecting to a robot.

LeRobot's async client aggregates predictions for the same future timestep.
Here the new chunk's weight ramps up over several control cycles so its first
available action cannot replace the previous chunk all at once. A separate
motion limiter is still required before any result is sent to the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QueueUpdate:
    source_cycle: int
    ready_cycle: int
    accepted: int
    merged: int
    expired: int


class OverlapActionQueue:
    def __init__(self, chunk_size: int, action_dim: int = 6,
                 new_weight: float = 0.7, blend_cycles: int = 6,
                 action_offset_frames: int = 0) -> None:
        if (chunk_size < 2 or action_dim < 1 or blend_cycles < 1 or
                type(action_offset_frames) is not int or action_offset_frames < 0):
            raise ValueError("Invalid ACT queue dimensions or blend duration")
        if not 0 < new_weight <= 1:
            raise ValueError("New-chunk weight must be in (0, 1]")
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.new_weight = new_weight
        self.blend_cycles = blend_cycles
        self.action_offset_frames = action_offset_frames
        self._future: dict[int, np.ndarray] = {}
        self._contributors: dict[int, int] = {}
        self._first_chunk_seen = False
        self._last_source_cycle = -1

    @property
    def queued_steps(self) -> int:
        return len(self._future)

    @property
    def next_cycle(self) -> int | None:
        return min(self._future) if self._future else None

    def add(self, actions: np.ndarray, source_cycle: int, ready_cycle: int) -> QueueUpdate:
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.chunk_size, self.action_dim) or not np.isfinite(values).all():
            raise ValueError("ACT chunk must have the expected finite shape")
        if source_cycle < 0 or ready_cycle < 0 or ready_cycle < source_cycle:
            raise ValueError("Invalid ACT source or ready cycle")
        if source_cycle <= self._last_source_cycle:
            raise ValueError("ACT chunks must arrive in source-cycle order")

        # Standard ACT's first chunk began when it became ready, because the
        # robot held its start pose during boot inference. A shifted-label
        # checkpoint predicts absolute future steps from its observation:
        # source + offset + step. Preserve that schedule even for chunk one.
        anchor = (ready_cycle if not self._first_chunk_seen and self.action_offset_frames == 0
                  else source_cycle + self.action_offset_frames)
        accepted = merged = expired = 0
        for step, new_action in enumerate(values):
            target_cycle = anchor + step
            if target_cycle < ready_cycle:
                expired += 1
                continue
            if target_cycle in self._future:
                lead = target_cycle - ready_cycle
                weight = self.new_weight * min((lead + 1) / self.blend_cycles, 1.0)
                old_action = self._future[target_cycle]
                self._future[target_cycle] = (1.0 - weight) * old_action + weight * new_action
                self._contributors[target_cycle] += 1
                merged += 1
            else:
                self._future[target_cycle] = new_action.copy()
                self._contributors[target_cycle] = 1
            accepted += 1
        self._first_chunk_seen = True
        self._last_source_cycle = source_cycle
        return QueueUpdate(source_cycle, ready_cycle, accepted, merged, expired)

    def pop(self, cycle: int, hold_position: np.ndarray) -> tuple[np.ndarray, int]:
        hold = np.asarray(hold_position, dtype=np.float32)
        if hold.shape != (self.action_dim,) or not np.isfinite(hold).all():
            raise ValueError("Hold position must have the expected finite shape")
        if cycle < 0:
            raise ValueError("Cycle must be nonnegative")
        action = self._future.pop(cycle, None)
        contributors = self._contributors.pop(cycle, 0)
        for stale in [key for key in self._future if key < cycle]:
            del self._future[stale]
            del self._contributors[stale]
        if action is None:
            return hold.copy(), 0
        return action.copy(), contributors
