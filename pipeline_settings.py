"""Load the shared policy settings without touching datasets or hardware."""

from __future__ import annotations

import hashlib
import math
import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("pipeline_config.toml")


def _positive_number(value: object, name: str, integer: bool = False) -> None:
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


def load_settings(path: Path | None = None) -> tuple[dict, Path, str]:
    """Return validated settings, resolved path, and SHA-256 of exact TOML bytes."""
    source = (path or DEFAULT_CONFIG).expanduser().resolve()
    raw = source.read_bytes()
    settings = tomllib.loads(raw.decode("utf-8"))
    if settings.get("format_version") != 1:
        raise ValueError("Unsupported pipeline_config.toml format_version")
    try:
        bc_model = settings["bc"]["model"]
        bc_train = settings["bc"]["train"]
        act_train = settings["act"]["train"]
        act_runtime = settings["act"]["runtime"]
        deployment = settings["deployment"]
    except KeyError as error:
        raise ValueError(f"Missing settings section: {error}") from error
    for section_name, section, integer_names in (
        ("bc.model", bc_model, ("image_height", "image_width", "future_offset_frames")),
        ("bc.train", bc_train, ("epochs", "batch_size")),
        ("act.train", act_train, ("steps", "batch_size", "chunk_size", "n_action_steps",
                              "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers",
                              "n_vae_encoder_layers", "eval_steps", "max_eval_samples",
                              "log_freq", "save_freq")),
        ("act.runtime", act_runtime, ("action_window", "cpu_threads")),
    ):
        for key in integer_names:
            _positive_number(section[key], f"{section_name}.{key}", integer=True)
    if type(act_train["num_workers"]) is not int or act_train["num_workers"] < 0:
        raise ValueError("act.train.num_workers must be a nonnegative integer")
    for section_name, section, keys in (
        ("bc.train", bc_train, ("learning_rate",)),
        ("act.train", act_train, ("optimizer_lr", "optimizer_lr_backbone")),
        ("deployment", deployment, ("control_hz", "preflight_start_tolerance",
                                    "shadow_seconds", "shadow_start_tolerance",
                                    "shadow_leader_start_tolerance", "shadow_max_tracking_gap",
                                    "bc_raw_start_tolerance")),
    ):
        for key in keys:
            _positive_number(section[key], f"{section_name}.{key}")
    snapshot = deployment["shadow_snapshot_every_s"]
    if type(snapshot) not in (int, float) or not math.isfinite(snapshot) or snapshot < 0:
        raise ValueError("deployment.shadow_snapshot_every_s must be nonnegative")
    # These values are part of the current BC artifact contract. Changing
    # them needs a corresponding code/checkpoint/review migration.
    if (bc_model["image_height"], bc_model["image_width"],
            bc_model["future_offset_frames"]) != (120, 160, 3):
        raise ValueError("Current BC pipeline requires image 120x160 and future offset 3")
    if act_train["n_action_steps"] > act_train["chunk_size"]:
        raise ValueError("act.train.n_action_steps exceeds act.train.chunk_size")
    if act_train["dim_model"] % act_train["n_heads"]:
        raise ValueError("act.train.dim_model must be divisible by n_heads")
    if deployment["control_hz"] != 30:
        raise ValueError("Current demonstrations and action chunks require 30 Hz control")
    return settings, source, hashlib.sha256(raw).hexdigest()


def apply_act_window(policy, checkpoint_config, settings: dict) -> int:
    """Apply a runtime queue length after loading ACT; leave files unchanged."""
    window = settings["act"]["runtime"]["action_window"]
    if not 1 <= window <= checkpoint_config.chunk_size:
        raise ValueError(f"ACT runtime window {window} must be within checkpoint chunk "
                         f"1..{checkpoint_config.chunk_size}")
    if checkpoint_config.temporal_ensemble_coeff is not None and window != 1:
        raise ValueError("Temporal ensembling requires a one-action runtime window")
    policy.config.n_action_steps = window
    policy.reset()
    return window
