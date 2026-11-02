"""Shared training implementation for the three named BC entry scripts.

All modes use the same small CNN and the shared whole-episode split. Training
starts only with --train. Nothing in this file connects to a robot.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import load_experiment
from pipeline_settings import load_settings
from bc_model import VanillaBC
from bc_targets import MODES, action_from_prediction, frame_indices, horizon_frames, labels

JOINTS = ("arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
          "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos")
RUN_NAMES = {"direct": "bc_direct", "residual": "bc_residual",
             "future-residual": "bc_future_residual3"}
LABEL_NAMES = {"direct": "direct_same_frame", "residual": "residual_same_frame",
               "future-residual": "future_residual_3_frames"}


def read_columns(source) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rows = source.hf_dataset
    states = torch.as_tensor(np.asarray(rows["observation.state"]), dtype=torch.float32)[:, :6]
    actions = torch.as_tensor(np.asarray(rows["action"]), dtype=torch.float32)[:, :6]
    episode_ids = np.asarray(rows["episode_index"], dtype=np.int64)
    if len(states) != len(source) or len(actions) != len(source):
        raise ValueError("Numerical rows do not match the decoded dataset length")
    return states, actions, episode_ids


class BCDataset(Dataset):
    def __init__(self, source, indices: np.ndarray, states: torch.Tensor,
                 actions: torch.Tensor, state_mean: torch.Tensor, state_std: torch.Tensor,
                 target_mean: torch.Tensor, target_std: torch.Tensor, mode: str):
        self.source = source
        self.indices = indices
        self.states = states
        self.commands = actions[indices + horizon_frames(mode)]
        self.normalized_targets = (labels(states, actions, indices, mode) - target_mean) / target_std
        self.state_mean, self.state_std = state_mean, state_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        frame = int(self.indices[index])
        row = self.source[frame]
        image = F.interpolate(row["observation.images.wrist"].unsqueeze(0),
                              size=(120, 160), mode="bilinear", align_corners=False).squeeze(0)
        state = self.states[frame]
        return image, (state - self.state_mean) / self.state_std, \
            self.normalized_targets[index], state, self.commands[index]


def main(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Unknown BC mode: {mode}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--epochs", type=int, help="Override bc.train.epochs")
    parser.add_argument("--batch-size", type=int, help="Override bc.train.batch_size")
    parser.add_argument("--train", action="store_true", help="Start training; omitted means preview only")
    args = parser.parse_args()
    settings, settings_path, settings_sha256 = load_settings(args.config)
    args.epochs = args.epochs if args.epochs is not None else settings["bc"]["train"]["epochs"]
    args.batch_size = (args.batch_size if args.batch_size is not None
                       else settings["bc"]["train"]["batch_size"])
    learning_rate = settings["bc"]["train"]["learning_rate"]
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive")
    config = load_experiment(args.experiment)
    info = json.loads((Path(config["dataset_root"]) / "meta" / "info.json").read_text())
    if info["features"]["action"]["names"][:6] != list(JOINTS):
        raise ValueError("The first six action channels are not the expected arm/gripper order")
    if info["features"]["observation.state"]["names"][:6] != list(JOINTS):
        raise ValueError("The first six state channels are not the expected arm/gripper order")
    run_name = RUN_NAMES[mode]
    checkpoint_name = f"{run_name}_best.pt"
    print(f"BC mode: {mode}; horizon: {horizon_frames(mode)} frames")
    print(f"Checkpoint name: {checkpoint_name}")
    print(f"Settings: {settings_path}; SHA-256: {settings_sha256}")
    print(f"BC training: epochs={args.epochs}, batch_size={args.batch_size}, lr={learning_rate}")
    print(f"Dataset: {config['total_episodes']} episodes, {config['total_frames']} frames")
    print(f"Training IDs: {config['train_episodes']}")
    print(f"Validation IDs: {config['validation_episodes']}")
    if not args.train:
        print("PREVIEW ONLY. Add --train to start training.")
        return

    from lerobot.datasets import LeRobotDataset

    torch.manual_seed(0)
    root = Path(config["dataset_root"])
    train_source = LeRobotDataset(config["repo_id"], root=root,
                                 episodes=config["train_episodes"], video_backend="pyav")
    val_source = LeRobotDataset(config["repo_id"], root=root,
                               episodes=config["validation_episodes"], video_backend="pyav")
    train_states, train_actions, train_episodes = read_columns(train_source)
    val_states, val_actions, val_episodes = read_columns(val_source)
    train_indices = frame_indices(train_episodes, mode)
    val_indices = frame_indices(val_episodes, mode)
    if len(train_indices) < 2 or len(val_indices) == 0:
        raise ValueError("Not enough within-episode frames for this label mode")

    selected_states = train_states[train_indices]
    train_labels = labels(train_states, train_actions, train_indices, mode)
    state_mean = selected_states.mean(0)
    state_std = selected_states.std(0).clamp_min(1e-6)
    target_mean = train_labels.mean(0)
    target_std = train_labels.std(0).clamp_min(1e-6)
    train_data = BCDataset(train_source, train_indices, train_states, train_actions,
                           state_mean, state_std, target_mean, target_std, mode)
    val_data = BCDataset(val_source, val_indices, val_states, val_actions,
                         state_mean, state_std, target_mean, target_std, mode)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VanillaBC().to(device)
    if mode != "direct":
        # A zero network output initially means observed pose + average train correction.
        nn.init.zeros_(model.head[-1].weight)
        nn.init.zeros_(model.head[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)
    run_dir = (Path(__file__).resolve().parents[1] / "runs" /
               datetime.now().strftime(f"{run_name}_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(settings_path, run_dir / "pipeline_config.toml")
    (run_dir / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Device: {device}; train frames: {len(train_data)}; "
          f"validation frames: {len(val_data)}; output: {run_dir}", flush=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for image, state, target, _, _ in train_loader:
            image, state, target = image.to(device), state.to(device), target.to(device)
            loss = nn.functional.mse_loss(model(image, state), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(image)
        model.eval()
        total_error = torch.zeros(6, dtype=torch.float64)
        hold_error = torch.zeros(6, dtype=torch.float64)
        frames = 0
        with torch.inference_mode():
            for image, state, _, raw_state, command in val_loader:
                image, state = image.to(device), state.to(device)
                predicted = action_from_prediction(model(image, state), raw_state.to(device),
                                                   target_mean_device, target_std_device, mode)
                total_error += (predicted.cpu() - command).abs().sum(0).double()
                hold_error += (raw_state - command).abs().sum(0).double()
                frames += len(image)
        per_joint = (total_error / frames).tolist()
        row = {"epoch": epoch, "label_mode": LABEL_NAMES[mode],
               "horizon_frames": horizon_frames(mode), "validation_frames": frames,
               "train_normalized_mse": train_loss / len(train_data),
               "validation_mae": float(np.mean(per_joint)),
               "validation_per_joint_mae": dict(zip(JOINTS, per_joint, strict=True)),
               "hold_position_baseline_mae": float((hold_error / frames).mean())}
        history.append(row)
        print(f"Epoch {epoch}: validation MAE={row['validation_mae']:.3f}; "
              f"gripper={per_joint[5]:.3f}; hold={row['hold_position_baseline_mae']:.3f}", flush=True)
        if row["validation_mae"] < best:
            best = row["validation_mae"]
            checkpoint = {"model": model.state_dict(), "state_mean": state_mean,
                          "state_std": state_std, "target_mean": target_mean,
                          "target_std": target_std, "label_mode": LABEL_NAMES[mode],
                          "horizon_frames": horizon_frames(mode),
                          "image_size": [120, 160], "joint_order": JOINTS,
                          "experiment": config, "epoch": epoch}
            if mode == "direct":
                checkpoint.update(action_mean=target_mean, action_std=target_std)
            else:
                checkpoint.update(delta_mean=target_mean, delta_std=target_std)
            torch.save(checkpoint, run_dir / checkpoint_name)
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Best validation MAE: {best:.3f}; checkpoint: {run_dir / checkpoint_name}")
