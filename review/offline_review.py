"""Read-only offline review of patch-4 BC and ACT checkpoints.

Screens every ACT checkpoint on the same held-out observations, evaluates the
three saved BC models, and draws time-aligned command plots for two episodes.
All predictions use recorded observations (teacher forcing). No robot connection.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "train"))

from experiment import load_experiment
from bc_model import VanillaBC
from bc_targets import action_from_prediction
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

JOINTS = ("shoulder pan", "shoulder lift", "elbow flex", "wrist flex", "wrist roll", "gripper")
COLORS = {"recorded": "#172b4d", "direct": "#38a6a5", "residual": "#ef9f3c",
          "future_residual": "#9b78c7", "act": "#e36a70"}
BC_RUNS = {
    "future_residual": ("bc_future_residual3_20261013_010925", "bc_future_residual3_best.pt"),
}
ACT_RUN = "act_arm_only_20261011_051140"


@dataclass
class BCModel:
    mode: str
    network: VanillaBC
    mean: torch.Tensor
    std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    horizon: int
    checkpoint: Path


def device_name(value: str) -> str:
    if value != "auto":
        if value == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        if value == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        return value
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def check_run_experiment(run: Path, config: dict, checkpoint: Path | None = None) -> None:
    manifest = run / "experiment.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        for key in ("train_episodes", "validation_episodes"):
            if saved[key] != config[key]:
                raise ValueError(f"Different {key} in {run}")
        source = saved.get("source_dataset_root", saved["dataset_root"])
        if Path(source).resolve() != Path(config["dataset_root"]).resolve():
            raise ValueError(f"Different source dataset in {run}")
        return

    if checkpoint is None:
        raise FileNotFoundError(f"No experiment manifest in {run}")
    training_config = checkpoint / "train_config.json"
    if not training_config.is_file():
        raise FileNotFoundError(f"No experiment manifest or training config in {run}")
    saved = json.loads(training_config.read_text())["dataset"]
    ordered = config["train_episodes"] + config["validation_episodes"]
    if saved["episodes"] != ordered:
        raise ValueError(f"Different training/validation episode order in {run}")
    if math.ceil(len(ordered) * saved["eval_split"]) != len(config["validation_episodes"]):
        raise ValueError(f"Different validation split in {run}")
    dataset_root = Path(saved["root"])
    if not dataset_root.is_absolute():
        dataset_root = PIPELINE / dataset_root
    if (dataset_root.resolve() != Path(config["dataset_root"]).resolve()
            or saved["repo_id"] != config["repo_id"]):
        raise ValueError(f"Different source dataset in {run}")


def load_bc(mode: str, run_root: Path, config: dict, device: str) -> BCModel:
    folder, filename = BC_RUNS[mode]
    run = run_root / folder
    check_run_experiment(run, config)
    checkpoint = run / filename
    # Only load checkpoints made by this local training script.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_horizon = 3 if mode == "future_residual" else 0
    if payload["horizon_frames"] != expected_horizon or list(payload["experiment"]["validation_episodes"]) != config["validation_episodes"]:
        raise ValueError(f"Unexpected BC checkpoint metadata: {checkpoint}")
    network = VanillaBC().to(device).eval()
    network.load_state_dict(payload["model"])
    label_mode = "future-residual" if mode == "future_residual" else mode
    return BCModel(label_mode, network, payload["target_mean"].to(device),
                   payload["target_std"].to(device), payload["state_mean"].to(device),
                   payload["state_std"].to(device), expected_horizon, checkpoint)


def sample_positions(episode_ids: np.ndarray, episodes: list[int], per_episode: int) -> tuple[list[int], list[int]]:
    indices, owners = [], []
    for ep in episodes:
        positions = np.flatnonzero(episode_ids == ep)
        if not len(positions):
            raise ValueError(f"Missing episode {ep}")
        count = min(per_episode, len(positions))
        chosen = positions[np.linspace(0, len(positions) - 1, count, dtype=int)]
        indices.extend(chosen.tolist())
        owners.extend([ep] * len(chosen))
    return indices, owners


def read_numbers(dataset: LeRobotDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = dataset.hf_dataset
    return (np.asarray(rows["observation.state"], dtype=np.float32)[:, :6],
            np.asarray(rows["action"], dtype=np.float32)[:, :6],
            np.asarray(rows["episode_index"], dtype=np.int64))


def predict_bc(model: BCModel, dataset: LeRobotDataset, states: np.ndarray,
               indices: list[int], device: str, batch_size: int) -> tuple[np.ndarray, float]:
    outputs = []
    durations = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            block = indices[start:start + batch_size]
            images = torch.stack([F.interpolate(dataset[i]["observation.images.wrist"].unsqueeze(0),
                                                 size=(120, 160), mode="bilinear",
                                                 align_corners=False).squeeze(0) for i in block]).to(device)
            state = torch.as_tensor(states[block], device=device)
            normalized = (state - model.state_mean) / model.state_std
            synchronize(device)
            began = time.perf_counter()
            prediction = model.network(images, normalized)
            actions = action_from_prediction(prediction, state, model.mean, model.std, model.mode)
            synchronize(device)
            durations.append((time.perf_counter() - began) * 1000 / len(block))
            outputs.append(actions.cpu().numpy())
    return np.concatenate(outputs), float(np.median(durations))


def load_act(path: Path, device: str):
    config = ACTConfig.from_pretrained(path)
    state_dim = tuple(config.input_features["observation.state"].shape)
    action_dim = tuple(config.output_features["action"].shape)
    if state_dim not in ((6,), (9,)) or action_dim != state_dim:
        raise ValueError(f"ACT checkpoint must have matching 6- or 9-channel state/action: {path}")
    config.device = device
    policy = ACTPolicy.from_pretrained(path, config=config).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}})
    return policy, preprocessor, postprocessor


def predict_act(policy, preprocessor, postprocessor, dataset: LeRobotDataset,
                indices: list[int], device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    predictions, targets, masks, durations = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            block = indices[start:start + batch_size]
            rows = [dataset[i] for i in block]
            images = torch.stack([row["observation.images.wrist"] for row in rows]).float() / 255.0
            states = torch.stack([row["observation.state"] for row in rows])
            inputs = {"observation.images.wrist": images, "observation.state": states}
            synchronize(device)
            began = time.perf_counter()
            output = postprocessor(policy.predict_action_chunk(preprocessor(inputs)))
            synchronize(device)
            durations.append((time.perf_counter() - began) * 1000 / len(block))
            # The nine-channel LeKiwi policy also predicts three locked base
            # velocities. Compare the same six arm/gripper targets as BC.
            predictions.append(output[:, :10, :6].cpu().numpy())
            targets.append(torch.stack([row["action"][:10, :6] for row in rows]).numpy())
            masks.append(~torch.stack([row["action_is_pad"][:10] for row in rows]).numpy())
    return (np.concatenate(predictions), np.concatenate(targets),
            np.concatenate(masks), float(np.median(durations)))


def summarize(predicted: np.ndarray, target: np.ndarray, owners: list[int],
              mask: np.ndarray | None = None) -> dict:
    if predicted.ndim == 2:
        predicted, target = predicted[:, None, :], target[:, None, :]
    if mask is None:
        mask = np.ones(predicted.shape[:2], dtype=bool)
    error = np.abs(predicted - target)
    valid = error[mask]
    by_episode = {}
    for ep in sorted(set(owners)):
        selected = np.asarray(owners) == ep
        by_episode[str(ep)] = float(error[selected][mask[selected]].mean())
    grip_target, grip_prediction = target[..., 5][mask], predicted[..., 5][mask]
    # A grasp command can settle above the fully-closed pose while holding the
    # cube. Compare high vs low aperture at 15, not "fully closed" vs "open".
    state_accuracy = float(np.mean((grip_prediction >= 15) == (grip_target >= 15)))
    return {"mae": float(valid.mean()), "per_joint_mae": {name: float(value) for name, value in zip(JOINTS, valid.mean(0))},
            "per_joint_p95": {name: float(value) for name, value in zip(JOINTS, np.percentile(valid, 95, axis=0))},
            "episodes": by_episode, "gripper_high_low_accuracy": state_accuracy,
            "gripper_classified_frames": int(len(grip_target)), "compared_actions": int(mask.sum())}


def episode_positions(episode_ids: np.ndarray, episode: int) -> np.ndarray:
    return np.flatnonzero(episode_ids == episode)


def median_smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    from scipy.ndimage import median_filter
    return median_filter(values, size=width, mode="nearest")


def gripper_events(values: np.ndarray, fps: float) -> dict[str, list[float]]:
    smooth = median_smooth(values)
    opening = np.flatnonzero((smooth[:-1] < 15) & (smooth[1:] >= 15)) + 1
    closing = np.flatnonzero((smooth[:-1] >= 15) & (smooth[1:] < 15)) + 1
    return {"open": (opening / fps).tolist(), "close": (closing / fps).tolist()}


def event_match(reference: list[float], prediction: list[float], tolerance_s: float = 2.0) -> dict:
    remaining = set(range(len(prediction)))
    offsets = []
    for observed in reference:
        if not remaining:
            break
        match = min(remaining, key=lambda i: abs(prediction[i] - observed))
        if abs(prediction[match] - observed) <= tolerance_s:
            offsets.append(prediction[match] - observed)
            remaining.remove(match)
    return {"matched": len(offsets), "reference_count": len(reference),
            "predicted_count": len(prediction), "median_signed_delay_s": float(np.median(offsets)) if offsets else None,
            "median_absolute_delay_s": float(np.median(np.abs(offsets))) if offsets else None}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.edgecolor": "#b8c2d1",
                         "grid.color": "#e5eaf1", "figure.facecolor": "#f8fafc",
                         "axes.facecolor": "#ffffff", "savefig.facecolor": "#f8fafc"})


def plot_scorecard(rows: list[dict], output: Path) -> None:
    style()
    bc = [row for row in rows if row["family"] == "BC"]
    act = [row for row in rows if row["family"] == "ACT"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, subset, title in ((axes[0], bc, "BC checkpoints · future model targets t+3"),
                              (axes[1], act, "ACT checkpoints")):
        labels = [r["name"] for r in subset]
        values = [r["first_action"]["mae"] for r in subset]
        colors = [COLORS.get(r["name"], COLORS["act"]) for r in subset]
        ax.barh(labels, values, color=colors, height=.62)
        for i, value in enumerate(values):
            ax.text(value + max(values) * .025, i, f"{value:.2f}", va="center", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.22)
        ax.set_xlabel("First-action MAE · encoder position units")
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="x", alpha=.8)
        ax.set_axisbelow(True)
    fig.suptitle("Held-out checkpoint screen", fontsize=17, fontweight="bold", x=.04, ha="left")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_episode(episode: int, fps: float, truth: np.ndarray, traces: dict[str, np.ndarray], output: Path) -> None:
    style()
    seconds = np.arange(len(truth)) / fps
    fig, axes = plt.subplots(3, 2, figsize=(17, 12), sharex=True, layout="constrained")
    for joint, ax in enumerate(axes.flat):
        ax.plot(seconds, truth[:, joint], label="recorded command", color=COLORS["recorded"], lw=2.3, zorder=5)
        for name, values in traces.items():
            ax.plot(seconds, values[:, joint], label=name.replace("_", " "), color=COLORS[name],
                    lw=1.25, alpha=.85)
        ax.set_title(JOINTS[joint].title(), loc="left")
        ax.grid(alpha=.8)
        ax.set_ylabel("encoder position")
    for ax in axes[-1]:
        ax.set_xlabel("seconds from episode start")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=5, frameon=False)
    fig.suptitle(f"Episode {episode:03d}  ·  time-aligned joint commands", y=1.055, fontsize=19, fontweight="bold")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_gripper(episode: int, fps: float, truth: np.ndarray, traces: dict[str, np.ndarray], output: Path) -> None:
    style()
    seconds = np.arange(len(truth)) / fps
    fig, ax = plt.subplots(figsize=(16, 5), layout="constrained")
    ax.axhspan(-5, 5, color="#eef6f5", alpha=.7, label="fully closed zone (≤5)")
    ax.axhline(15, color="#9aa8b7", lw=1, ls="--", label="high/low aperture threshold (15)")
    ax.plot(seconds, truth[:, 5], color=COLORS["recorded"], lw=3, label="recorded", zorder=6)
    for name, values in traces.items():
        ax.plot(seconds, values[:, 5], color=COLORS[name], lw=1.7, alpha=.85, label=name.replace("_", " "))
    for moment in gripper_events(truth[:, 5], fps)["close"]:
        ax.axvline(moment, color="#172b4d", ls=":", lw=1, alpha=.45)
    ax.set(xlabel="seconds from episode start", ylabel="gripper encoder position",
           title=f"Episode {episode:03d}  ·  grasp and release timing")
    ax.grid(alpha=.8)
    ax.legend(loc="upper right", ncol=3, frameon=False)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_storyboard(episode: int, fps: float, truth: np.ndarray, dataset: LeRobotDataset,
                    positions: np.ndarray, output: Path) -> None:
    style()
    events = gripper_events(truth[:, 5], fps)
    middle = round(len(positions) / 2)
    center = round(events["close"][0] * fps) if events["close"] else middle
    samples = np.clip([center - 20, center - 10, center, center + 10, center + 20], 0, len(positions) - 1)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), layout="constrained")
    for ax, local in zip(axes, samples):
        row = dataset[int(positions[local])]
        image = row["observation.images.wrist"].permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(f"t={local/fps:.1f}s  ·  grip={truth[local, 5]:.1f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Episode {episode:03d}  ·  recorded wrist frames near first low-aperture command", fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_episodes(plot_episodes: list[int], fps: float, bc_data: LeRobotDataset,
                       act_data: LeRobotDataset, states: np.ndarray, actions: np.ndarray,
                       episode_ids: np.ndarray, models: dict[str, BCModel], act_model,
                       device: str, batch_size: int, output: Path,
                       act_action_offset_frames: int = 0) -> dict:
    policy, preprocessor, postprocessor = act_model
    timing = {}
    for ep in plot_episodes:
        positions = episode_positions(episode_ids, ep)
        truth = actions[positions].copy()
        traces = {}
        for name, model in models.items():
            eligible = positions[:-model.horizon] if model.horizon else positions
            predicted, _ = predict_bc(model, bc_data, states, eligible.tolist(), device, batch_size)
            curve = np.full_like(truth, np.nan)
            curve[model.horizon:model.horizon + len(predicted)] = predicted
            traces[name] = curve
        act_curve = np.full_like(truth, np.nan)
        for start in range(0, len(positions), 10):
            predicted, _, _, _ = predict_act(policy, preprocessor, postprocessor, act_data,
                                              [int(positions[start])], device, 1)
            target_start = start + act_action_offset_frames
            end = min(target_start + 10, len(positions))
            if target_start < end:
                act_curve[target_start:end] = predicted[0, :end - target_start]
        traces["act"] = act_curve
        plot_episode(ep, fps, truth, traces, output / f"episode_{ep:03d}_joints.png")
        plot_gripper(ep, fps, truth, traces, output / f"episode_{ep:03d}_gripper.png")
        plot_storyboard(ep, fps, truth, bc_data, positions, output / f"episode_{ep:03d}_wrist.png")
        reference = gripper_events(truth[:, 5], fps)
        timing[str(ep)] = {name: {kind: event_match(reference[kind], gripper_events(curve[:, 5], fps)[kind])
                                  for kind in ("open", "close")} for name, curve in traces.items()}
        print(f"Plotted episode {ep}: joints, gripper, wrist frames", flush=True)
    return timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=PIPELINE / "experiment.json")
    parser.add_argument("--run-root", type=Path, default=PIPELINE / "runs")
    parser.add_argument("--act-run", type=Path, default=PIPELINE / "runs" / ACT_RUN)
    parser.add_argument("--output", type=Path, default=PIPELINE / "reviews" / "patch4_offline_v1")
    parser.add_argument("--samples-per-episode", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--plot-episodes", type=int, nargs="*")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite review: {args.output}")
    if args.samples_per_episode < 1 or args.batch_size < 1:
        parser.error("Samples and batch size must be positive")
    config = load_experiment(args.experiment)
    device = device_name(args.device)
    validation = config["validation_episodes"]
    selected = args.plot_episodes if args.plot_episodes is not None else [validation[0], validation[len(validation) // 2]]
    if any(ep not in validation for ep in selected) or len(set(selected)) != len(selected):
        parser.error("Plot episodes must be unique validation episode IDs")
    checkpoints = sorted(p / "pretrained_model" for p in (args.act_run / "checkpoints").iterdir()
                         if p.is_dir() and p.name.isdigit() and (p / "pretrained_model/model.safetensors").is_file())
    if not checkpoints:
        raise FileNotFoundError("No complete ACT checkpoints")
    check_run_experiment(args.act_run, config, checkpoints[0])
    full = LeRobotDataset(config["repo_id"], root=config["dataset_root"], episodes=validation,
                          video_backend="pyav")
    states, commands, episode_ids = read_numbers(full)
    indices, owners = sample_positions(episode_ids, validation, args.samples_per_episode)
    print(f"Held-out episodes: {len(validation)}; screening observations: {len(indices)}; device: {device}", flush=True)
    models = {name: load_bc(name, args.run_root, config, device) for name in BC_RUNS}
    rows = []
    for name, model in models.items():
        eligible = [(i, ep) for i, ep in zip(indices, owners) if i + model.horizon < len(episode_ids)
                    and episode_ids[i] == episode_ids[i + model.horizon]]
        kept, labels = [item[0] for item in eligible], [item[1] for item in eligible]
        predicted, ms = predict_bc(model, full, states, kept, device, args.batch_size)
        target = commands[np.asarray(kept) + model.horizon]
        score = summarize(predicted, target, labels)
        rows.append({"family": "BC", "name": name, "checkpoint": str(model.checkpoint),
                     "target_offset_frames": model.horizon, "first_action": score,
                     "inference_ms_per_observation_median_batch": ms})
        print(f"{name}: MAE={score['mae']:.3f}, gripper={score['per_joint_mae']['gripper']:.3f}", flush=True)

    first_config = ACTConfig.from_pretrained(checkpoints[0])
    act_state_dim = first_config.input_features["observation.state"].shape[0]
    act_action_dim = first_config.output_features["action"].shape[0]
    if (act_state_dim, act_action_dim) == (6, 6):
        act_root = PIPELINE / "datasets/patch4_158_merged_arm_only"
        manifest = json.loads((act_root / "arm_only_source.json").read_text())
        if Path(manifest["source_root"]).resolve() != Path(config["dataset_root"]).resolve():
            raise ValueError("ACT data and experiment source do not match")
        act_repo_id = manifest["repo_id"]
    elif (act_state_dim, act_action_dim) == (9, 9):
        act_root = Path(config["dataset_root"])
        act_repo_id = config["repo_id"]
    else:
        raise ValueError("ACT state/action must both have six or nine channels")
    metadata = LeRobotDatasetMetadata(act_repo_id, root=act_root)
    act_offset = first_config.action_offset_frames
    delta = resolve_delta_timestamps(first_config, metadata, {})
    act_data = LeRobotDataset(act_repo_id, root=act_root, episodes=validation,
                              delta_timestamps=delta, video_backend="pyav", return_uint8=True)
    if not np.array_equal(episode_ids, np.asarray(act_data.hf_dataset["episode_index"])):
        raise ValueError("Source and ACT data episode order differs")
    best, best_path = float("inf"), None
    for checkpoint in checkpoints:
        checkpoint_config = ACTConfig.from_pretrained(checkpoint)
        if (checkpoint_config.action_offset_frames != act_offset
                or checkpoint_config.input_features["observation.state"].shape[0] != act_state_dim
                or checkpoint_config.output_features["action"].shape[0] != act_action_dim):
            raise ValueError("ACT run contains checkpoints with different feature dimensions or label offsets")
        policy, preprocessor, postprocessor = load_act(checkpoint, device)
        prediction, target, valid, ms = predict_act(policy, preprocessor, postprocessor, act_data,
                                                     indices, device, args.batch_size)
        first = summarize(prediction[:, 0], target[:, 0], owners, valid[:, :1])
        chunk = summarize(prediction, target, owners, valid)
        name = checkpoint.parent.name
        rows.append({"family": "ACT", "name": name, "checkpoint": str(checkpoint),
                     "target_offset_frames": act_offset, "first_action": first, "chunk_10": chunk,
                     "inference_ms_per_observation_median_batch": ms})
        print(f"ACT {name}: first MAE={first['mae']:.3f}; 10-step MAE={chunk['mae']:.3f}; "
              f"gripper={first['per_joint_mae']['gripper']:.3f}", flush=True)
        if first["mae"] < best:
            best, best_path = first["mae"], checkpoint
        del policy, preprocessor, postprocessor
        if device == "mps":
            torch.mps.empty_cache()

    args.output.mkdir(parents=True)
    plot_scorecard(rows, args.output / "checkpoint_scorecard.png")
    chosen = load_act(best_path, device)
    fps = float(json.loads((Path(config["dataset_root"]) / "meta/info.json").read_text())["fps"])
    event_timing = plot_full_episodes(selected, fps, full, act_data, states, commands, episode_ids,
                                      models, chosen, device, args.batch_size, args.output,
                                      act_offset)
    report = {"experiment": str(args.experiment.resolve()), "validation_episodes": validation,
              "sampled_observations_per_episode": args.samples_per_episode,
              "selected_act_checkpoint": str(best_path), "selection_metric": "sampled first-action raw MAE",
              "act_action_offset_frames": act_offset,
              "act_state_channels": act_state_dim, "act_action_channels": act_action_dim,
              "scored_channels": "six arm/gripper commands; base velocities excluded",
              "plotted_episodes": selected, "gripper_thresholds": {"fully_closed_le": 5, "high_aperture_ge": 15,
                                                                     "low_aperture_lt": 15},
              "scores": rows, "plotted_episode_gripper_events": event_timing,
              "limitations": ["Recorded observations only; no closed-loop success claim.",
                              "First-action error and ten-step ACT chunk error are different measures.",
                              "Future BC targets three frames after its observation.",
                              "Inference timing is batch-screen time on this computer, not NUC loop timing."]}
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected ACT for plots: {best_path.parent.name}; review saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
