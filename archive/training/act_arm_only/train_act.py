"""Preview or train arm-only LeRobot ACT using the BC episode split.

Preview: python train/train_act.py --experiment experiment.json
Train:   python train/train_act.py --experiment experiment.json --train
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import load_experiment
from pipeline_settings import load_settings


JOINTS = ("arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
          "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos")


def arm_only_config(config: dict, root: Path) -> dict:
    root = root.expanduser().resolve()
    manifest_path = root / "arm_only_source.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Arm-only dataset missing: {root}; run data/derive_arm_only.py")
    manifest = json.loads(manifest_path.read_text())
    source_root = Path(config["dataset_root"])
    source_hash = hashlib.sha256((source_root / "meta/info.json").read_bytes()).hexdigest()
    if (Path(manifest["source_root"]) != source_root or
            manifest["source_info_sha256"] != source_hash):
        raise ValueError("Arm-only dataset was not derived from this experiment's merged dataset")
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] != config["total_episodes"] or info["total_frames"] != config["total_frames"]:
        raise ValueError("Arm-only dataset episode/frame counts differ from the common split")
    for feature in ("action", "observation.state"):
        spec = info["features"][feature]
        if spec["shape"] != [6] or spec["names"] != list(JOINTS):
            raise ValueError(f"Arm-only {feature} is not in six-joint order")
    act_config = dict(config)
    act_config["dataset_root"] = str(root)
    act_config["repo_id"] = manifest["repo_id"]
    act_config["source_dataset_root"] = str(source_root)
    return act_config


def command(config: dict, output: Path, steps: int, batch_size: int,
            train_settings: dict, action_offset_frames: int = 0) -> list[str]:
    trainer = Path(sys.executable).with_name("lerobot-train")
    if not trainer.is_file():
        raise FileNotFoundError(f"Cannot find {trainer}; use the LeRobot Mac environment")
    ordered = config["train_episodes"] + config["validation_episodes"]
    # LeRobot uses ceil(eval_split * total). Nudge the ratio downward to avoid
    # floating-point rounding selecting one extra validation episode.
    fraction = math.nextafter(len(config["validation_episodes"]) / len(ordered), 0.0)
    return [
        str(trainer),
        f"--dataset.repo_id={config['repo_id']}",
        f"--dataset.root={config['dataset_root']}",
        f"--dataset.episodes={json.dumps(ordered)}",
        f"--dataset.eval_split={fraction}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        "--policy.device=mps",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={train_settings['chunk_size']}",
        f"--policy.n_action_steps={train_settings['n_action_steps']}",
        f"--policy.action_offset_frames={action_offset_frames}",
        f"--policy.dim_model={train_settings['dim_model']}",
        f"--policy.n_heads={train_settings['n_heads']}",
        f"--policy.dim_feedforward={train_settings['dim_feedforward']}",
        f"--policy.n_encoder_layers={train_settings['n_encoder_layers']}",
        f"--policy.n_vae_encoder_layers={train_settings['n_vae_encoder_layers']}",
        f"--policy.optimizer_lr={train_settings['optimizer_lr']}",
        f"--policy.optimizer_lr_backbone={train_settings['optimizer_lr_backbone']}",
        f"--batch_size={batch_size}",
        f"--num_workers={train_settings['num_workers']}",
        f"--steps={steps}",
        f"--eval_steps={train_settings['eval_steps']}",
        f"--max_eval_samples={train_settings['max_eval_samples']}",
        f"--log_freq={train_settings['log_freq']}",
        f"--save_freq={train_settings['save_freq']}",
        "--env_eval_freq=0",
        "--wandb.enable=false",
        f"--output_dir={output}",
        f"--job_name=mikey17_act_patch4_offset{action_offset_frames}",
    ]


def main(action_offset_frames: int = 0) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Shared pipeline_config.toml")
    parser.add_argument("--arm-only-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets/patch4_merged_arm_only")
    parser.add_argument("--steps", type=int, help="Override act.train.steps")
    parser.add_argument("--batch-size", type=int, help="Override act.train.batch_size")
    parser.add_argument("--train", action="store_true", help="Start training; omitted means preview only")
    args = parser.parse_args()
    settings, settings_path, settings_sha256 = load_settings(args.config)
    train_settings = settings["act"]["train"]
    args.steps = args.steps if args.steps is not None else train_settings["steps"]
    args.batch_size = (args.batch_size if args.batch_size is not None
                       else train_settings["batch_size"])
    if args.steps < 1 or args.batch_size < 1:
        parser.error("Steps and batch size must be positive")
    if type(action_offset_frames) is not int or action_offset_frames < 0:
        parser.error("action_offset_frames must be a nonnegative integer")
    config = arm_only_config(load_experiment(args.experiment), args.arm_only_root)
    output = Path(__file__).resolve().parents[1] / "runs" / datetime.now().strftime(
        f"act_arm_only_offset{action_offset_frames}_%Y%m%d_%H%M%S")
    print(f"ARM-ONLY ACT | {config['total_episodes']} episodes, {config['total_frames']} frames")
    print(f"Input/output joint channels: 6/6; dataset: {config['dataset_root']}")
    print(f"Training IDs: {config['train_episodes']}")
    print(f"Validation IDs: {config['validation_episodes']}")
    print(f"Settings: {settings_path}; SHA-256: {settings_sha256}")
    print(f"Action chunk: {train_settings['chunk_size']}; "
          f"training action window: {train_settings['n_action_steps']}; "
          f"label offset: {action_offset_frames} frames; output: {output}")
    cmd = command(config, output, args.steps, args.batch_size, train_settings,
                  action_offset_frames)
    print("Command:")
    for argument in cmd:
        print(f"  {argument}")
    if not args.train:
        print("PREVIEW ONLY. Add --train to start training.")
        return
    if output.exists():
        raise FileExistsError(output)
    # The LeRobot trainer creates output. Keep the common split and source
    # provenance with its checkpoint configuration.
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)
    config["act_action_offset_frames"] = action_offset_frames
    (output / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(settings_path, output / "pipeline_config.toml")


if __name__ == "__main__":
    main()
