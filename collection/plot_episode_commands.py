"""Save a six-motor command plot for one recorded Mikey 17 demonstration.

Run on the NUC from this directory, for example:
    python plot_episode_commands.py "../../outputs/008_demonstration"
    python plot_episode_commands.py --batch "../../outputs/4th_patch"

The PNG is saved inside the episode directory. No desktop display is required.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

MOTORS = (
    ("arm_shoulder_pan.pos", "Shoulder pan"),
    ("arm_shoulder_lift.pos", "Shoulder lift"),
    ("arm_elbow_flex.pos", "Elbow flex"),
    ("arm_wrist_flex.pos", "Wrist flex"),
    ("arm_wrist_roll.pos", "Wrist roll"),
    ("arm_gripper.pos", "Gripper"),
)


def plot_episode(episode_dir: Path) -> Path:
    episode_dir = Path(episode_dir).expanduser().resolve()
    with (episode_dir / "meta" / "info.json").open() as file:
        info = json.load(file)
    action_names = info["features"]["action"]["names"]
    indices = [action_names.index(name) for name, _ in MOTORS]

    parquet_files = sorted((episode_dir / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode data found in {episode_dir / 'data'}")
    tables = [pq.read_table(path, columns=["timestamp", "action"]) for path in parquet_files]
    seconds = np.concatenate([np.asarray(t["timestamp"].to_numpy(), dtype=float) for t in tables])
    actions = np.concatenate([np.asarray(t["action"].to_pylist(), dtype=float) for t in tables])
    if len(seconds) < 2 or actions.ndim != 2 or not np.isfinite(actions).all():
        raise ValueError("Episode needs at least two frames with finite motor commands")
    seconds -= seconds[0]
    if not np.isfinite(seconds).all() or seconds[-1] <= 0:
        raise ValueError("Episode timestamps must be finite and increasing")

    width, height = 1500, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((55, 22), f"{episode_dir.name}  |  {len(seconds)} frames  |  {seconds[-1]:.1f} s", fill="#202632", font=font)
    panel_width, panel_height = 690, 322
    for motor_number, (name, label) in enumerate(MOTORS):
        row, col = divmod(motor_number, 2)
        left = 55 + col * 745
        top = 65 + row * 342
        x0, x1 = left + 68, left + panel_width - 18
        y0, y1 = top + 35, top + panel_height - 42
        values = actions[:, indices[motor_number]]
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.08, 1.0)
        low -= padding
        high += padding

        draw.text((left, top), label, fill="#202632", font=font)
        for tick in range(5):
            y = round(y1 - tick * (y1 - y0) / 4)
            value = low + tick * (high - low) / 4
            draw.line((x0, y, x1, y), fill="#e3e7ec", width=1)
            draw.text((left, y - 6), f"{value:.1f}", fill="#606a76", font=font)
        draw.line((x0, y0, x0, y1), fill="#707984", width=2)
        draw.line((x0, y1, x1, y1), fill="#707984", width=2)
        for tick in range(5):
            x = round(x0 + tick * (x1 - x0) / 4)
            draw.text((x - 12, y1 + 8), f"{seconds[-1] * tick / 4:.0f}", fill="#606a76", font=font)
        points = [
            (
                round(x0 + float(t) / seconds[-1] * (x1 - x0)),
                round(y1 - (float(v) - low) / (high - low) * (y1 - y0)),
            )
            for t, v in zip(seconds, values, strict=True)
        ]
        draw.line(points, fill="#2468b8", width=3, joint="curve")
        draw.text((x1 - 65, y1 + 8), "time (s)", fill="#606a76", font=font)

    output = episode_dir / "motor_commands.png"
    image.save(output)
    return output


def _video_review(episode_dir: Path) -> tuple[Path, dict]:
    """Sample wrist frames for visual review; indicators are not quality verdicts."""
    videos = sorted((episode_dir / "videos" / "observation.images.wrist").glob("**/*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"Expected one wrist video in {episode_dir}; found {len(videos)}")
    capture = cv2.VideoCapture(str(videos[0]))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open wrist video: {videos[0]}")
    frames = []
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count < 1 or fps <= 0:
            raise ValueError(f"Invalid wrist video duration or FPS: {videos[0]}")
        for index in np.unique(np.linspace(0, count - 1, min(7, count), dtype=int)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index} in {videos[0]}")
            frames.append((float(index) / fps, bgr))
    finally:
        capture.release()

    tile_width, tile_height = 320, 240
    sheet = Image.new("RGB", (tile_width * len(frames), tile_height + 42), "white")
    draw = ImageDraw.Draw(sheet)
    brightness, bright_fraction, sharpness = [], [], []
    for i, (when, bgr) in enumerate(frames):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        bright_fraction.append(float(np.mean(gray >= 250)))
        center = gray[gray.shape[0] // 4:3 * gray.shape[0] // 4,
                      gray.shape[1] // 4:3 * gray.shape[1] // 4]
        sharpness.append(float(cv2.Laplacian(center, cv2.CV_64F).var()))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(rgb).resize((tile_width, tile_height)),
                    (i * tile_width, 32))
        draw.text((i * tile_width + 8, 7), f"{when:.1f} s", fill="#202632")
    output = episode_dir / "wrist_review.png"
    sheet.save(output)
    return output, {
        "sampled_video_frames": len(frames),
        "mean_brightness_0_255": round(float(np.mean(brightness)), 2),
        "mean_fraction_pixels_ge_250": round(float(np.mean(bright_fraction)), 4),
        "median_center_sharpness_proxy": round(float(np.median(sharpness)), 2),
    }


def analyze_episode(episode_dir: Path) -> dict:
    """Produce one command plot, wrist frame strip, and machine-readable summary."""
    episode_dir = Path(episode_dir).expanduser().resolve()
    plot = plot_episode(episode_dir)
    sheet, image_metrics = _video_review(episode_dir)
    with (episode_dir / "meta" / "info.json").open() as file:
        info = json.load(file)
    action_names = info["features"]["action"]["names"]
    gripper_index = action_names.index("arm_gripper.pos")
    paths = sorted((episode_dir / "data").glob("**/*.parquet"))
    tables = [pq.read_table(path, columns=["timestamp", "action"]) for path in paths]
    seconds = np.concatenate([np.asarray(t["timestamp"].to_numpy(), dtype=float) for t in tables])
    actions = np.concatenate([np.asarray(t["action"].to_pylist(), dtype=float) for t in tables])
    gripper = actions[:, gripper_index]
    summary = {
        "episode": episode_dir.name,
        "frames": len(seconds),
        "duration_s": round(float(seconds[-1] - seconds[0]), 2),
        "sample_rate_hz": round(float((len(seconds) - 1) / (seconds[-1] - seconds[0])), 2),
        "gripper_command_min": round(float(gripper.min()), 2),
        "gripper_command_max": round(float(gripper.max()), 2),
        **image_metrics,
    }
    summary_path = episode_dir / "episode_review.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {**summary, "command_plot": str(plot), "wrist_review": str(sheet),
            "summary": str(summary_path)}


def analyze_batch(batch_dir: Path, output_dir: Path | None = None) -> dict:
    """Review immediate child episodes and save a batch table and overview plot."""
    batch_dir = Path(batch_dir).expanduser().resolve()
    episodes = sorted(path for path in batch_dir.iterdir()
                      if path.is_dir() and (path / "meta" / "info.json").is_file())
    if not episodes:
        raise ValueError(f"No episode directories found in {batch_dir}")
    output_dir = (Path(output_dir).expanduser().resolve() if output_dir
                  else batch_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for episode in episodes:
        try:
            rows.append(analyze_episode(episode))
            print(f"Reviewed {episode.name}", flush=True)
        except Exception as exc:
            failures.append({"episode": episode.name, "error": str(exc)})
            print(f"Could not review {episode.name}: {exc}", flush=True)
    if rows:
        keys = [key for key in rows[0] if key not in ("command_plot", "wrist_review", "summary")]
        with (output_dir / "episodes.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: row[key] for key in keys} for row in rows)
        _plot_batch(rows, output_dir / "batch_overview.png")
        _plot_batch_commands([batch_dir / row["episode"] for row in rows],
                             output_dir / "batch_command_curves.png")
    result = {"batch": str(batch_dir), "reviewed": len(rows), "failed": failures,
              "output_dir": str(output_dir)}
    (output_dir / "batch_review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _plot_batch(rows: list[dict], output: Path) -> None:
    width = max(1100, 80 * len(rows) + 150)
    image = Image.new("RGB", (width, 700), "white")
    draw = ImageDraw.Draw(image)
    metrics = (
        ("duration_s", "Duration (s)", "#2468b8"),
        ("gripper_command_max", "Maximum gripper command", "#6547a8"),
        ("mean_fraction_pixels_ge_250", "Very bright pixel fraction", "#df783e"),
        ("median_center_sharpness_proxy", "Center sharpness proxy", "#258a71"),
    )
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for panel, (key, title, color) in enumerate(metrics):
        top = 20 + panel * 165
        bottom = top + 120
        values = np.asarray([float(row[key]) for row in rows])
        scale = max(float(values.max()) * 1.1, 0.01)
        draw.text((32, top), title, fill="#202632", font=font)
        draw.line((150, bottom, width - 25, bottom), fill="#707984", width=2)
        spacing = (width - 190) / len(rows)
        for i, (row, value) in enumerate(zip(rows, values, strict=True)):
            x = 155 + (i + 0.5) * spacing
            half_bar = min(14, spacing * 0.3)
            draw.rectangle((x - half_bar, bottom - value / scale * 94,
                            x + half_bar, bottom), fill=color)
            draw.text((x - 12, bottom + 5), row["episode"].split("_")[0],
                      fill="#606a76", font=font)
    image.save(output)


def _plot_batch_commands(episodes: list[Path], output: Path) -> None:
    """Show per-episode commands and the batch median against relative time."""
    samples = np.linspace(0.0, 1.0, 101)
    trajectories = []
    for episode in episodes:
        with (episode / "meta" / "info.json").open() as file:
            names = json.load(file)["features"]["action"]["names"]
        indices = [names.index(name) for name, _ in MOTORS]
        paths = sorted((episode / "data").glob("**/*.parquet"))
        tables = [pq.read_table(path, columns=["timestamp", "action"]) for path in paths]
        seconds = np.concatenate([np.asarray(t["timestamp"].to_numpy(), dtype=float)
                                  for t in tables])
        commands = np.concatenate([np.asarray(t["action"].to_pylist(), dtype=float)
                                   for t in tables])[:, indices]
        if seconds[-1] <= seconds[0] or not np.all(np.diff(seconds) > 0):
            raise ValueError(f"Invalid episode timestamps in {episode}")
        relative = (seconds - seconds[0]) / (seconds[-1] - seconds[0])
        trajectories.append(np.stack([np.interp(samples, relative, commands[:, j])
                                      for j in range(len(MOTORS))], axis=1))
    trajectories = np.stack(trajectories)
    image = Image.new("RGB", (1500, 1120), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((50, 15), f"{len(episodes)} demonstrations | gray: each | blue: median | relative episode time",
              fill="#202632", font=font)
    for joint, (_, label) in enumerate(MOTORS):
        row, col = divmod(joint, 2)
        left, top = 55 + col * 745, 65 + row * 342
        x0, x1, y0, y1 = left + 68, left + 672, top + 35, top + 280
        values = trajectories[:, :, joint]
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.08, 1.0)
        low -= padding
        high += padding
        draw.text((left, top), label, fill="#202632", font=font)
        draw.line((x0, y0, x0, y1), fill="#707984", width=2)
        draw.line((x0, y1, x1, y1), fill="#707984", width=2)
        for tick in range(5):
            y = round(y1 - tick * (y1 - y0) / 4)
            draw.line((x0, y, x1, y), fill="#e3e7ec", width=1)
            draw.text((left, y - 6), f"{low + tick * (high - low) / 4:.1f}",
                      fill="#606a76", font=font)
            x = round(x0 + tick * (x1 - x0) / 4)
            draw.text((x - 12, y1 + 8), f"{tick * 25}%", fill="#606a76", font=font)
        for series in values:
            points = [(round(x0 + t * (x1 - x0)),
                       round(y1 - (float(v) - low) / (high - low) * (y1 - y0)))
                      for t, v in zip(samples, series, strict=True)]
            draw.line(points, fill="#c6ccd5", width=1)
        median = np.median(values, axis=0)
        points = [(round(x0 + t * (x1 - x0)),
                   round(y1 - (float(v) - low) / (high - low) * (y1 - y0)))
                  for t, v in zip(samples, median, strict=True)]
        draw.line(points, fill="#2468b8", width=3)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", nargs="?", type=Path)
    parser.add_argument("--batch", type=Path, help="Directory containing episode folders")
    parser.add_argument("--output-dir", type=Path, help="Batch summary destination")
    args = parser.parse_args()
    if bool(args.episode_dir) == bool(args.batch):
        parser.error("Give exactly one episode directory or --batch directory")
    if args.batch:
        print(json.dumps(analyze_batch(args.batch, args.output_dir), indent=2))
    else:
        print(json.dumps(analyze_episode(args.episode_dir), indent=2))


if __name__ == "__main__":
    main()
