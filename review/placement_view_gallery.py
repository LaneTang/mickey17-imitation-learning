"""Compare wrist-camera views around the demonstrated placement release.

Read only: source demonstrations and merged action parquet are never changed.
Writes a browsable gallery, four contact sheets, and an event CSV to --output.
The event detector locates the largest 0.5-second gripper-command rise after
6 seconds; inspect the images to verify that this is the actual release.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
    "wrist_roll", "gripper",
)


def release_event(actions: np.ndarray, fps: int) -> tuple[int, float]:
    """Find the strongest later opening, preserving the pre-opening image."""
    gripper = actions[:, 5]
    window = round(0.5 * fps)
    first = round(6 * fps)
    last = min(round(14.5 * fps), len(gripper) - window - 1)
    if last <= first:
        raise ValueError("Episode is too short for a placement release search")
    rises = gripper[first + window:last + window + 1] - gripper[first:last + 1]
    onset = int(first + np.argmax(rises))
    return onset, float(rises.max())


def frame_at(container: av.container.InputContainer, stream, index: int,
             fps: int) -> Image.Image:
    seconds = index / fps
    container.seek(round(seconds / float(stream.time_base)), stream=stream,
                   backward=True)
    for frame in container.decode(stream):
        if float(frame.pts * frame.time_base) >= seconds - 0.001:
            return frame.to_image().convert("RGB")
    raise RuntimeError(f"Could not decode frame {index}")


def write_sheet(records: list[dict], output: Path, start: int, stop: int) -> None:
    subset = records[start:stop]
    columns = 5
    rows = (len(subset) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(19, rows * 3.15),
                             constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, item in zip(axes, subset, strict=False):
        ax.imshow(Image.open(output / item["image_onset"]))
        ax.set_title(f"E{item['episode']:03d} · {item['group']} · {item['release_s']:.1f}s\n"
                     f"gripper +{item['gripper_rise']:.0f}", fontsize=9)
    fig.suptitle("Wrist view when the placement release begins", fontsize=17)
    fig.savefig(output / f"contact_sheet_{start:03d}_{stop - 1:03d}.png", dpi=145)
    plt.close(fig)


def write_gallery(records: list[dict], output: Path) -> None:
    cards = []
    for item in records:
        badges = (f"E{item['episode']:03d} · {html.escape(item['source_name'])} · "
                  f"{item['group']} · {item['split']}")
        images = "".join(
            f'<figure><a href="{item[key]}"><img loading="lazy" src="{item[key]}" '
            f'alt="Episode {item["episode"]} {label}"></a><figcaption>{label}</figcaption></figure>'
            for key, label in (("image_before", "0.3 s before"),
                               ("image_onset", "release begins"),
                               ("image_after", "0.5 s after"))
        )
        cards.append(
            f'<article data-group="{item["group"]}" data-split="{item["split"]}">'
            f'<h2>{badges}</h2><p>{item["release_s"]:.2f} s · gripper command '
            f'{item["gripper_before"]:.1f} → {item["gripper_after"]:.1f} '
            f'(+{item["gripper_rise"]:.1f})</p><div class="frames">{images}</div>'
            '</article>'
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mikey17 placement views</title>
<style>
body{font:16px system-ui,sans-serif;color:#17212b;background:#eef2f5;margin:0 auto;max-width:1380px;padding:24px}
h1{margin-bottom:4px}p{line-height:1.4}header{position:sticky;top:0;background:#eef2f5;padding:8px 0;z-index:2}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}select,input{font:inherit;padding:8px}
article{background:white;border:1px solid #d7e0e5;border-radius:12px;padding:14px;margin:14px 0}
article h2{font-size:18px;margin:0}article p{margin:5px 0 12px}.frames{display:flex;gap:9px}
figure{flex:1;min-width:0;margin:0}img{display:block;width:100%;height:auto;border-radius:6px}
figcaption{text-align:center;font-size:13px;color:#52616d;margin-top:4px}
@media(max-width:720px){.frames{display:block}figure{margin-bottom:10px}}
</style><header><h1>Placement camera views · 158 demonstrations</h1>
<p>Aligned to the strongest half-second gripper opening after 6 s. This is a command-based candidate release; inspect frames for physical placement.</p>
<div class="controls"><label>Group <select id="group"><option value="all">All</option><option value="original">Original 110</option><option value="extra">Extra 48</option></select></label>
<label>Split <select id="split"><option value="all">All</option><option value="train">Train</option><option value="validation">Validation</option></select></label>
<label>Episode <input id="search" type="number" min="0" max="157" placeholder="0–157"></label></div>
<p id="count"></p></header><main>""" + "\n".join(cards) + """</main>
<script>
const cards=[...document.querySelectorAll('article')];
function filter(){const g=document.querySelector('#group').value,s=document.querySelector('#split').value,
e=document.querySelector('#search').value.trim();let shown=0;
cards.forEach((card,i)=>{const yes=(g==='all'||card.dataset.group===g)&&(s==='all'||card.dataset.split===s)&&(!e||i===Number(e));card.hidden=!yes;shown+=yes});
document.querySelector('#count').textContent=`Showing ${shown} of ${cards.length} episodes`;}
document.querySelectorAll('select,input').forEach(x=>x.addEventListener('input',filter));filter();
</script></html>"""
    (output / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--experiment", type=Path, default=root / "experiment_patch4_158.json")
    parser.add_argument("--output", type=Path,
                        default=root / "reviews" / "placement_view_gallery_158")
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text())
    dataset = Path(experiment["dataset_root"])
    source_map = json.loads((dataset / "source_episode_map.json").read_text())
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = int(info["fps"])
    data = pq.read_table(dataset / "data/chunk-000/file-000.parquet",
                         columns=["episode_index", "action"]).to_pydict()
    episodes = np.asarray(data["episode_index"])
    actions = np.asarray(data["action"], dtype=np.float32)
    output = args.output.expanduser().resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    validation = set(experiment["validation_episodes"])
    records = []
    for episode in sorted(map(int, source_map)):
        segment = actions[episodes == episode]
        if len(segment) != source_map[str(episode)]["frames"]:
            raise ValueError(f"Episode {episode} length does not match source map")
        onset, rise = release_event(segment, fps)
        before = max(0, onset - round(0.3 * fps))
        after = min(len(segment) - 1, onset + round(0.5 * fps))
        source = Path(source_map[str(episode)]["source_dir"])
        video = source / "videos/observation.images.wrist/chunk-000/file-000.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        image_paths = {}
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            for label, index in (("before", before), ("onset", onset), ("after", after)):
                relative = Path("frames") / f"e{episode:03d}_{label}.jpg"
                frame_at(container, stream, index, fps).save(output / relative,
                                                               quality=88, optimize=True)
                image_paths[f"image_{label}"] = relative.as_posix()
        record = {
            "episode": episode,
            "source_name": source.name,
            "group": "extra" if "4th_patch_extra" in str(source) else "original",
            "split": "validation" if episode in validation else "train",
            "release_frame": onset,
            "release_s": onset / fps,
            "gripper_before": float(segment[onset, 5]),
            "gripper_after": float(segment[after, 5]),
            "gripper_rise": rise,
            **{name: float(segment[onset, i]) for i, name in enumerate(JOINTS)},
            **image_paths,
        }
        records.append(record)
        if (episode + 1) % 20 == 0 or episode == len(source_map) - 1:
            print(f"Extracted placement views for {episode + 1}/{len(source_map)} episodes",
                  flush=True)
    with (output / "placement_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    write_gallery(records, output)
    for start in range(0, len(records), 40):
        write_sheet(records, output, start, min(start + 40, len(records)))
    summary = {
        "episodes": len(records),
        "detector": "largest 0.5-second commanded gripper opening from 6.0 to 14.5 seconds",
        "fps": fps,
        "median_release_s": float(np.median([r["release_s"] for r in records])),
        "min_gripper_rise": min(r["gripper_rise"] for r in records),
        "source_dataset": str(dataset),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Gallery: {output / 'index.html'}")
    print(f"Contact sheets and event table: {output}")


if __name__ == "__main__":
    main()
