#!/usr/bin/env python3
"""Build a real-time illustrative comparison from three supplied camera clips.

Requires ffmpeg/ffprobe, Pillow and matplotlib (for a portable bundled font).
No trial synchronization, speed changes, synthesized footage or audio mixing.
The original clips are local inputs and are not distributed in the snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    args = parser.parse_args()
    assets = args.assets_dir
    # Preserve the original camera recording locally; remove metadata from the
    # publication copy without re-encoding video/audio or changing playback.
    hero_source = assets / "showcase_in_alpha_0.5.mp4"
    hero_public = assets / "hero_demo.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(hero_source),
                    "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-map_metadata", "-1",
                    "-movflags", "+faststart", str(hero_public)], check=True)
    sources = [assets / f"demo_alpha{x}.MOV" for x in ["0.2", "0.5", "1"]]
    durations = [float(probe(p)["format"]["duration"]) for p in sources]
    duration = max(durations)
    font = font_manager.findfont("DejaVu Sans")
    large, small = ImageFont.truetype(font, 24), ImageFont.truetype(font, 17)
    with tempfile.TemporaryDirectory(prefix="mickey17-media-") as tmp:
        tmp = Path(tmp)
        for frame in range(round(duration * 30)):
            elapsed = frame / 30
            img = Image.new("RGBA", (1440, 370), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            for i, (alpha, end) in enumerate(zip(["0.2", "0.5", "1.0"], durations)):
                x = i * 480
                draw.rectangle((x, 0, x+479, 59), fill="white")
                draw.text((x+18, 7), f"α = {alpha}", font=large, fill="#263E5A")
                message = f"Video time {elapsed:05.2f} s" if elapsed < end else f"CLIP ENDED · {end:.2f} s"
                draw.text((x+18, 36), message, font=small, fill="#263E5A" if elapsed < end else "#B95B36")
                if elapsed >= end:
                    draw.rectangle((x, 60, x+479, 329), fill=(255,255,255,100))
                    draw.rectangle((x+110, 170, x+370, 220), fill="white")
                    draw.text((x+163, 181), "CLIP ENDED", font=large, fill="#B95B36")
            draw.rectangle((0, 330, 1440, 370), fill="white")
            draw.text((24, 341), "Illustrative recordings · original speed · independent starts, not synchronized trials · audio omitted",
                      font=small, fill="#263E5A")
            img.save(tmp / f"overlay{frame:04d}.png")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for source in sources:
            command += ["-i", str(source)]
        command += ["-framerate", "30", "-i", str(tmp / "overlay%04d.png")]
        filters = []
        for i in range(3):
            filters.append(f"[{i}:v]setpts=PTS-STARTPTS,fps=30,scale=480:270:force_original_aspect_ratio=decrease,"
                           f"pad=480:370:(ow-iw)/2:60+(270-ih)/2:color=white,setsar=1,"
                           f"tpad=stop_mode=clone:stop_duration={duration}[v{i}]")
        filters += ["[v0][v1][v2]hstack=inputs=3[stack]", "[stack][3:v]overlay=shortest=1[out]"]
        command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-t", str(duration),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "25", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(assets / "alpha_comparison.mp4")]
        subprocess.run(command, check=True)
    for source, time, poster in [(hero_public, 9, "hero_poster.jpg"),
                                 (assets/"alpha_comparison.mp4", 8, "alpha_comparison_poster.jpg")]:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(time),
                        "-i", str(source), "-frames:v", "1", "-q:v", "2", str(assets/poster)], check=True)
    manifest = {"comparison": {"fps": 30, "speed": 1, "audio": "omitted", "alignment": "independent clip starts",
                               "sources": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                            "duration_s": d} for p, d in zip(sources, durations)]},
                "hero": {"file": "hero_demo.mp4", "source": hero_source.name,
                         "source_sha256": hashlib.sha256(hero_source.read_bytes()).hexdigest(),
                         "processing": "metadata stripped; video/audio stream copy, original speed",
                         "poster_at_s": 9,
                         "caption": "Supervised successful demonstration; a hand steadies the base."}}
    (assets/"media_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
