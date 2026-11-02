# Reproduce the evidence; explore the robot code

## Verified offline analysis

From a checkout, Python 3.10+ can audit the evidence using only its standard library:

```bash
python3 tools/analyze_outcomes.py --no-figures --output-dir /tmp/mickey17-audit
python3 -m unittest discover -s tests -v
python3 tools/verify_source_snapshot.py
```

The default input is `evidence/2026-11-01/outcomes.json`. Outputs are
`summary.json`, `audit.json`, `trial_registry.csv` and `results_table.md`.
No robot libraries are imported. Tests cover ranks 0–5, null/missing labels,
absent times, controller failures, shared baselines and trace selection.

To regenerate all ten PNG/SVG figures, use **Python 3.11+** in an isolated environment:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
.venv-analysis/bin/python tools/analyze_outcomes.py --output-dir /tmp/mickey17-analysis
```

Checked figures use NumPy 2.2.6 and Matplotlib 3.11.2 on macOS. Calculations
use the standard library. Font rendering can vary across systems. No original
workspace path is needed. Do not install hardware dependencies for this path.

The command additionally generates `scene_sensitivity`, `outcome_flow`,
`system_runtime` and `pipeline` figures. Vector SVG files retain editable text.
Outcome-flow widths come from the snapshot; the scheduling example is an
illustration of the existing implementation, not another measured trial.

If you hold the raw archive, verify the SHA-256 in the
[evidence notes](../evidence/2026-11-01/README.md) and extract it separately:

```bash
python3 tools/analyze_outcomes.py \
  --raw-dir /path/to/e2e_rollout_official_act_offset0 \
  --output-dir /tmp/mickey17-from-raw
```

This read-only importer additionally emits `outcomes.json`, records source-file
hashes, rejects conflicting duplicate IDs, and validates primary final status
and row counts. The public snapshot reproduces published calculations;
independently authenticating raw observations requires the local archive.

## Training provenance

The deployed zero-offset ACT is step **018000** from a **20,000-step** official
LeRobot training run. The earlier `train/train_act.py` wrapper is a separate
six-channel experiment and does not describe this checkpoint. Saved
[training settings](../evidence/2026-11-01/train_config.json),
[policy settings](../evidence/2026-11-01/config.json) and
[split IDs](../evidence/2026-11-01/split.json) are included.

| Item | Recorded setting |
| --- | --- |
| Demonstrations | 158 merged episodes; 134 training, 24 validation |
| Input | One 640 × 480 wrist image plus nine state channels |
| Output | Six arm/gripper commands plus three base velocities |
| Architecture | ImageNet-pretrained ResNet18; model dim 256; four heads; FFN 1024 |
| Transformer | Two encoder, one decoder, two VAE encoder layers; latent dim 32 |
| ACT | Chunk 30; label offset 0; KL weight 10; dropout 0.1 |
| Optimizer | AdamW; LR 1e-4; backbone LR 1e-5; weight decay 1e-4 |
| Training | Batch 4; seed 1000; MPS; no AMP or image augmentation |
| Selection | Sampled first-action raw MAE in the saved held-out review |

The training configuration orders 134 training IDs before 24 validation IDs
and sets `eval_split=24/158`; the inspected LeRobot factory holds out the last
episodes per task. Split IDs match the review validation IDs. Weights,
processors and demonstration videos remain local, so this checkout alone
cannot reproduce model training or policy inference.

## Evaluated deployment

The entry point is **`deployment/rollout_act_overlap.py`**. Other rollout scripts
are development history. The requested control rate is 30 Hz; a worker runs
inference, requests replanning about every 12 frames and supplies 30-step
chunks. Base commands are forced to zero, including with nine-channel models.

During the first zero-offset inference the robot holds its measured starting
pose and the first chunk starts when ready. Later chunks target
`source_cycle + step`; past entries of the **new chunk** are discarded. Older
queued commands continue executing during inference. Predictions sharing a
future cycle blend with new weight `0.7 * min((lead + 1) / 6, 1)`, where lead
is cycles after readiness.

Example: observed at cycle 24, ready at 34, a new chunk drops steps 0–9 and
step 10 targets cycle 34. If an older prediction exists there, new weight is
0.7/6; at cycle 39 it reaches 0.7. This queue blending is separate from the
controller filter:

```text
filtered_t = alpha * proposal_t + (1 - alpha) * previous_filtered_t
```

Only the first five arm channels are filtered; the gripper is unchanged. Raw
proposal checks come first, then filtered-step and bounded-target checks.
Queue exhaustion stops execution. Echoed commands and observed positions are
logged separately. See the [rollout](../deployment/rollout_act_overlap.py),
[queue](../deployment/act_overlap_queue.py), [settings](../pipeline_config.toml)
and [empirical limits](../deployment/empirical_limits_patch4_158.json).

## Hardware is a separate workflow

The platform is SO-101 on LeKiwi, Intel N95 NUC, CPU inference, one wrist camera
and zero base commands. The Mac's MPS device handled training. Historical
[setup](ROBOT_SETUP.md), [workflow](IL_PIPELINE.md) and
[deployment notes](../deployment/README.md) retain old paths and policy variants.
Camera `/dev/videoN` numbering can change: identify the wrist camera using its
tested `/dev/v4l/by-path/...-video-index0` link and configure the actual host.

External requirements include installed LeRobot/PyTorch, OpenCV, serial/robot
communication dependencies, complete checkpoint and processor files,
calibration, the actual camera configuration and matching controller limits.
The Mac LeRobot base commit is `c364df18c54b670b6450073039aa2cd6024a6855`;
project changes outside this standalone directory also exist. The installed
NUC LeRobot revision, Python/package lockfile and full calibration were not
captured. `support/configuration_act.py` is a reference extension, not an installer.

Offline commands above are verified. Historical robot commands require
machine-specific preflight. This showcase work performs no hardware operation,
training or deployment-behavior change.

## Media

With the original local `assets/demo_alpha0.2.MOV`, `demo_alpha0.5.MOV` and
`demo_alpha1.MOV`, the local `assets/showcase_in_alpha_0.5.mp4`, analysis
dependencies, Pillow with WebP support and FFmpeg/ffprobe with libx264:

```bash
python3 tools/build_showcase_media.py
```

The comparison preserves speed, starts each independent clip at its own zero,
displays video time, holds and marks ended clips, and omits audio. Originals
remain local; their hashes are in [media provenance](../assets/media_provenance.json).
These recordings are not synchronized experimental trials. The existing hero
MP4 has a poster at 9 s. A hand steadies the base; it is a supervised demo.
The public `hero_demo.mp4` copies the original video/audio streams while stripping
metadata. Original videos remain unchanged locally and are excluded from the
current published tree. Earlier repository commits contained two source videos;
this update does not rewrite existing Git history.

The landscape α = 0.5 recording supplies the README opening. Its complete MP4
is resized to 1280 px with the full field of view and original speed; audio
and metadata are omitted. The inline animated WebP is 900 px wide at 10 fps,
with 100 ms frame durations. This reduces the frame rate without speeding up
the motion. Source hashes and transformations are recorded in media provenance.
The original short, front-view demonstration remains available separately.

The three static pages in `media/` provide browser playback via GitHub Pages,
which serves MP4 MIME types correctly. GitHub's repository file viewer does
not play these videos. The pages can also be served locally with Python's
`http.server`; they require no build system or JavaScript.
