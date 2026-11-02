# Derived demonstration commands

This compact snapshot contains six commanded arm/gripper channels for all
158 final demonstrations. Source recordings are untouched and remain local.

- Source: 75,570 frames at 30 Hz, 110 original + 48 extra demonstrations.
- Plot snapshot: 25,327 samples, keeping every third frame plus each episode's
  final frame. Thus the nominal sampling rate is 10 Hz; the final interval can
  be shorter. Values and timestamps are rounded to six decimals.
- `commands_10hz.csv.gz`: episode ID, time in seconds, then shoulder pan,
  shoulder lift, elbow flex, wrist flex, wrist roll and gripper commands.
- `dataset.json`: per-episode source/sample counts, endpoints, collection group,
  split membership, derivation and SHA-256 hashes of the source Parquet,
  metadata, source map, split and compressed snapshot.

Units are recorded position-command units, not assumed physical degrees.
Group names follow the author's description of collection intent and source
folders. No automatic classification or smoothing was applied. The public
snapshot omits images, raw observations and three base-velocity channels.

To re-export from a local source dataset, using an environment with PyArrow:

```bash
python3 tools/build_readme_figures.py \
  --dataset-root /path/to/patch4_158_merged_all \
  --snapshot-dir /tmp/mickey17-dataset-export \
  --output-dir /tmp/mickey17-dataset-figures
```

To render from the snapshot alone, PyArrow is not required. Use the command in
[the figure notes](../../analysis/2026-11-02-readme/README.md).
