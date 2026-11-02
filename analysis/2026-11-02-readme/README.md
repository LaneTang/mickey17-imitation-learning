# README figures

From the repository root, with the existing analysis dependencies:

```bash
python3 tools/build_readme_figures.py --output-dir /tmp/mickey17-readme-figures
```

This renders six editable SVG/PNG pairs without the original dataset or robot
libraries. Inputs default to `evidence/2026-11-02-dataset/`. The original
experiment-pipeline figure remains in `analysis/2026-11-01/pipeline.png`.
The [completion chart](../2026-11-02-completion/README.md) has its own command.

## What the figures show

- **runtime_architecture:** the selected zero-offset, 30-step ACT checkpoint.
  ResNet18 image features are projected to tokens with positional embeddings;
  normalized nine-channel state is projected to a state token. Inference uses
  a zero latent vector, not the training-only VAE encoder. Two transformer
  encoder layers and one decoder layer produce 30 nine-channel predictions.
  Output denormalization precedes retention of six arm/gripper channels.
  Queue blending, raw checks, filtering and bounded-target checks follow.
- **scheduling_filter:** an illustrative source-cycle-24, ready-cycle-34 chunk.
  Old commands execute while inference runs. New steps 0–9 expire; matching
  future cycles blend with new weight `0.7 * min((lead + 1) / 6, 1)`. The
  example assumes old predictions through cycle 41. New-only targets have
  weight 1. The separate alpha filter leaves the gripper unchanged.
- **dataset_flow:** source groups are the author's collection intent,
  identified through source folders, not inferred from the motion. The saved
  split gives scene 94 train / 16 validation and motion 40 train / 8 validation.
  Each group branches into its own training/validation endpoints to avoid
  crossings; together they total 134 training and 24 validation demos.
  Percentages use all 158 demos as the denominator. The shared
  `tools/flow_figures.py` renderer keeps every band proportional to its count.
- **dataset_joint_commands:** all 158 demonstrations, six commanded arm/gripper
  positions, actual episode timestamps. No time warping, interpolation or
  curve averaging. The three base channels are excluded.
- **practice_pipeline / policy_deployment:** redrawn from the author's supplied
  handwritten references and checked against the documented development
  workflow. These are workflow diagrams, not additional measurements.

## Provenance

The architecture was checked against the selected
`evidence/2026-11-01/config.json`, `deployment/shadow_act.py::predict_chunk`,
the rollout/queue implementation, and the local LeRobot
`src/lerobot/policies/act/modeling_act.py::ACT.forward`. This standalone
repository does not contain that entire external LeRobot checkout.
The selected config has model dimension 256, four heads, chunk size 30,
ResNet18 and zero action-label offset. The VAE latent dimension is 32.

Public figures are generated with the versions in `requirements-analysis.txt`.
Fonts may render differently on other systems. The original November 1
figures and numerical evidence have not been overwritten.
