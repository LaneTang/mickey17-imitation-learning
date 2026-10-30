# Offline checkpoint review

## Official zero-offset ACT on the full LeKiwi dataset

From the Mac `Imitation Learning` directory, run:

```bash
../../.venv/bin/python review/offline_review.py \
  --experiment experiment_patch4_158.json \
  --act-run runs/act_zero_offset_official_pipeline \
  --output reviews/patch4_158_act_zero_offset_official \
  --plot-episodes 129 151
```

The official trainer saves `train_config.json` in each checkpoint instead of
the local wrapper's `experiment.json`. The reviewer checks its dataset path,
episode order, and validation split before comparing predictions. It uses the
full nine-channel dataset for this ACT checkpoint but scores and plots only
the six arm/gripper commands, making its results comparable to the earlier
arm-only review. The three base-velocity outputs are excluded from these
scores. This is a recorded-observation review; it never connects to the robot.

For the latency-12 ACT variant, provide its own `--act-run` and a new
`--output` to `offline_review.py`. The review reads each checkpoint's saved
offset, masks padded end-of-episode commands, and plots the first prediction
at `t+12`. Raw MAE against a future command is a different target from the
original ACT's same-frame MAE; also inspect the time-aligned gripper curves.
`act_teacher_forced_trace.py` reads the offset from the checkpoint and uses
it in its overlap queue and saved summary.

## Mac ACT inference timing

Run `../../.venv/bin/python review/benchmark_act_inference.py` from this
pipeline directory. This times 30 single-observation ACT chunk predictions
after five warmups using a saved wrist image and joint state. It includes
preprocessing and conversion of the full chunk to NumPy; it excludes camera
capture and network transfer. It does not connect to the robot or write data.


## ACT timing and target-gap check

Run the focused ACT review on the Mac from the `Imitation Learning` directory:

```bash
../../.venv/bin/python review/act_teacher_forced_trace.py
```

It replays held-out episodes 6 and 63 through the 004000 ACT checkpoint. A new
30-action chunk starts every 12 recorded frames, becomes available 12 frames
later, and overlaps earlier predictions with the same queue used by the NUC
trial. This is an approximate timing model: inference latency varied on the
NUC, and the recorded robot followed the human rather than ACT. The output in
`reviews/act_teacher_forced_patch4_004000` shows expert commands, measured
positions, and ACT proposals for shoulder lift and gripper, plus their gaps.
It also saves a CSV and summary for each episode. Nothing is sent to a robot,
and the source dataset and checkpoint are only read.

The command refuses to overwrite an existing review. To repeat the diagnostic,
pass `--output reviews/ANOTHER_NAME`; choose other held-out IDs with
`--episodes 9 21`.

## Checkpoint screen

Run on the Mac from the `Imitation Learning` directory:

```bash
../../.venv/bin/python review/offline_review.py
```

This reads the 17 validation episodes selected in `experiment.json`, all three
BC checkpoints, and every numbered checkpoint in the patch-4 ACT run. It does
not train, connect to the robot, or change the dataset or checkpoints. It
refuses to overwrite a previous review directory; use `--output NEW_PATH` for
another run.

The default output, `reviews/patch4_offline_v1`, contains:

- `review.json`: first-action mean absolute error (MAE), per-joint MAE and
  95th-percentile absolute error, per-episode MAE, a high/low gripper aperture
  classification, ACT ten-step chunk MAE, and batch inference timing.
- `checkpoint_scorecard.png`: the sampled first-action screen for every model.
- `episode_*_joints.png`: six command curves, aligned to the command's target
  time. Future BC's output at observation `t` is drawn at `t+3`.
- `episode_*_gripper.png`: a larger gripper curve with the recorded low-aperture
  transitions marked.
- `episode_*_wrist.png`: camera frames around the first recorded low-aperture
  transition.

The screen samples 16 evenly spaced observations per validation episode by
default. Set `--samples-per-episode 32` or `64` for a denser check. The plots
use all frames from two validation episodes. Specify their IDs with, for
example, `--plot-episodes 6 63`. ACT's plotted actions are the first ten
commands of a new chunk every ten recorded frames, matching its configured
action queue. The tool chooses an ACT checkpoint for plotting using sampled
first-action MAE; `review.json` records that choice.

MAE is in recorded joint encoder-position units, averaged over the six joints.
ACT ten-step chunk error uses only unpadded future commands. BC future-residual
predicts three frames ahead, so its error is a distinct target from the
current-command errors. The high/low gripper measure uses a threshold of 15:
real grasp commands may remain above the fully closed pose while holding a
cube. Transition timing is heuristic, based on a seven-frame median-filtered
threshold crossing; inspect the curves and wrist frames before interpreting it.

All predictions are made from recorded camera images and observed follower
states. The review cannot establish closed-loop task success. Mac batch
inference times cannot establish the NUC's real-time performance.
