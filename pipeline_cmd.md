# Mikey17 imitation-learning command sheet

This is the **patch-4 command sheet** for the current six-joint BC and ACT
checkpoints. Run commands yourself on the computer named in each section.
Each code block is a separate step; later steps do not run automatically.
`IL_PIPELINE.md` at the project root explains the experiment loop and why a
failed trial returns to an earlier stage.

**Command labels:** `READ` observes without moving the robot; `MOTION` moves
the follower; `WRITE` creates a new artifact. Keep the camera-enabled LeKiwi
host running in a separate NUC terminal for all live steps. The robot base is
kept at zero velocity by the collection and deployment scripts. Start each
trial with the cube and mat reset and the follower at its saved default pose.

## Current comparison: same-frame residual BC, then ACT

The patch-4 dataset, split, three BC checkpoints, and ACT checkpoint already
exist. The residual-BC live preflight has passed. To continue the comparison,
start at **Stage 6B** below. The direct-BC trial and future-residual BC trials
are recorded in `reviews/`.

### Policy identities on the NUC

| `--policy` | Checkpoint | Weight SHA-256 |
| --- | --- | --- |
| `bc-direct` | `checkpoints/bc_direct_best.pt` | `5c6500fa4eccbbfd7281ca2c74f882f07279e9e27f40b45ecd0acd776884fc89` |
| `bc-residual` | `checkpoints/bc_residual_best.pt` | `7758e44b56c5e18dd273e7f782361cec82b980a236ef92c361d93509fd162447` |
| `bc-future-residual` | `checkpoints/bc_future_residual3_best.pt` | `c83723c1c7d791126f084e05c03055a4ad096b9e3f58926ff49d2887a8e21e47` |
| `act` | `checkpoints/act_004000` (whole directory) | `756f654209f5c8f746dea7a29c058ffa2f13ec841634176702ee1459b1f83dfc` for `model.safetensors` |

The ACT checkpoint needs its config and processor files alongside the weights.
Do not substitute the older archived ACT scripts: they expect nine state and
action channels, while this checkpoint uses six. The current
`pipeline_config.toml` sets the ACT **runtime** action window to 12 without
changing the saved 30-action checkpoint. Copy the TOML and `pipeline_settings.py`
with updated deployment scripts to the NUC before using these new commands.

**One-time code transfer for this config update.** On the NUC, first preserve
the current working code (this does not touch checkpoints or recordings):

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
cp deployment/preflight.py deployment/preflight.py.before_shared_config
cp deployment/shadow_bc.py deployment/shadow_bc.py.before_shared_config
cp deployment/rollout_bc_unbounded.py deployment/rollout_bc_unbounded.py.before_shared_config
```

Then on the **Mac**, from the pipeline directory, set `NUC` to your actual
Tailscale SSH address and copy the new config, loader, preflight, and shadow
scripts. This replaces the NUC scripts backed up above and adds
`shadow_act.py`; it does not change the saved checkpoint:

```bash
cd "/Users/tlane/Developer/Personal/ROB6003/lerobot/Atom's/Imitation Learning"
NUC="team13@YOUR_NUC_TAILSCALE_ADDRESS"
scp pipeline_config.toml pipeline_settings.py \
  "$NUC:/home/team13/Projects/lerobot/atom's/IL/"
scp deployment/preflight.py deployment/shadow_bc.py \
  deployment/rollout_bc_unbounded.py deployment/shadow_act.py \
  "$NUC:/home/team13/Projects/lerobot/atom's/IL/deployment/"
```

## Stage 1: assembly and calibration

**NUC + robot, manual.** Verify robot power, leader and follower calibration,
leader `/dev/ttyACM1`, follower host `/dev/ttyACM0`, wrist camera
`/dev/video0`, and that the base cannot roll into the workspace. These ports
are the ones currently configured in the scripts; recheck after reconnecting
USB devices. Calibration is not repeated just to run another checkpoint.

## Stage 2: task and criteria

**Manual.** Define the cube start area, target area, default arm pose, scene
variation, trial duration, and what counts as grasp, transport, release, and
success. Record scene changes when comparing policies. No command is needed
for this stage.

## Stage 3: scene, host, and teleoperation

**NUC terminal A — MOTION-capable host.** Activate the `(lerobot)` environment
and start the same camera-enabled host used for recording and deployment:

```bash
cd "/home/team13/Projects/lerobot"
python -m lerobot.robots.lekiwi.lekiwi_host \
  --robot.id=Mickey17_awesome_kiwi \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30}}' \
  --host.connection_time_s=6000
```

**NUC terminal B — MOTION under leader control.** Before a new demonstration,
shadow, or policy trial, align the follower to `IL/default_pose.json`:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python teleoperation/teleoperate_to_default.py
```

This script reports the live position difference; the leader, rather than the
saved JSON file, moves the follower. If a live preflight says the start pose is
outside tolerance, align again instead of changing the tolerance blindly.

## Stage 4: collect and review demonstrations

**NUC terminal B — MOTION + WRITE.** Example for a *new* patch; replace
`5th_patch` with its own versioned directory. The recorder teleoperates during
reset, starts on Enter, and creates the next numbered episode without
overwriting a previous one:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python demo_collection/record_demo_wrist_cam.py \
  --seconds 18 \
  --output-dir "/home/team13/Projects/lerobot/atom's/outputs/5th_patch"
```

**NUC terminal B — WRITE review files.** Plot one episode or summarize a
batch after recording. Replace the example path with the actual saved folder:

```bash
python demo_collection/plot_episode_commands.py \
  "/home/team13/Projects/lerobot/atom's/outputs/5th_patch/001_demonstration"
python demo_collection/plot_episode_commands.py \
  --batch "/home/team13/Projects/lerobot/atom's/outputs/5th_patch" \
  --output-dir "/home/team13/Projects/lerobot/atom's/outputs/5th_patch_review"
```

**Mac terminal — WRITE local copy.** Once a patch is complete, set `NUC` to
the SSH address you use from the Mac and copy the folder. Check the episode
count and video readability before merging; keep the NUC originals:

```bash
cd "/Users/tlane/Developer/Personal/ROB6003"
NUC="team13@YOUR_NUC_TAILSCALE_ADDRESS"
scp -r "$NUC:/home/team13/Projects/lerobot/atom's/outputs/5th_patch" \
  "Mickey17 remote/outputs/"
```

The collection recorder, not the policy deployment scripts, creates
demonstration labels.

## Stage 5: Mac dataset, training, offline review

**Mac.** Patch 4 has already been merged and split. For a *new* patch, first
set these names to match the directory you downloaded. Use a new dataset name
and output paths for each patch. The examples below use patch 5; the raw
episode directory must exist before you continue. Keep using the same Mac
terminal so these shell variables remain defined.

```bash
cd "/Users/tlane/Developer/Personal/ROB6003/lerobot/Atom's/Imitation Learning"
RAW_ROOT="../../../Mickey17 remote/outputs/5th_patch"
MERGED="datasets/patch5_merged_all"
ARM_ONLY="datasets/patch5_merged_arm_only"
SPLIT="experiments/patch5.json"
REPO="local/mikey17_patch5_all"
ARM_REPO="local/mikey17_patch5_arm_only"
../../.venv/bin/python data/merge.py \
  --raw-root "$RAW_ROOT" \
  --output "$MERGED" \
  --repo-id "$REPO"
```

The merge command is a **READ preview**. Check its episode count and source
mapping. Run the same command with `--merge` only when those are right; it
creates `$MERGED` and leaves the raw episodes untouched. It refuses to use
an output path that already exists, even in preview mode. You can add
`--expected-episodes N` when you know the exact count.

After merging, preview the ACT-specific six-channel copy:

```bash
../../.venv/bin/python data/derive_arm_only.py \
  --source "$MERGED" \
  --output "$ARM_ONLY" \
  --repo-id "$ARM_REPO"
```

Run that command with `--create` to write a separate copy. Then create one
whole-episode split for both policy families; the script refuses to overwrite
an existing split:

```bash
../../.venv/bin/python data/make_split.py \
  --dataset-root "$MERGED" \
  --output "$SPLIT" \
  --repo-id "$REPO"
```

**Mac — READ preview, then WRITE training only with `--train`.** Each BC
command trains a separate model and checkpoint. ACT trains a fourth model.
These patch-4 runs are already complete. For a new patch, preview each run
using the new split. ACT also needs the new arm-only dataset path:

```bash
../../.venv/bin/python train/train_bc_direct.py --experiment "$SPLIT"
../../.venv/bin/python train/train_bc_residual.py --experiment "$SPLIT"
../../.venv/bin/python train/train_bc_future_residual.py --experiment "$SPLIT"
../../.venv/bin/python train/train_act.py --experiment "$SPLIT" --arm-only-root "$ARM_ONLY"
```

Add `--train` to exactly the command you intend to start. The current offline
review script has a patch-4 ACT dataset path inside its code, so the command
below reviews **patch 4 only**. It needs an update before reviewing patch 5:

```bash
../../.venv/bin/python review/offline_review.py \
  --output reviews/patch4_offline_next_review
```

This writes plots and metrics but does not change checkpoints. Compare held-
out episodes, especially gripper transition timing. After choosing a model,
transfer the exact checkpoint to NUC `IL/checkpoints/`, preserve its processor
files if ACT, and compare SHA-256 on both machines. Patch-4 checkpoints are
already present on the NUC and their digests appear in the policy table above.

**Mac terminal — WRITE NUC checkpoint copy.** For a *new* residual-BC run,
replace the timestamped source path with the run you chose. Use a new
destination name, so an earlier selected checkpoint is not overwritten:

```bash
BC_CHECKPOINT="runs/bc_residual_YYYYMMDD_HHMMSS/bc_residual_best.pt"
shasum -a 256 "$BC_CHECKPOINT"
scp "$BC_CHECKPOINT" \
  "$NUC:/home/team13/Projects/lerobot/atom's/IL/checkpoints/bc_residual_patch5_best.pt"
```

On the **NUC**, run `sha256sum` on the copied file and compare the full digest.
For ACT, copy its entire selected `pretrained_model/` directory to a new
checkpoint directory; its processor and config files are required. The
deployment `--checkpoint` argument must use that exact copied path.

## Stage 6A: NUC checkpoint and live preflight

**NUC terminal B — READ.** First load the selected checkpoint without robot
contact; then add `--live` to read one camera frame and measured pose. Example
for the current same-frame residual BC candidate:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/preflight.py \
  --policy bc-residual \
  --checkpoint checkpoints/bc_residual_best.pt
python deployment/preflight.py \
  --policy bc-residual \
  --checkpoint checkpoints/bc_residual_best.pt \
  --live \
  --default-pose default_pose.json
```

The current residual-BC live preflight has passed. For ACT, use the **whole
checkpoint directory**, not just `model.safetensors`:

```bash
python deployment/preflight.py \
  --policy act \
  --checkpoint checkpoints/act_004000
python deployment/preflight.py \
  --policy act \
  --checkpoint checkpoints/act_004000 \
  --live \
  --default-pose default_pose.json
```

ACT currently runs on the NUC CPU. The configured twelve-action window is
400 ms at 30 Hz; check the reported inference time before planning a live ACT
controller.
Preflight never sends a policy action.

## Stage 6B: NUC leader-controlled shadow

**NUC terminal B — MOTION under leader control.** This is the **next current
command** for same-frame residual BC. The leader drives the follower through
one full task. The BC predictions are logged, never sent:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/shadow_bc.py \
  --policy bc-residual \
  --checkpoint checkpoints/bc_residual_best.pt \
  --default-pose default_pose.json \
  --seconds 20 \
  --enable-leader-motion
```

Review the reported rate, overruns, per-phase gripper error, and saved
`runs/shadow_bc/<run>/cycles.csv`. The same script accepts `bc-direct` and
`bc-future-residual` with their matching checkpoint paths from the table.

**NUC terminal B — MOTION under leader control; ACT proposals logged only.**
After copying `deployment/shadow_act.py`, updated `deployment/preflight.py`,
updated `deployment/shadow_bc.py`, `pipeline_settings.py`, and
`pipeline_config.toml` from the Mac pipeline to the same locations on the NUC,
run a full-task ACT shadow. The script uses the config's 12-action window and
runs inference in a background thread so it can measure timing without sending
policy commands:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/shadow_act.py \
  --checkpoint checkpoints/act_004000 \
  --default-pose default_pose.json \
  --seconds 20 \
  --enable-leader-motion
```

Inspect `runs/shadow_act/<run>/manifest.json`, `control.csv`, and
`predictions.csv`. The report compares ACT predictions with time-aligned
leader commands and shows inference time versus the 400 ms action window.

**ACT timing rehearsal before any policy-controlled ACT trial.** Copy the
updated Mac `deployment/shadow_act.py` to the same NUC path. Repeat the
leader-controlled command above with `--full-chunk` added. This logs all 30
predictions per inference while keeping the 12-step inference cadence and
does not send policy actions. After copying the new run directory or a tar
archive to the Mac, run:

```bash
cd "/Users/tlane/Developer/Personal/ROB6003/lerobot/Atom's/Imitation Learning"
../../.venv/bin/python review/act_timing_rehearsal.py \
  --log "/path/to/new_act_shadow_run_or_archive.tar.gz" \
  --output-dir reviews/act_timing_rehearsal_01
```

Check `summary.json`, `handoffs.csv`, and `timing.png`. This offline replay
uses leader-driven observations, so it can expose stale predictions and large
handoffs but cannot establish closed-loop task success.

## Stage 7: NUC supervised trial and review

**NUC terminal B — MOTION by BC policy.** After reviewing that policy's
shadow log, preview its first raw target without `--arm`. Example for
same-frame residual BC:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/rollout_bc_unbounded.py \
  --policy bc-residual \
  --checkpoint checkpoints/bc_residual_best.pt \
  --expected-sha256 7758e44b56c5e18dd273e7f782361cec82b980a236ef92c361d93509fd162447 \
  --default-pose default_pose.json \
  --seconds 2
```

Omitting `--arm` is a **READ preview**. To permit motion, repeat the same
command with `--arm` and type `RUN` only after inspecting the actual first
target with the workspace clear and a person at the power switch. The raw
script has **no arm/gripper target clipping or travel cap**; `--seconds` is
the operator-chosen duration. Begin with two seconds and inspect the measured
motion before choosing a longer task trial. The guarded `rollout_bc.py` is a
different option with short-trial motion limits.

Before the next trial, copy the updated Mac
`deployment/rollout_bc_unbounded.py` to NUC `IL/deployment/` if you want new
folders named like `bc_residual_No001_<UTC timestamp>`. The numbering reads
older manifests in `runs/rollout_bc_unbounded/` and never renames them.

After a trial, inspect its printed log path and `manifest.json`; copy the run
folder to the Mac for plots and failure analysis. `requested.*` means a
command the client queued, **not** proof the NUC motor acknowledged it.
`observed.*` shows the follower's reported joint positions. Record cube grasp,
transport, release, placement, any contact, and whether an operator stopped
the robot. A completed time limit alone is not task success.

**ACT overlap rehearsal:** After a full-chunk shadow log is on the Mac, run
`review/act_overlap_rehearsal.py` with `--log <shadow folder>` and a new
`--output-dir <review folder>`. It compares the old hard chunk switch with
time-aligned overlap and writes a plot, CSV, and summary. Its recorded
observations came from leader control, so it cannot prove policy success.

**ACT supervised trial:** `deployment/rollout_act_overlap.py` and
`deployment/act_overlap_queue.py` are the current Mac-side controller files.
Copy both into NUC `IL/deployment/` before trying them. The controller merges
overlapping future actions with a gradual handoff and keeps the two-second,
per-cycle, target-lead, start-pose, and travel limits. Omit `--arm` for its
read-only preview. `--arm` requires an interactive `RUN` and a person at the
power switch. The old `rollout_act_unbounded.py` is retired after its abrupt
chunk switches produced severe motion. Do not use it again. Do not use a BC
rollout command with an ACT checkpoint.

**Mac terminal — transfer only, no robot motion:**

```bash
cd "/Users/tlane/Developer/Personal/ROB6003/lerobot/Atom's/Imitation Learning"
scp deployment/act_overlap_queue.py deployment/rollout_act_overlap.py \
  "team13-mini-s:/home/team13/Projects/lerobot/atom's/IL/deployment/"
```

**NUC terminal — preview only, no robot motion:**

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/rollout_act_overlap.py \
  --checkpoint checkpoints/act_004000 \
  --expected-sha256 756f654209f5c8f746dea7a29c058ffa2f13ec841634176702ee1459b1f83dfc \
  --default-pose default_pose.json
```

The preview verifies alignment, checksum, first proposal, and first metered
target. Do not add `--arm` before reviewing its output and clearing the
workspace. A two-second trial is a motion check, not a grasp success test.
