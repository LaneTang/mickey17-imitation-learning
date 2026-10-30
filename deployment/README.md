# Deployment preflight

## Offline NUC ACT timing profile

The latency-12 ACT variant requires the matching
`src/lerobot/policies/act/configuration_act.py` on both Mac and NUC. Its
checkpoint records the action-label offset, and the loader rejects a NUC
LeRobot install that does not interpret it. For this variant, use
`rollout_act_overlap.py` only after an offset-aware shadow review; the older
ACT rollout scripts reject shifted checkpoints. No supervised trial is
authorized by transferring files or passing preflight.

`profile_act_cpu.py` reads one saved wrist snapshot and start pose from a past
rollout. It separates tensor creation, preprocessing, ACT model inference,
postprocessing, and NumPy conversion. It does not connect to the robot or
change any data. Run it on the NUC in the `(lerobot)` environment with
`--run runs/rollout_act_overlap/YOUR_RUN_DIRECTORY`; the default checkpoint is
`checkpoints/act_004000`. Camera capture and robot/network time are excluded.
Use `--cpu-threads 1` or `--cpu-threads 4` to benchmark a thread count without
changing the shared pipeline settings or the robot controller.


`preflight.py` is a read-only gate for the three new BC checkpoint types and
both six-channel and official nine-channel ACT checkpoints. It loads the model, validates its input/output
shape and joint order, prints a SHA-256 digest, and checks the arm-only adapter
that would add `x.vel = y.vel = theta.vel = 0` to a LeKiwi command. It never
sends that command.

## Step 1: checkpoint-only on the Mac

From the `Imitation Learning` directory, in the LeRobot virtual environment:

```bash
../../.venv/bin/python deployment/preflight.py \
  --policy bc-residual \
  --checkpoint runs/bc_residual_20261011_045054/bc_residual_best.pt
```

For the ACT candidate:

```bash
../../.venv/bin/python deployment/preflight.py \
  --policy act \
  --checkpoint runs/act_arm_only_20261011_051140/checkpoints/004000/pretrained_model
```

The ACT checkpoint path must be the whole `pretrained_model` directory,
including model weights and both processor files. The BC types are
`bc-direct`, `bc-residual`, and `bc-future-residual`, each requiring its
corresponding checkpoint. Checkpoint-only mode does not connect to Mikey17.

## Step 2: live, read-only observation on the NUC

The canonical NUC runtime is
`/home/team13/Projects/lerobot/atom's/IL`. It contains the existing
`demo_collection/` and `default_pose.json`, plus `deployment/`, `train/`
(the two BC import dependencies), `checkpoints/`, and `runs/` after migration.
The Mac workspace retains the
merged datasets, training runs, and offline reviews. Raw NUC demonstrations
stay in `atom's/outputs`.

If files were previously copied to `/home/team13/mikey17_patch4_preflight`,
transfer the **updated Mac copy** of `deployment/migrate_nuc_layout.sh` before
using it on the NUC; an older copy may still target the discarded
`Imitation Learning` path. The updated script previews the move by
default and refuses conflicting files. Only `--apply` moves them into the
existing `IL` directory. It leaves `IL/default_pose.json` and raw episodes
untouched. Confirm the staging directory's actual location before applying.

After the migration, run preflight in the NUC `(lerobot)` environment with the
camera-enabled LeKiwi host running:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/preflight.py \
  --policy bc-residual \
  --checkpoint checkpoints/bc_residual_best.pt \
  --live \
  --default-pose default_pose.json
```

`--live` reads one observation, confirms the wrist frame is 480×640 uint8,
checks all six measured joints against the saved default pose, predicts one
action or ACT chunk, and prints the first target and its difference from the
measured pose. The default start tolerance is **7 recorded position units per
joint**; adjust it explicitly only after inspecting the observed difference.
This is an alignment check against your captured pose, not a mechanical limit
or proof of safe motion. The script has no `send_action` call and does not move
the follower or base.

The ACT checkpoint also reports inference time for one chunk on the NUC.
`pipeline_config.toml` retains the checkpoint's 30-action chunk and selects
a **12-action runtime window** (400 ms at 30 Hz). Preflight reports both the
saved 10-action setting and the runtime setting. A chunk slower than the
runtime window needs a different deployment timing design before a
policy-controlled trial. Passing preflight does not authorize autonomous
motion; shadow testing is the next stage.

## Step 3: leader-controlled BC shadow test

`shadow_bc.py` uses the new BC checkpoint format. It commands the follower
from the SO-101 leader and sets base velocity to zero. BC predictions are
logged only; they never enter `send_action`. Unlike preflight, **this script
does move the follower under your leader control**. It requires
`--enable-leader-motion`, verifies both the follower's default pose and the
initial leader/follower gap, limits the run to at most 60 seconds, and queues a
hold-current-position command at exit.

Copy `deployment/shadow_bc.py` into the canonical NUC runtime. With the
LeKiwi host running, the workspace clear, and the
leader near the saved start pose, run in the NUC `(lerobot)` environment:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/shadow_bc.py \
  --policy bc-residual \
  --checkpoint checkpoints/bc_residual_best.pt \
  --default-pose default_pose.json \
  --seconds 20 \
  --enable-leader-motion
```

The script reports control-loop rate, overruns, policy-versus-leader command
error, gripper error, and inference time every five seconds. It saves
`cycles.csv`, a checkpoint digest, and wrist snapshots under
`runs/shadow_bc/bc_shadow_*`. A low gap here means agreement with the leader
on these trajectories; it is not evidence that closed-loop autonomy succeeds.
Use Ctrl-C to stop the supervised run. Do not use the archived `bc_shadow.py`
for these new checkpoints: it expects the older normalization fields.

### ACT shadow test

`shadow_act.py` uses the same leader-only motion rule. A background worker
predicts 30-action ACT chunks, logs the first 12 actions selected by
`pipeline_config.toml`, and never sends them to the robot. It records leader
commands and time-aligned predictions separately so inference delays remain
visible. Copy `pipeline_config.toml`, `pipeline_settings.py`, updated
`preflight.py` and `shadow_bc.py`, and `shadow_act.py` to their matching NUC
paths as described in [pipeline_cmd.md](../pipeline_cmd.md). Then run:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/shadow_act.py \
  --checkpoint checkpoints/act_004000 \
  --default-pose default_pose.json \
  --seconds 20 \
  --enable-leader-motion
```

This moves the follower **only under leader control**. The saved
`control.csv`, `predictions.csv`, and manifest under `runs/shadow_act/`
support timing and grasp/release review. There is no six-joint ACT autonomous
rollout in this shadow step.

The official patch-4 ACT checkpoint has nine state and action values: six arm
and gripper positions followed by `x.vel`, `y.vel`, `theta.vel`. Preflight
passes all nine live observation values to that model. The shadow test logs
all nine predictions in `predictions.csv`, compares the first six against the
leader, and sends the leader's six arm targets with zero base velocity. The
overlap rollout accepts this nine-channel checkpoint. It passes all nine
observation values into ACT, logs the predicted base velocities, and commands
the six arm joints with zero base velocity. On the NUC, point `--checkpoint`
at the transferred `pretrained_model` folder for step 018000.

For a timing rehearsal, add `--full-chunk` to the same command. It still starts
one inference approximately every 12 control cycles, but logs all 30 predicted
actions from each result. The extra actions are recorded only; the model,
checkpoint, and robot commands are unchanged. Copy or archive that new run to
the Mac and review it without a robot connection:

```bash
cd "/Users/tlane/Developer/Personal/ROB6003/lerobot/Atom's/Imitation Learning"
../../.venv/bin/python review/act_timing_rehearsal.py \
  --log "/path/to/new_act_shadow_run_or_archive.tar.gz" \
  --output-dir reviews/act_timing_rehearsal_01
```

The review saves per-cycle candidate selection, chunk handoff gaps, a summary,
and a timing plot. The first inference uses a stationary-start assumption;
later selections use the action index corresponding to the observation's age.
These are **counterfactual proposals** evaluated on leader-driven observations,
not a motor test or evidence of autonomous success. Inspect large handoff gaps
before designing an ACT rollout.

### ACT overlap rehearsal and supervised trial

The first `rollout_act.py` trial stopped at a chunk jump. A later raw ACT
rollout produced severe motion because it sent those jumps to the follower.
**Do not run `rollout_act_unbounded.py` again.** Keep its log for analysis.

`review/act_overlap_rehearsal.py` replays a full-chunk shadow log with the old
hard switch and a queue that gradually blends predictions for the same future
control cycle. It creates a comparison plot and numerical summary without
connecting to the robot. This replay uses leader-driven observations and
cannot show whether the ACT policy will complete the task on its own.

`rollout_act_overlap.py` and `act_overlap_queue.py` form the supervised ACT
controller. The overlap queue draws on LeRobot's asynchronous
future-action aggregation, with an extra six-cycle ramp for new predictions.
The controller defaults to a **two-second maximum**. `--full-trial` permits up
to 20 seconds and removes the start-relative travel window. Both modes retain
checkpoint and default-pose checks, a stopped base, per-cycle joint limits,
target-lead and demonstration-envelope limits, a maximum proposal gap, and a
stop if the queue runs dry. These are
software checks, not collision detection or mechanical joint limits. Its
preview without `--arm` sends no command. With `--arm`, interactive `RUN` is
also required. The operator must review the preview and remain at the power
switch. Run logs go to `runs/rollout_act_overlap/`. `chunks.csv` records raw
model predictions, including base predictions for nine-channel checkpoints;
`cycles.csv` records the selected, limited, and sent arm targets alongside
observed positions.

## Step 4: short supervised BC rollout

`rollout_bc.py` loads any one of the three patch-4 BC checkpoints and commands
the follower arm from its predictions. The first trial is limited to **two
seconds**. It checks the checkpoint digest, saved start pose, demonstrated
command envelope, prediction gap, per-cycle step, target lead over the measured
position, total displacement, camera frame shape, and loop timing. The target
can advance from its previous request when a joint has not yet moved, but it
cannot get more than three position units ahead of an arm joint (four for the
gripper). Every action sets all three base velocities to zero. These guards
are not mechanical joint limits or collision detection.

Run it only with someone beside the robot who can stop power immediately.
The camera-enabled LeKiwi host must already be running. After copying the
script to NUC `IL/deployment`, a first direct-BC trial uses:

```bash
cd "/home/team13/Projects/lerobot/atom's/IL"
python deployment/rollout_bc.py \
  --policy bc-direct \
  --checkpoint checkpoints/bc_direct_best.pt \
  --expected-sha256 5c6500fa4eccbbfd7281ca2c74f882f07279e9e27f40b45ecd0acd776884fc89 \
  --default-pose default_pose.json \
  --seconds 2 \
  --arm
```

Before `RUN`, it prints the current state, raw policy proposal, metered first
target, and command delta. Entering anything else cancels without motion. With
`--arm` omitted, it is preview-only. The script logs `cycles.csv`, a manifest,
and wrist snapshots under `runs/rollout_bc`. The CSV's `requested.*` fields
are commands queued by the client, not motor acknowledgements; use the later
`observed.*` fields to check actual motion. A two-second trial may stop before
the cube is reached; review the log and physical motion before changing trial
duration or guards. Do not use the archived `bc_guarded_rollout.py`, which
expects a second-patch checkpoint and different start limits.

### Raw BC rollout option

`rollout_bc_unbounded.py` is a separate, operator-supervised option. It sends
the six finite BC outputs directly as absolute arm/gripper targets: no 10-unit
proposal-gap stop, per-cycle clipping, target-lead cap, or total-travel cap.
The operator chooses any positive duration with `--seconds`. It still verifies
the checkpoint and start pose, requires `--arm` plus interactive `RUN`, keeps
the base velocity at zero, and records observations and raw requests. Its
`requested.*` CSV fields are local requests, not motor acknowledgements. Use
its preview without `--arm` to inspect the first raw target before permitting
motion. New run folders include the policy and a per-policy number, for
example `bc_future_residual_No004_20261011_...`. The numbering counts older
`bc_raw_*` folders with matching manifests in the same log root; it does not
rename or edit those older runs. Keep the guarded script and its logs for
comparison.
