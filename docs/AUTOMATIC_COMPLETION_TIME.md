# Automatic place-and-return timing

Run from the repository root with Python 3.10+; no robot libraries required:

```bash
python3 tools/analyze_completion_time.py \
  --raw-dir /path/to/e2e_rollout_official_act_offset0 \
  --output-dir /tmp/mickey17-completion-review
```

Choose a new output directory for each run. The script reads `manifest.json`
and `cycles.csv` only. It never reads `quality_label.json`, changes recordings,
or replaces existing published timing results. The full raw logs are needed;
the compact public snapshot contains only two selected trajectories.

## What it measures

The metric is a **motion-based completion candidate**, measured from rollout
start, including the initial inference hold. Using observed positions, it
requires these events in order:

1. An arm joint departs at least 15 recorded position units from the start.
2. The gripper opens at least 15 units above its starting value.
3. It closes at least 15 units below the subsequent observed opening peak.
4. It reopens at least 10 units above the subsequent closing minimum.
5. All six arm/gripper channels return within 7 units of the starting pose
   for at least 0.5 seconds.

Events 1–4 must each persist for 0.25 seconds. Observation gaps over 0.2 seconds
reset the pending dwell. Return time is the **onset** of the qualifying home
dwell; the confirmation timestamp is also recorded. The reference is the
manifest's measured `start_state`, or the first observation if absent.
Units are recorded joint-position units, not assumed physical degrees.

The script cannot establish grasp, transport, cube release at the destination,
or successful placement from joint positions alone. Empty-hand motion can
produce the same signature. Multiple attempts and unusual starting gripper
positions can defeat this simple ordered detector. It takes the first complete
signature. Missing detection means **undetected**, not task failure.

These thresholds are exploratory defaults, not validated task criteria or
controller settings. All are configurable through `--help`; adjust them from
inspection of recordings, not to improve success numbers. Check candidates
against video before presenting them as actual full-task completion times.

## Outputs

- `completion_candidates.csv`: one row per unique trial, memberships, runtime
  status, phase reached, candidate return time and confirmation time.
- `completion_candidates.json`: the same records, intermediate event times,
  detector settings, source SHA-256 hashes and per-folder summaries.

Shared baseline trial IDs are deduplicated and retain their memberships.
Conflicting copies raise an error. Abnormal/unfinished rollouts are identified
separately and excluded from the normal-ending candidate medians, even if a
motion signature occurred earlier. No result is assigned a task-success rank.
Folder summaries retain their original memberships, including supplementary
attempts; they are not automatically the 95-primary-trial study cohorts.

The existing label-based `completion_time_s` describes placement. This new
candidate measures return after a release-like sequence. They are different
metrics and should not replace one another silently.
