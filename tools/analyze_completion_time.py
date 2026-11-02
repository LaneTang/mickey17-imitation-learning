#!/usr/bin/env python3
"""Estimate place-and-return timing from observed motion, never quality labels.

This detects a motion signature, not visual task success. Source files are
read only; outputs must go to a separate, previously nonexistent directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
COLUMNS = [f"observed.arm_{joint}.pos" for joint in JOINTS]
DEFAULTS = dict(home_tolerance=7.0, departure_distance=15.0, open_rise=15.0,
                close_drop=15.0, release_rise=10.0, event_hold_s=0.25,
                home_hold_s=0.5, max_gap_s=0.2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect(rows, start, settings):
    """Ordered departure -> opening -> closing -> reopening -> home dwell.

    All six observed channels must return within tolerance. Completion is the
    start of the qualifying home dwell, confirmed only after home_hold_s.
    """
    if len(start) != 6 or not all(math.isfinite(x) for x in start):
        raise ValueError("Starting pose must contain six finite numbers")
    phase = "departure"
    since = None
    peak = low = None
    previous_t = None
    events = {}
    for row in rows:
        t = float(row["elapsed_s"])
        state = [float(row[key]) for key in COLUMNS]
        if not math.isfinite(t) or t < 0 or not all(math.isfinite(x) for x in state):
            raise ValueError("Nonfinite or negative observation/time")
        if previous_t is not None:
            if t <= previous_t:
                raise ValueError("Observation timestamps must strictly increase")
            if t - previous_t > settings["max_gap_s"]:
                since = None
        previous_t = t
        grip = state[5]
        if phase == "departure":
            condition = max(abs(a-b) for a, b in zip(state[:5], start[:5])) >= settings["departure_distance"]
        elif phase == "opening":
            condition = grip >= start[5] + settings["open_rise"]
        elif phase == "closing":
            peak = max(peak, grip)
            condition = grip <= peak - settings["close_drop"]
        elif phase == "release":
            low = min(low, grip)
            condition = grip >= low + settings["release_rise"]
        else:
            condition = max(abs(a-b) for a, b in zip(state, start)) <= settings["home_tolerance"]
        if not condition:
            since = None
            continue
        if since is None:
            since = t
        hold = settings["home_hold_s"] if phase == "home" else settings["event_hold_s"]
        if t - since < hold:
            continue
        events[phase + "_onset_s"] = round(since, 5)
        if phase == "home":
            return dict(candidate_completion_time_s=round(since, 5),
                        confirmed_at_s=round(t, 5), phase="home_confirmed", events=events)
        if phase == "departure":
            phase = "opening"
        elif phase == "opening":
            peak = grip
            phase = "closing"
        elif phase == "closing":
            low = grip
            phase = "release"
        else:
            phase = "home"
        since = None
    return dict(candidate_completion_time_s=None, confirmed_at_s=None,
                phase="awaiting_" + phase, events=events)


def analyze(raw, settings):
    """Deduplicate trial IDs; do not read quality_label.json or its ranks/times."""
    raw = Path(raw).resolve()
    if not raw.is_dir():
        raise ValueError("Raw outcome directory not found")
    trials = {}
    for path in sorted(raw.rglob("manifest.json")):
        run = path.parent
        if not run.name.startswith("act_offset0_overlap_"):
            continue
        cycles = run / "cycles.csv"
        hashes = {"manifest.json": digest(path), "cycles.csv": digest(cycles)}
        group = run.parent.relative_to(raw).as_posix()
        if run.name in trials:
            if hashes != trials[run.name]["source_sha256"]:
                raise ValueError("Conflicting duplicate trial: " + run.name)
            trials[run.name]["memberships"].append(group)
            continue
        manifest = json.loads(path.read_text())
        with cycles.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            result = dict(candidate_completion_time_s=None, confirmed_at_s=None,
                          phase="no_observations", events={})
        else:
            start = manifest.get("start_state", [float(rows[0][key]) for key in COLUMNS])
            result = detect(rows, start, settings)
        status = manifest.get("status", "unknown")
        candidate = result["candidate_completion_time_s"]
        trials[run.name] = dict(trial_id=run.name, memberships=[group],
                               rollout_status=status, alpha=manifest.get("arm_filter_alpha"),
                               source_sha256=hashes,
                               starting_pose_source="manifest.start_state" if "start_state" in manifest else "first_observation",
                               normal_ending_candidate=status == "time limit" and candidate is not None,
                               **result)
    if not trials:
        raise ValueError("No rollout manifests found")
    return list(trials.values())


def summarize(trials):
    groups = {}
    for trial in trials:
        for group in trial["memberships"]:
            groups.setdefault(group, []).append(trial)
    result = {}
    for group, items in sorted(groups.items()):
        times = [t["candidate_completion_time_s"] for t in items if t["normal_ending_candidate"]]
        result[group] = dict(n=len(items), normal_ending_candidates_n=len(times),
                             normal_ending_candidate_median_s=statistics.median(times) if times else None,
                             undetected_n=sum(t["candidate_completion_time_s"] is None for t in items),
                             abnormal_or_unfinished_n=sum(t["rollout_status"] != "time limit" for t in items))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    args = parser.parse_args()
    settings = {key: getattr(args, key) for key in DEFAULTS}
    if any(not math.isfinite(v) or v <= 0 for v in settings.values()):
        parser.error("Detector settings must be finite and positive")
    if settings["departure_distance"] <= settings["home_tolerance"]:
        parser.error("Departure distance must exceed home tolerance")
    raw, out = args.raw_dir.resolve(), args.output_dir.resolve()
    if out == raw or raw in out.parents or out in raw.parents:
        parser.error("Output must be separate from the raw input tree")
    if out.exists():
        parser.error("Output directory already exists; choose a new directory")
    trials = analyze(raw, settings)
    report = dict(schema_version=1, metric="observed_motion_place_and_return_candidate",
                  definition="Elapsed rollout time at onset of sustained six-channel home return after departure, opening, closing and reopening; includes initial inference hold.",
                  warning="Motion signature only: no visual placement verification. Labels are not read. Thresholds are exploratory; review candidates before reporting task completion.",
                  settings=settings, unique_trials=len(trials), groups=summarize(trials), trials=trials)
    out.mkdir(parents=True)
    (out / "completion_candidates.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ["trial_id", "memberships", "rollout_status", "alpha", "phase",
              "candidate_completion_time_s", "confirmed_at_s", "normal_ending_candidate"]
    with (out / "completion_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for trial in trials:
            writer.writerow({key: ";".join(trial[key]) if key == "memberships" else trial[key] for key in fields})
    print(json.dumps(dict(output_dir=str(out), unique_trials=len(trials),
                          detected_candidates=sum(t["candidate_completion_time_s"] is not None for t in trials),
                          note=report["warning"]), indent=2))


if __name__ == "__main__":
    main()
