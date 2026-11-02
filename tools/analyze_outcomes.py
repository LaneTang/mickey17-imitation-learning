#!/usr/bin/env python3
"""Offline outcome audit and scientific figures. Never imports robot code.

Default: published numerical snapshot. --raw-dir: read-only import of extracted
November 1 outcomes. All writes go to --output-dir; originals are never edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "f9bec52554f3706d93810b7e9e6fade4a4783c83b54151df8f5f4fb58c3ad719"
MODEL_SHA = "c2de68211a91d5fda1dbd47cb57aaef8fa5945df824363ed9f93fb06147a6528"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GROUPS = {
    "alpha": [("alpha/alpha-" + x, x) for x in ["0.1", "0.2", "0.5", "0.8", "1[baseline]"]],
    "base": [("base/" + x, label) for x, label in [
        ("origin[baseline]", "Origin"), ("2cm_left", "2 cm left"), ("2cm_right", "2 cm right")]],
    "orientation": [("cube/deg" + x, x.replace("[baseline]", "") + "°")
                    for x in ["0[baseline]", "10", "30", "45", "60", "80"]],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def wilson(k, n):
    """Two-sided Wilson score interval, z=1.959963984540054 (95%)."""
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - half), min(1., center + half)]


def controller_error(status):
    if not status.startswith("error:"):
        return None
    if "queue ran dry" in status:
        return "queue exhausted"
    if "averaged proposal" in status:
        return "averaged-step limit"
    if "proposal gap" in status or "proposal-gap" in status or "demonstration envelope" in status:
        return "proposal envelope"
    return "other runtime error"


def outcome(trial, threshold):
    """Abnormal is failure, missing/unfinished is unknown, rank zero is valid."""
    if trial["status"].startswith("error:") or trial["status"] == "operator Ctrl-C":
        return False
    if trial["status"] != "time limit" or trial["label_missing"] or trial["quality_rank"] is None:
        return None
    return trial["quality_rank"] >= threshold


def select_trajectories(trials):
    eligible = sorted((t for t in trials if "alpha/alpha-0.5" in t["memberships"]
                       and outcome(t, 5) is True and t["completion_time_s"] is not None),
                      key=lambda t: (t["completion_time_s"], t["trial_id"]))
    failures = sorted((t for t in trials if "alpha/alpha-1[baseline]" in t["memberships"]
                       and controller_error(t["status"])), key=lambda t: t["trial_id"])
    # For an even number choose the lower middle (documented deterministic tie rule).
    return ([eligible[(len(eligible) - 1) // 2]["trial_id"]] if eligible else []) + (
        [failures[0]["trial_id"]] if failures else [])


def load_raw(raw):
    raw = Path(raw)
    if not raw.is_dir():
        raise ValueError("--raw-dir must be the extracted e2e_rollout_official_act_offset0 directory")
    trials, directories, memberships = {}, {}, []
    for path in sorted(raw.rglob("manifest.json")):
        run = path.parent
        if not run.name.startswith("act_offset0_overlap_"):
            continue
        trial_id = run.name
        group = run.parent.relative_to(raw).as_posix()
        manifest = json.loads(path.read_text())
        label_path = run / "quality_label.json"
        label = json.loads(label_path.read_text()) if label_path.exists() else {}
        # Hash every source file, including images, without publishing raw images.
        hashes = {p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob("*")) if p.is_file()}
        memberships.append({"trial_id": trial_id, "folder": group, "files_sha256": hashes})
        if trial_id in trials:
            if hashes != trials[trial_id]["files_sha256"]:
                raise ValueError("Conflicting duplicate trial: " + trial_id)
            trials[trial_id]["memberships"].append(group)
            continue
        with (run / "cycles.csv").open() as f:
            cycles = list(csv.DictReader(f))
        inference = {}
        with (run / "chunks.csv").open() as f:
            for row in csv.DictReader(f):
                key = int(row["source_cycle"])
                value = float(row["inference_ms"])
                if key in inference and inference[key] != value:
                    raise ValueError("Inconsistent inference timing: " + trial_id)
                inference[key] = value
        alpha = manifest["arm_filter_alpha"]
        cohort = ("preliminary" if group == "test" else
                  "extra_base_right" if group == "base/2cm_right" and alpha != .5 else "primary")
        trial = {
            "trial_id": trial_id, "memberships": [group], "cohort": cohort,
            "alpha": alpha, "status": manifest["status"],
            "quality_rank": label.get("quality_rank"), "label_missing": not label_path.exists(),
            "label_format_version": label.get("format_version"),
            "completion_time_s": label.get("completion_time_s"),
            "circle_rating": label.get("cube_in_the_circle", {}).get("rating"),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "settings_sha256": manifest["settings_sha256"],
            "limits_sha256": manifest["controller_limits_sha256"],
            "logged_cycles": len(cycles), "manifest_cycles": manifest.get("cycles"),
            "label_logged_cycles": label.get("logged_cycles"),
            "label_status": label.get("rollout_status"),
            "clipped_cycles": manifest.get("clipped_cycles"),
            "inference_ms": list(inference.values()), "files_sha256": hashes,
        }
        if cohort == "primary":
            if label.get("quality_score") is not None:
                raise ValueError("Legacy score encountered in primary trial: " + trial_id)
            if manifest["status"] != "started" and manifest.get("cycles") != len(cycles):
                raise ValueError("Cycle-count mismatch: " + trial_id)
            if label_path.exists() and label.get("logged_cycles") != len(cycles):
                raise ValueError("Label cycle-count mismatch: " + trial_id)
            if label_path.exists() and label.get("rollout_status") != manifest["status"]:
                raise ValueError("Status mismatch: " + trial_id)
        trials[trial_id] = trial
        directories[trial_id] = run
    if not trials:
        raise ValueError("No trials found; check --raw-dir")
    records = list(trials.values())
    selected = select_trajectories(records)
    trajectories = []
    columns = ["elapsed_s"] + [f"{stage}.arm_{joint}.pos" for stage in
                               ["observed", "proposed", "filtered", "requested", "sent"] for joint in JOINTS]
    for trial_id in selected:
        with (directories[trial_id] / "cycles.csv").open() as f:
            rows = [[round(float(row[c]), 6) for c in columns] for row in csv.DictReader(f)]
        trajectories.append({"trial_id": trial_id, "columns": columns, "rows": rows,
                             "precision": "six decimals; controller command units, not calibrated joint degrees"})
    return {"schema_version": 1, "source_archive": "mickey17-outcomes-20261101.tar",
            "source_archive_sha256": ARCHIVE_SHA,
            "source_note": "Imported from checksum-verified archive; file hashes refer to originals.",
            "trials": records, "folder_registry": memberships, "trajectories": trajectories}


def group_summary(trials):
    n = len(trials)
    placed = sum(outcome(t, 4) is True for t in trials)
    returned = sum(outcome(t, 5) is True for t in trials)
    unknown = sum(outcome(t, 4) is None for t in trials)
    times = [t["completion_time_s"] for t in trials if outcome(t, 4) is True
             and t["completion_time_s"] is not None]
    return {"n": n, "placement": placed, "return": returned, "unknown": unknown,
            "placement_wilson95": wilson(placed, n), "return_wilson95": wilson(returned, n),
            "controller_errors": sum(controller_error(t["status"]) is not None for t in trials),
            "timed_n": len(times), "placement_time_median_s": statistics.median(times) if times else None}


def analyze(snapshot):
    trials = snapshot["trials"]
    ids = [t["trial_id"] for t in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("Snapshot contains duplicate trial records")
    for t in trials:
        rank = t["quality_rank"]
        if rank is not None and (type(rank) is not int or not 0 <= rank <= 5):
            raise ValueError("Rank outside 0–5: " + t["trial_id"])
        seconds = t["completion_time_s"]
        if seconds is not None and (not isinstance(seconds, (int, float)) or
                                    not math.isfinite(seconds) or not 0 <= seconds <= 40):
            raise ValueError("Invalid placement time: " + t["trial_id"])
        if t["cohort"] == "primary" and t["checkpoint_sha256"] != MODEL_SHA:
            raise ValueError("Unexpected primary checkpoint")
    summary = {}
    for family, groups in GROUPS.items():
        for folder, label in groups:
            members = [t for t in trials if t["cohort"] == "primary" and folder in t["memberships"]]
            summary[folder] = {"family": family, "label": label.replace("[baseline]", ""),
                               **group_summary(members)}
    primary = [t for t in trials if t["cohort"] == "primary"]
    inference = [v for t in primary for v in t["inference_ms"]]
    audit = {
        "unique_trials": len(trials), "folder_memberships": sum(len(t["memberships"]) for t in trials),
        "cohorts": dict(Counter(t["cohort"] for t in trials)),
        "primary_label_missing": [t["trial_id"] for t in primary if t["label_missing"]],
        "all_label_missing": [t["trial_id"] for t in trials if t["label_missing"]],
        "placed_without_time": [t["trial_id"] for t in primary if outcome(t, 4) is True
                                and t["completion_time_s"] is None],
        "shared_baseline_ids": [t["trial_id"] for t in trials if len(t["memberships"]) > 1],
        "selected_trajectory_ids": select_trajectories(trials),
        "inference_completed_chunks_n": len(inference),
        "inference_median_ms": statistics.median(inference) if inference else None,
        "primary_controller_errors": dict(Counter(controller_error(t["status"]) for t in primary
                                                  if controller_error(t["status"]))),
        "supplementary": {cohort: {"n": len(ts), "statuses": dict(Counter(t["status"] for t in ts)),
                                    "alpha_counts": dict(Counter(str(t["alpha"]) for t in ts)),
                                    "missing_labels": sum(t["label_missing"] for t in ts)}
                          for cohort in ["extra_base_right", "preliminary"]
                          for ts in [[t for t in trials if t["cohort"] == cohort]]},
    }
    return summary, audit


def write_tables(snapshot, summary, out):
    with (out / "trial_registry.csv").open("w", newline="") as f:
        fields = ["trial_id", "cohort", "memberships", "alpha", "status", "quality_rank",
                  "label_missing", "completion_time_s", "circle_rating", "logged_cycles"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for t in snapshot["trials"]:
            writer.writerow({k: ";".join(t[k]) if k == "memberships" else t[k] for k in fields})
    lines = ["| Configuration | Placement (95% CI) | Placement + return (95% CI) | Controller errors | Placement time, median (timed n) |",
             "| --- | --- | --- | --- | --- |"]
    for folder, s in summary.items():
        def metric(key):
            lo, hi = s[key + "_wilson95"]
            return f"{s[key]}/{s['n']} ({100*lo:.0f}–{100*hi:.0f}%)"
        timing = f"{s['placement_time_median_s']:.2f} s ({s['timed_n']})" if s["timed_n"] else "— (0)"
        lines.append(f"| {folder} | {metric('placement')} | {metric('return')} | {s['controller_errors']}/{s['n']} | {timing} |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")


def figures(snapshot, summary, out):
    from showcase_figures import render_figures
    render_figures(snapshot, summary, out, GROUPS, JOINTS, outcome, wilson)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "evidence/2026-11-01/outcomes.json")
    parser.add_argument("--raw-dir", type=Path, help="Extracted outcome directory (read only); overrides --snapshot")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/2026-11-01")
    parser.add_argument("--no-figures", action="store_true", help="Standard-library-only audit and tables")
    args = parser.parse_args()
    snapshot = load_raw(args.raw_dir) if args.raw_dir else json.loads(args.snapshot.read_text())
    summary, audit = analyze(snapshot)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "summary.json", summary)
    dump(out / "audit.json", audit)
    write_tables(snapshot, summary, out)
    if args.raw_dir:
        dump(out / "outcomes.json", snapshot)
    if not args.no_figures:
        figures(snapshot, summary, out)
    print(json.dumps({"output_dir": str(out), **audit}, indent=2))


if __name__ == "__main__":
    main()
