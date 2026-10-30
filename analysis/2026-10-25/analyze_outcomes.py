"""Read original rollout files and regenerate derived outcome figures and JSON.

Run from any directory with the project's Python environment. No robot imports.
"""
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "Mickey17 remote/outputs/outcome!"
COLORS = {"fast": "#167d9a", "full": "#69bac5", "incomplete": "#e7ad61",
          "null_failure": "#ce6471", "unknown": "#bdc5d0"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.edgecolor": "#d9dfe6",
                     "text.color": "#203247", "axes.labelcolor": "#203247",
                     "xtick.color": "#52677b", "ytick.color": "#203247",
                     "figure.facecolor": "#f6f8fb", "axes.facecolor": "#f6f8fb",
                     "savefig.facecolor": "#f6f8fb"})


def collect():
    groups = {}
    for manifest_path in sorted(SOURCE.rglob("manifest.json")):
        run = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        label_path = run / "quality_label.json"
        label = json.loads(label_path.read_text()) if label_path.exists() else None
        quality = label.get("quality_score") if label is not None else None
        category = ("unknown" if label is None else "null_failure" if quality is None
                    else "fast" if quality == 5 else "full" if quality == 4 else "incomplete")
        group = run.parent.relative_to(SOURCE).as_posix()
        groups.setdefault(group, []).append({
            "run_id": run.name, "source": run.relative_to(ROOT).as_posix(),
            "alpha": manifest["arm_filter_alpha"], "quality": quality,
            "label_file_exists": label is not None, "category": category,
            "status": manifest["status"], "controller_error": manifest["status"].startswith("error:"),
            "circle_rating": (label or {}).get("cube_in_the_circle", {}).get("rating"),
            "inside_percent": (label or {}).get("cube_in_the_circle", {}).get("estimated_inside_percent"),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "settings_sha256": manifest["settings_sha256"],
            "limits_sha256": manifest["controller_limits_sha256"],
            "act_state_channels": manifest["act_state_channels"],
            "action_offset_frames": manifest["action_offset_frames"],
            "seconds_requested": manifest["seconds_requested"],
            "cycles": manifest["cycles"], "clipped_cycles": manifest["clipped_cycles"],
        })
    return groups


def display_label(group, runs):
    light = "Midnight" if "midnight" in group else "Daylight"
    suffix = " · set 1" if group.endswith("0.5_1") else " · set 2" if group.endswith("0.5_2") else ""
    return f"{light}  α {runs[0]['alpha']:.1f}{suffix}"


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.png", dpi=190)
    fig.savefig(HERE / f"{stem}.svg")
    plt.close(fig)


def main():
    groups = collect()
    labels = [display_label(g, r) for g, r in groups.items()]
    rows = list(groups.values())
    summaries = []
    for group, runs in groups.items():
        counts = Counter(r["category"] for r in runs)
        summaries.append({"group": group, "attempts": len(runs),
                          "full_completion": counts["fast"] + counts["full"],
                          "fast_completion": counts["fast"],
                          "categories": dict(counts),
                          "controller_errors": sum(r["controller_error"] for r in runs)})
    (HERE / "outcome_snapshot.json").write_text(json.dumps({
        "source": str(SOURCE), "date": "2026-10-25", "rules": {
            "full_completion": "quality >= 4", "fast_completion": "quality == 5",
            "quality_null": "failure by operator convention", "missing_label_file": "unknown",
            "lighting": "inferred from operator folder names", "all_attempts_denominator": True},
        "summary": summaries, "runs": groups}, indent=2) + "\n")

    fig, (ax, errors) = plt.subplots(1, 2, figsize=(15, 7.6),
                                   gridspec_kw={"width_ratios": [3.9, 1.15]})
    fig.subplots_adjust(left=.21, right=.96, top=.77, bottom=.27, wspace=.13)
    fig.text(.04, .94, "Mickey17 / ACT rollout outcomes", fontsize=25, weight="bold")
    fig.text(.04, .885, f"{sum(map(len, rows))} attempts · zero-offset checkpoint · 40-second trials", fontsize=13, color="#52677b")
    fig.text(.04, .84, "Compare within lighting. The two daylight α 0.5 sets remain separate.", fontsize=11, color="#52677b")
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    names = [("fast", "Full ≤30 s"), ("full", "Full by 40 s, not fast"),
             ("incomplete", "Scored incomplete"), ("null_failure", "Null score → failure"),
             ("unknown", "Missing label file")]
    for key, title in names:
        values = np.array([sum(r["category"] == key for r in runs) for runs in rows])
        ax.barh(y, values, left=left, color=COLORS[key], height=.66, label=title)
        for i, v in enumerate(values):
            if v:
                ax.text(left[i]+v/2, i, str(v), ha="center", va="center", weight="bold",
                        color="white" if key in ("fast", "null_failure") else "#203247")
        left += values
    for i, summary in enumerate(summaries):
        ax.text(left[i]+.15, i, f"{summary['full_completion']}/{summary['attempts']} full", va="center", fontsize=10, weight="bold")
    ax.set_yticks(y, labels)
    ax.set_xticks(range(0, max(map(len, rows))+1, 2))
    ax.set_xlim(0, max(map(len, rows))+1.9)
    ax.invert_yaxis()
    ax.set_xlabel("Attempts (counts, including failures and unknowns)", labelpad=12)
    ax.set_title("Task outcome", loc="left", weight="bold", pad=17)
    ax.tick_params(axis="y", length=0)
    error_counts = [s["controller_errors"] for s in summaries]
    errors.barh(y, error_counts, height=.46, color="#735b84")
    for i, n in enumerate(error_counts):
        errors.text(n+.12, i, str(n), va="center", weight="bold")
    errors.set_ylim(ax.get_ylim())
    errors.set_yticks([])
    errors.set_xlim(0, max(5, max(error_counts)+1))
    errors.set_xticks([0, 2, 4])
    errors.set_xlabel("Attempts")
    errors.set_title("Controller stops", loc="left", weight="bold", pad=17)
    handles = [Patch(facecolor=COLORS[key], label=title) for key, title in names]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.035, .083), ncol=3, frameon=False, fontsize=10)
    fig.text(.04, .055, "Full = quality 4 or 5; fast = quality 5. Placement coverage is a separate rating.", fontsize=10, color="#52677b")
    fig.text(.04, .025, "Controller stops can occur after task completion, so stop counts may overlap full completions. Observational groups; no causal lighting claim.", fontsize=9, color="#52677b")
    save(fig, "outcome_overview")

    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    fig.subplots_adjust(left=.25, right=.94, top=.83, bottom=.16, hspace=.56)
    fig.text(.04, .95, "Every trial, visible", fontsize=25, weight="bold")
    fig.text(.04, .905, "Columns show chronological run order within each set; they are not matched scenes.", fontsize=12, color="#52677b")
    quality_colors = ["#d4d9e1", "#ce6471", "#b8c0cc", "#eac486", "#e6b365", "#cfc57c", "#69bac5", "#167d9a"]
    circle_colors = ["#d4d9e1", "#e5e9ef", "#ead5b4", "#dfbd8d", "#b9d0bb", "#90c4bc", "#56a8b5", "#167d9a"]
    for ax, mode, colors, title in zip(axes, ["quality", "circle"], [quality_colors, circle_colors],
        ["Task quality / higher is better", "Cube inside circle / estimated coverage"]):
        values = np.full((len(rows), max(map(len, rows))), -2)
        annotations = {}
        for i, runs in enumerate(rows):
            for j, r in enumerate(runs):
                if not r["label_file_exists"]:
                    value, annotation = -2, "?"
                elif mode == "quality":
                    value = r["quality"] if r["quality"] is not None else -1
                    annotation = str(r["quality"]) if r["quality"] is not None else "F"
                else:
                    rating = r["circle_rating"]
                    value = 5-rating if rating is not None else -1
                    annotation = f"{r['inside_percent']}%" if r["inside_percent"] is not None else "—"
                values[i,j] = value
                annotations[i,j] = annotation + ("*" if r["controller_error"] else "")
        ax.imshow(values, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-2.5, 6.5), 8), aspect="auto")
        for (i,j), annotation in annotations.items():
            ax.text(j, i, annotation, ha="center", va="center", fontsize=10,
                    color="white" if values[i,j] == 5 else "#203247", weight="bold")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(values.shape[1]), range(1, values.shape[1]+1))
        ax.set_xticks(np.arange(-.5, values.shape[1]), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
        ax.grid(which="minor", color="#f6f8fb", linewidth=4)
        ax.tick_params(which="both", length=0)
        ax.set_xlabel("Run order within set", labelpad=10)
        ax.set_title(title, loc="left", weight="bold", pad=14)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.04, .092, "* Controller error    F Null quality → failure    ? Missing label file    — Null placement rating", fontsize=11)
    fig.text(.04, .056, "Quality: 0=no phase; 1–4=phases completed; 5=all four within 30 s. Circle: operator estimate, not pixel measurement.", fontsize=10, color="#52677b")
    fig.text(.04, .026, "Null placement is shown separately from 0% coverage. A controller error and a completed task can coexist.", fontsize=10, color="#52677b")
    save(fig, "trial_details")
    for label, summary in zip(labels, summaries):
        print(f"{label}: full {summary['full_completion']}/{summary['attempts']}; fast {summary['fast_completion']}; errors {summary['controller_errors']}; {summary['categories']}")


if __name__ == "__main__":
    main()
