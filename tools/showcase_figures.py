"""Publication-style figures from the existing public numerical evidence.

All art is vector geometry, text, and measured traces. The runtime drawing is
an implementation schematic; the scheduling example is explicitly illustrative.
"""
from collections import Counter
import statistics

BLUE = "#A9D0E8"
BLUE_DARK = "#397BA5"
PEACH = "#F7C79E"
ROSE = "#E9A4AC"
ROSE_DARK = "#AB4B60"
GRAY = "#CFD3DB"
INK = "#243746"
MUTED = "#667581"
PALE = "#F6F8FA"


def outcome_flow_counts(snapshot):
    """Keep controller failures out of any inferred phase progression."""
    primary = [t for t in snapshot["trials"] if t["cohort"] == "primary"]
    normal = [t for t in primary if t["status"] == "time limit" and
              not t["label_missing"] and t["quality_rank"] is not None]
    stopped = [t for t in primary if t["status"].startswith("error:")]
    other = len(primary) - len(normal) - len(stopped)
    ranks = Counter(t["quality_rank"] for t in normal)
    return {"all": len(primary), "normal": len(normal), "controller_stops": len(stopped),
            "returned": ranks[5], "placed_only": ranks[4],
            "not_placed": sum(ranks[r] for r in range(4)), "other_unknown": other}


def render_figures(snapshot, summary, out, groups, joints, outcome, wilson):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle, FancyBboxPatch, FancyArrowPatch, PathPatch
    from matplotlib.path import Path

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#7D8992", "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "mickey17-paper-style-v2",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=.13)
        path = out / f"{name}.svg"
        fig.savefig(path, bbox_inches="tight", pad_inches=.13, metadata={"Date": None})
        if name == "outcome_flow":
            path.write_text(path.read_text().replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif"))
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)

    def title(ax, letter, text):
        ax.set_title(f"({letter})  {text}", loc="left", fontsize=13, fontweight="bold", pad=17)

    def clean(ax):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#D9DEE3", linestyle=":", linewidth=.65)
        ax.tick_params(axis="y", length=0, pad=8)

    def points(ax, s, y, key="placement", color=BLUE_DARK, offset=0):
        n, k = s["n"], s[key]
        lo, hi = (s[key + "_wilson95"] if key in ["placement", "return"] else wilson(k, n))
        rate = k / n * 100
        ax.errorbar(rate, y + offset, xerr=[[max(0, rate - 100*lo)], [max(0, 100*hi-rate)]],
                    fmt="o", color=color, capsize=3, elinewidth=1.35, markersize=5)

    # Compact three-panel filtering comparison, patterned after the user's paper figure.
    alpha = groups["alpha"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True,
                             gridspec_kw={"width_ratios": [1.55, 1, 1], "wspace": .20})
    for i, (folder, _) in enumerate(alpha):
        s = summary[folder]
        counts = [s["return"], s["placement"] - s["return"], s["n"] - s["placement"]]
        left = 0
        for k, color in zip(counts, [BLUE, PEACH, GRAY]):
            width = 100 * k / s["n"]
            axes[0].barh(i, width, left=left, height=.48, color=color, edgecolor="white", linewidth=.8)
            if k:
                axes[0].text(left + width/2, i, str(k), ha="center", va="center", fontsize=11)
            left += width
        for ax, key, color in [(axes[1], "placement", BLUE_DARK),
                                (axes[2], "controller_errors", ROSE_DARK)]:
            points(ax, s, i, key, color)
            rate = 100 * s[key] / s["n"]
            ax.text(min(89, max(10, rate)), i+.32, f"{s[key]}/{s['n']}", color=color,
                    ha="center", va="center", fontsize=10)
    for ax in axes:
        ax.set(xlim=(0, 100), ylim=(4.65, -.65))
        clean(ax)
        ax.set_xticks([0, 50, 100], ["0", "50", "100"])
    axes[0].set_yticks(range(5), [f"α = {label.replace('[baseline]', '')}" for _, label in alpha])
    axes[0].set_xlabel("Trials (%) · 10 per setting", labelpad=8)
    axes[1].set_xlabel("Placement (%)", labelpad=8)
    axes[2].set_xlabel("Controller stops (%)", labelpad=8)
    for ax, letter, name in zip(axes, "abc", ["Task outcome", "Placement", "Controller stops"]):
        title(ax, letter, name)
    fig.legend(handles=[Patch(color=BLUE, label="Placed + returned"), Patch(color=PEACH, label="Placed only"),
                        Patch(color=GRAY, label="Unsuccessful"),
                        Line2D([], [], color=INK, marker="o", label="Wilson 95% interval", markersize=4)],
               loc="lower center", bbox_to_anchor=(.51, -.12), ncol=4, frameon=False, fontsize=10)
    save(fig, "alpha")

    def scene(ax, family, letter):
        entries = groups[family]
        for i, (folder, _) in enumerate(entries):
            s = summary[folder]
            points(ax, s, i, "placement", BLUE_DARK, -.09)
            points(ax, s, i, "return", "#B47B4E", .09)
            ax.text(112, i, f"{s['placement']}/{s['n']}", ha="center", va="center", fontsize=10)
            ax.text(145, i, f"{s['controller_errors']}/{s['n']}", ha="center", va="center",
                    color=ROSE_DARK, fontsize=10)
        ax.set(xlim=(0, 162), ylim=(len(entries)-.5, -.5), xlabel="Observed success (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks(range(len(entries)), [label.replace("[baseline]", "") + ("*" if i == 0 else "")
                                          for i, (_, label) in enumerate(entries)])
        title(ax, letter, "Base position" if family == "base" else "Cube orientation")
        ax.text(112, 1.025, "Placed", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.text(145, 1.025, "Stops", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.xaxis.set_label_coords(.31, -.13)
        clean(ax)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8), gridspec_kw={"wspace": .32})
    scene(axes[0], "base", "a")
    scene(axes[1], "orientation", "b")
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE_DARK, label="Placement"),
                        Line2D([], [], marker="o", color="#B47B4E", label="Placement + return")],
               loc="lower center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False, fontsize=10)
    fig.text(.5, -.17, "α = 0.5 throughout · *the same ten baseline trials · whiskers: Wilson 95% intervals",
             ha="center", fontsize=10, color=MUTED)
    save(fig, "scene_sensitivity")
    for family in ["base", "orientation"]:
        fig, ax = plt.subplots(figsize=(7.1, 3.6))
        scene(ax, family, "a")
        fig.text(.5, -.055, "Blue: placement · peach: placement + return · *shared baseline", ha="center", fontsize=10)
        save(fig, family)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharey=True,
                             gridspec_kw={"width_ratios": [5, 3, 6], "wspace": .24})
    for ax, (family, entries), letter in zip(axes, groups.items(), "abc"):
        for i, (folder, label) in enumerate(entries):
            values = sorted(t["completion_time_s"] for t in snapshot["trials"] if t["cohort"] == "primary"
                            and folder in t["memberships"] and outcome(t, 4) is True
                            and t["completion_time_s"] is not None)
            jitter = np.linspace(-.16, .16, len(values)) if len(values) > 1 else [0] * len(values)
            ax.scatter(i + np.asarray(jitter), values, color=BLUE_DARK, s=24, edgecolors="white", linewidths=.4)
            if values:
                ax.plot([i-.27, i+.27], [statistics.median(values)] * 2, color="#B47B4E", linewidth=2)
            ax.text(i, 41, str(len(values)), ha="center", va="center", fontsize=10)
        ax.set_xticks(range(len(entries)), [label.replace("[baseline]", "").replace("2 cm ", "") for _, label in entries], rotation=25)
        ax.set(ylim=(0, 44))
        ax.grid(axis="y", color="#E0E4E8", linestyle=":", linewidth=.65)
        title(ax, letter, {"alpha": "Filter α", "base": "Base position", "orientation": "Cube rotation"}[family])
    axes[0].set_ylabel("Placement time (s)")
    fig.text(.5, -.10, "Dots: recorded placements · bars: medians · numbers: timed sample count", ha="center", fontsize=10)
    save(fig, "placement_timing")

    lookup = {t["trial_id"]: t for t in snapshot["trials"]}
    for trace in snapshot["trajectories"]:
        t = lookup[trace["trial_id"]]
        data = np.asarray(trace["rows"])
        fig, axes = plt.subplots(3, 2, figsize=(11.6, 6.8), sharex=True)
        for j, ax in enumerate(axes.flat):
            for stage, color, linewidth, linestyle in [
                ("proposed", "#B9BFC8", 1.25, "-"), ("filtered", "#D99C6A", 1.25, "--"),
                ("sent", BLUE_DARK, 1.4, "-"), ("observed", INK, 1.0, "-")]:
                column = trace["columns"].index(f"{stage}.arm_{joints[j]}.pos")
                ax.plot(data[:, 0], data[:, column], color=color, linewidth=linewidth, linestyle=linestyle, label=stage)
            if t["completion_time_s"] is not None:
                ax.axvline(t["completion_time_s"], color=ROSE_DARK, linestyle=":", linewidth=1)
            ax.set_title(f"({chr(97+j)})  {joints[j].replace('_', ' ')}", fontsize=11, loc="left", pad=7)
            ax.grid(color="#E1E5E9", linestyle=":", linewidth=.6)
            if j % 2 == 0:
                ax.set_ylabel("Position units", fontsize=10)
        for ax in axes[-1]:
            ax.set_xlabel("Elapsed time (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.015), frameon=False, fontsize=10)
        fig.suptitle(f"α = {t['alpha']:g} · " + ("median timed rank-5 trial" if t['alpha'] == .5 else "earliest controller stop"),
                     x=.08, ha="left", fontsize=13, y=.995, fontweight="bold")
        fig.tight_layout(rect=(0, .025, 1, .965), h_pad=.9, w_pad=1.7)
        save(fig, "trajectory_alpha_" + str(t["alpha"]).replace(".", "_"))

    # Outcome flow: four terminal classes, no guessed phase labels for errors.
    counts = outcome_flow_counts(snapshot)
    if counts["other_unknown"]:
        raise ValueError("Outcome-flow figure needs an explicit unknown branch for this input")
    assert counts["normal"] == counts["returned"] + counts["placed_only"] + counts["not_placed"]
    assert counts["all"] == counts["normal"] + counts["controller_stops"]
    from flow_figures import outcome_flow
    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    outcome_flow(ax, counts)
    ax.text(.012, 1.31, "Where the trials ended", fontsize=16, fontweight="bold")
    ax.text(.012, .015, "95 unique primary trials · percentages of all trials · pooled across conditions",
            fontsize=10, color=MUTED)
    save(fig, "outcome_flow")

    # Shared scientific-diagram vocabulary: pastel modules, dashed boundaries, sparse arrows.
    def box(ax, x, y, w, h, text, color=PALE, fontsize=12, dashed=False, edge=INK):
        patch = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.015,rounding_size=.13",
                               facecolor=color, edgecolor=edge, linewidth=1,
                               linestyle=(0,(5,4)) if dashed else "solid")
        ax.add_patch(patch)
        if text:
            ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fontsize)
        return patch
    def arrow(ax, start, end, **kwargs):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=11,
                                    linewidth=1.15,color=INK, **kwargs))

    fig = plt.figure(figsize=(13.6, 8.2))
    runtime = fig.add_axes([.035,.61,.93,.34])
    runtime.set(xlim=(0,16),ylim=(0,4.2));runtime.axis("off")
    runtime.text(0,4.0,"(a)  Runtime data flow",fontsize=14,fontweight="bold")
    box(runtime,2.8,.85,2.5,2.7,"",color="#F1F7FB",dashed=True,edge=BLUE_DARK)
    runtime.text(4.05,3.24,"Async worker",ha="center",fontsize=11,color=BLUE_DARK)
    box(runtime,5.75,.85,7.95,2.7,"",dashed=True)
    runtime.text(9.72,3.24,"Controller · requested 30 Hz",ha="center",fontsize=11)
    box(runtime,.05,1.53,2.0,1.05,"Wrist RGB\n+ robot state",BLUE,11)
    box(runtime,3.10,1.53,1.9,1.05,"ACT\n≈321 ms",BLUE,12)
    box(runtime,6.03,1.53,1.65,1.05,"Action queue\n30-step chunks",BLUE,10)
    box(runtime,8.08,1.53,1.15,1.05,"Raw\nchecks",GRAY,11)
    box(runtime,9.63,1.53,1.35,1.05,"α filter\n5 arm joints",PEACH,10)
    box(runtime,11.38,1.53,1.95,1.05,"Bounded target\n+ gripper",PEACH,10)
    box(runtime,14.2,1.53,1.7,1.05,"SO-101\nbase held still",BLUE,10)
    for x0,x1 in [(2.05,3.10),(5.,6.03),(7.68,8.08),(9.23,9.63),(10.98,11.38),(13.33,14.2)]:
        arrow(runtime,(x0,2.055),(x1,2.055))
    runtime.plot([15.05,15.05,1.05,1.05],[1.51,.33,.33,1.50],color=INK,lw=1)
    arrow(runtime,(1.05,.65),(1.05,1.5))
    runtime.text(8.0,.18,"Measured state + next wrist image",ha="center",va="top",fontsize=10,color=MUTED)

    schedule = fig.add_axes([.045,.10,.575,.42])
    schedule.set(xlim=(22.7,54),ylim=(-1.25,4.2));schedule.axis("off")
    schedule.text(22.7,3.95,"(b)  One arriving chunk",fontsize=13,fontweight="bold")
    schedule.text(24,3.32,"Observed at 24",fontsize=10,ha="left")
    schedule.text(34,3.32,"Ready at 34",fontsize=10,ha="left",color=BLUE_DARK)
    arrow(schedule,(24,3.05),(34,3.05))
    schedule.text(29,2.67,"Inference",fontsize=9,ha="center",color=MUTED)
    rows=[("Older queue",1.85),("New chunk",.95),("Execute",.05)]
    for label,y in rows:
        schedule.text(23.5,y+.27,label,ha="right",va="center",fontsize=10)
        for cycle in range(24,54):
            color="white"; hatch=None; alpha=1
            if label=="Older queue":
                if cycle<34:color=BLUE;alpha=.65
                elif cycle<42:color=BLUE
            elif label=="New chunk":
                if cycle<34:color=GRAY;hatch="///"
                else:color=PEACH
            else:
                if 34<=cycle<42:color="#C3C4DC"
                elif cycle>=42:color=PEACH
            schedule.add_patch(Rectangle((cycle,y),.84,.54,facecolor=color,edgecolor="#A0ABB5",
                                         lw=.4,hatch=hatch,alpha=alpha))
    schedule.plot([34,34],[-.10,2.53],color=BLUE_DARK,lw=1,ls="--")
    for cycle in [24,29,34,39,42,53]:
        schedule.text(cycle+.42,-.32,str(cycle),ha="center",fontsize=9)
    schedule.text(29,1.64,"Executed during inference",fontsize=8.5,ha="center",color=MUTED)
    schedule.text(29,.62,"10 expired new steps",fontsize=8.5,ha="center",color=MUTED)
    schedule.text(37.8,-.72,"Overlap: new weight ramps to 0.7",fontsize=9,ha="center")
    schedule.text(47.4,-1.04,"New-only targets: weight 1.0",fontsize=9,ha="center")

    filt = fig.add_axes([.68,.12,.285,.40])
    filt.set(xlim=(0,5.7),ylim=(0,5.3));filt.axis("off")
    filt.text(0,5.0,"(c)  Controller filter",fontsize=13,fontweight="bold")
    box(filt,.12,3.1,1.1,1.0,"Queued\nproposal",BLUE,10)
    box(filt,2.05,3.1,1.5,1.0,"α filter",PEACH,12)
    box(filt,4.35,3.1,1.1,1.0,"Target\nchecks",GRAY,10)
    arrow(filt,(1.22,3.6),(2.05,3.6));arrow(filt,(3.55,3.6),(4.35,3.6))
    filt.text(2.8,4.35,"First five arm channels",ha="center",fontsize=10)
    filt.plot([.67,.67,4.9,4.9],[3.08,2.2,2.2,3.08],color=INK,lw=1)
    arrow(filt,(4.9,2.5),(4.9,3.08))
    filt.text(2.8,1.85,"Gripper bypass",ha="center",fontsize=10)
    filt.text(2.8,1.05,r"$u_t = \alpha q_t + (1-\alpha)u_{t-1}$",ha="center",fontsize=13)
    filt.text(2.8,.43,"q: queued · u: filtered proposal",ha="center",fontsize=9,color=MUTED)
    fig.text(.045,.025,"Scheduling example is illustrative · zero-offset ACT · raw proposal checks precede filtering",
             fontsize=10,color=MUTED)
    save(fig,"system_runtime")

    fig,ax=plt.subplots(figsize=(13.2,2.9))
    ax.set(xlim=(-.2,15.7),ylim=(0,3.4));ax.axis("off")
    for x,w,label,color in [(0,2.35,"Collection",BLUE), (2.65,9.3,"Training · Mac / MPS",PEACH),
                             (12.25,3.2,"Deployment · NUC",BLUE)]:
        box(ax,x,.4,w,2.35,"",color="#F8F9FB",dashed=True)
        ax.text(x+w/2,2.45,label,ha="center",fontsize=11)
    nodes=[(.25,1.4,1.85,"Teleoperate\n158 demos",BLUE),
           (2.95,1.4,2.2,"Split\n134 train / 24 val",BLUE),
           (5.75,1.4,2.2,"Train ACT\n20k-step budget",PEACH),
           (8.55,1.4,2.95,"Select step 18k\nheld-out review",PEACH),
           (12.60,1.4,2.5,"CPU rollout\nqueue + filter",BLUE)]
    for x,y,w,text,color in nodes:box(ax,x,y,w,.8,text,color,11)
    for left,right in zip(nodes,nodes[1:]):arrow(ax,(left[0]+left[2],1.8),(right[0],1.8))
    ax.text(1.18,.82,"Wrist RGB\nstate + action",ha="center",fontsize=10,color=MUTED)
    ax.text(7.30,.85,"ResNet18 + ACT · 30-action chunks · zero label offset",ha="center",fontsize=10,color=MUTED)
    ax.text(13.85,.85,"Evaluation: 95 primary trials",ha="center",fontsize=9,color=MUTED)
    ax.text(0,3.1,"From demonstrations to deployment",fontsize=14,fontweight="bold")
    save(fig,"pipeline")
