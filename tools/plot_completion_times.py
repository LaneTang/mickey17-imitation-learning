#!/usr/bin/env python3
"""Plot automatic return-time candidates separately from placement markers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ALPHAS = [('0.1', 'alpha/alpha-0.1'), ('0.2', 'alpha/alpha-0.2'),
          ('0.5', 'alpha/alpha-0.5'), ('0.8', 'alpha/alpha-0.8'),
          ('1.0', 'alpha/alpha-1[baseline]')]


def alpha_results(report):
    if report.get('metric') != 'observed_motion_place_and_return_candidate':
        raise ValueError('Expected automatic completion candidates, not placement times')
    results = []
    ids = [t['trial_id'] for t in report['trials']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate trial IDs')
    for alpha, group in ALPHAS:
        trials = [t for t in report['trials'] if group in t['memberships']]
        times = sorted(t['candidate_completion_time_s'] for t in trials
                       if t['normal_ending_candidate'])
        if any(t['rollout_status'] != 'time limit' or t['candidate_completion_time_s'] is None
               for t in trials if t['normal_ending_candidate']):
            raise ValueError('Invalid normal-ending candidate')
        results.append(dict(alpha=alpha, group=group, n=len(trials), timed_n=len(times),
                            median_s=statistics.median(times) if times else None,
                            min_s=min(times) if times else None, max_s=max(times) if times else None,
                            abnormal_n=sum(t['rollout_status'] != 'time limit' for t in trials),
                            times_s=times))
    return results


def render(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'mickey17-completion',
                         'axes.linewidth': 0.7, 'text.color': '#243746',
                         'axes.labelcolor': '#243746', 'xtick.color': '#52616D',
                         'ytick.color': '#243746'})
    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    fig.subplots_adjust(left=.12, right=.77, top=.79, bottom=.30)
    for y, row in enumerate(results):
        values = row['times_s']
        offsets = [((i % 3)-1)*.11 for i in range(len(values))]
        ax.scatter(values, [y+v for v in offsets], s=38, color='#A9D0E8',
                   edgecolors='white', linewidths=.6, zorder=3)
        if values:
            ax.plot([row['median_s']]*2, [y-.23, y+.23], color='#AB4B60', lw=2.3, zorder=4)
        median = f"{row['median_s']:.2f} s" if values else '—'
        ax.text(1.03, y, median, transform=ax.get_yaxis_transform(), va='center', fontsize=11)
        ax.text(1.22, y, f"{row['timed_n']}/{row['n']}", transform=ax.get_yaxis_transform(), va='center', fontsize=11)
    ax.text(1.03, 1.07, 'Median', transform=ax.transAxes, fontsize=10, color='#667581')
    ax.text(1.20, 1.07, 'Timed / total', transform=ax.transAxes, fontsize=9, color='#667581')
    ax.set_yticks(range(len(results)), [r'$\alpha$ = '+r['alpha'] for r in results])
    ax.set_ylim(len(results)-.5, -.5)
    ax.set_xlim(0, 40)
    ax.set_xticks(range(0, 41, 10))
    ax.set_xlabel('Estimated completion time (s)', labelpad=10)
    ax.grid(axis='x', color='#DDE1E5', ls=':', lw=.7)
    ax.set_axisbelow(True)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    fig.text(.12, .94, 'How long did a rollout take?', fontsize=16, weight='bold')
    fig.text(.12, .88, 'Release and return, estimated from joint logs', fontsize=10.5, color='#667581')
    handles = [Line2D([], [], marker='o', color='none', markerfacecolor='#A9D0E8',
                      markeredgecolor='white', markersize=7, label='Individual rollout'),
               Line2D([], [], marker='|', color='none', markeredgecolor='#AB4B60',
                      markeredgewidth=2, markersize=12, label='Median')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.11,.08),
               frameon=False, ncol=2, fontsize=10)
    for suffix in ['png', 'svg']:
        fig.savefig(out / ('alpha_completion.'+suffix), dpi=180, facecolor='white', metadata={'Date':None} if suffix=='svg' else None)
    svg = out / 'alpha_completion.svg'
    text = svg.read_text().replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif")
    svg.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='completion_candidates.json or alpha_completion_data.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    results = alpha_results(report)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = [t for t in report['trials'] if any(g in t['memberships'] for _,g in ALPHAS)]
    snapshot = {key:report[key] for key in ['schema_version','metric','definition','warning','settings']}
    snapshot.update(trials=selected, source_report_sha256=report.get('source_report_sha256',hashlib.sha256(args.input.read_bytes()).hexdigest()))
    (out/'alpha_completion_data.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
    fields = ['alpha','n','timed_n','median_s','min_s','max_s','abnormal_n']
    with (out/'alpha_completion_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)
    lines=['| α | Timed / all | Median (s) | Range (s) | Abnormal endings |',
           '| --- | --- | --- | --- | --- |']
    for r in results:
        median=f"{r['median_s']:.2f}" if r['median_s'] is not None else '—'
        span=f"{r['min_s']:.2f}–{r['max_s']:.2f}" if r['min_s'] is not None else '—'
        lines.append(f"| {r['alpha']} | {r['timed_n']}/{r['n']} | {median} | {span} | {r['abnormal_n']} |")
    (out/'alpha_completion_table.md').write_text('\n'.join(lines)+'\n')
    render(results,out)
    print('\n'.join(lines))


if __name__=='__main__':
    main()
