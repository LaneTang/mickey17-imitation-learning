#!/usr/bin/env python3
"""README illustrations and dataset traces, reproducible from compact evidence.

Default rendering uses only the public snapshot. --dataset-root additionally
exports a read-only 30 -> 10 Hz derivation from the local merged dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/2026-11-02-dataset'
OUT = ROOT / 'analysis/2026-11-02-readme'
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
BLUE, PEACH, ROSE, GRAY = '#A9D0E8', '#F7C79E', '#E9A4AC', '#CFD3DB'
INK, MUTED = '#243746', '#667581'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_dataset(root, destination, split_path):
    import pyarrow.parquet as pq
    info = json.loads((root / 'meta/info.json').read_text())
    sources = json.loads((root / 'source_episode_map.json').read_text())
    split = json.loads(split_path.read_text())
    expected_order = ['arm_' + j + '.pos' for j in JOINTS]
    if info['features']['action']['names'][:6] != expected_order or info['fps'] != 30:
        raise ValueError('Unexpected command layout or recording rate')
    train, val = set(split['train_episodes']), set(split['validation_episodes'])
    if train & val or train | val != set(range(158)) or len(train) != 134 or len(val) != 24:
        raise ValueError('Unexpected 158-episode training split')
    paths = sorted(root.glob('data/**/*.parquet'))
    episodes = {}
    for path in paths:
        columns = pq.read_table(path, columns=['episode_index', 'timestamp', 'action']).to_pydict()
        for episode, timestamp, action in zip(columns['episode_index'], columns['timestamp'], columns['action']):
            episodes.setdefault(episode, []).append((float(timestamp), action[:6]))
    if set(episodes) != train | val or len(sources) != 158:
        raise ValueError('Episode/source-map mismatch')
    trace = io.StringIO(newline='')
    writer = csv.writer(trace, lineterminator='\n')
    writer.writerow(['episode', 'time_s'] + JOINTS)
    records = []
    counts = Counter()
    for episode, rows in sorted(episodes.items()):
        source = sources[str(episode)]
        if len(rows) != source['frames'] or any(b[0] <= a[0] for a,b in zip(rows, rows[1:])):
            raise ValueError('Invalid episode frames/timestamps')
        group = 'motion' if '4th_patch_extra' in source['source_dir'] else 'scene'
        partition = 'train' if episode in train else 'validation'
        indices = sorted(set(range(0, len(rows), 3)) | {len(rows)-1})
        for index in indices:
            t, action = rows[index]
            writer.writerow([episode, f'{t:.6f}'] + [f'{x:.6f}' for x in action])
        records.append(dict(episode=episode, group=group, split=partition,
                            source_frames=len(rows), sampled_frames=len(indices),
                            first_time_s=rows[0][0], last_time_s=rows[-1][0]))
        counts[group] += 1
    if counts != {'scene':110, 'motion':48} or sum(r['source_frames'] for r in records) != info['total_frames']:
        raise ValueError('Unexpected dataset group/frame counts')
    destination.mkdir(parents=True, exist_ok=True)
    packed = gzip.compress(trace.getvalue().encode(), mtime=0)
    (destination / 'commands_10hz.csv.gz').write_bytes(packed)
    manifest = dict(schema_version=1, episodes=records, joints=JOINTS,
                    units='Recorded command-position units, not assumed physical degrees',
                    source_fps=30, plot_fps=10, total_source_frames=info['total_frames'],
                    derivation='Keep every third sample and each episode final sample; timestamps/commands rounded to six decimals. No interpolation or smoothing.',
                    collection_intent='Scene/motion descriptions are supplied by the author; membership follows source folders.',
                    source_sha256={'meta/info.json':sha(root/'meta/info.json'),
                                   'source_episode_map.json':sha(root/'source_episode_map.json'),
                                   'split.json':sha(split_path),
                                   **{p.relative_to(root).as_posix():sha(p) for p in paths}},
                    commands_sha256=hashlib.sha256(packed).hexdigest())
    (destination/'dataset.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_snapshot(path):
    manifest = json.loads((path/'dataset.json').read_text())
    if sha(path/'commands_10hz.csv.gz') != manifest['commands_sha256']:
        raise ValueError('Derived trace hash mismatch')
    traces = {}
    with gzip.open(path/'commands_10hz.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            traces.setdefault(int(row['episode']), []).append([float(row['time_s'])] + [float(row[j]) for j in JOINTS])
    records = manifest['episodes']
    if len(records) != 158 or set(traces) != {r['episode'] for r in records}:
        raise ValueError('Incomplete public trace snapshot')
    for r in records:
        rows = traces[r['episode']]
        if len(rows) != r['sampled_frames'] or abs(rows[-1][0]-r['last_time_s']) > 1e-6:
            raise ValueError('Final frame missing')
    return manifest, traces


def render(manifest, traces, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, PathPatch, Patch
    from matplotlib.path import Path as MPath
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'text.color':INK,
                         'axes.labelcolor':INK, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.linewidth':.7, 'svg.fonttype':'none', 'svg.hashsalt':'mickey17-readme-20261102'})
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out/(name+'.png'), dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.15)
        fig.savefig(out/(name+'.svg'), facecolor='white', bbox_inches='tight', pad_inches=.15, metadata={'Date':None})
        svg = out/(name+'.svg')
        text = svg.read_text().replace("font-family: 'DejaVu Sans'", "font-family: 'DejaVu Sans', Arial, sans-serif")
        svg.write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
        plt.close(fig)

    def canvas(w, h, xmax, ymax):
        fig, ax = plt.subplots(figsize=(w,h))
        ax.set(xlim=(0,xmax), ylim=(0,ymax)); ax.axis('off')
        return fig, ax

    def box(ax, x, y, w, h, label='', color='#F7F9FB', dashed=False, size=11):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.015,rounding_size=.10',
                                   facecolor=color, edgecolor=INK, lw=.85,
                                   linestyle=(0,(5,4)) if dashed else '-'))
        if label: ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)

    def arrow(ax, start, end, **kw):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>', mutation_scale=11, color=INK,lw=1, **kw))

    # Architecture verified against selected config and LeRobot ACT forward().
    fig,ax=canvas(13.0,6.8,16,8.6)
    ax.text(.05,8.15,'Inside the real-time system',fontsize=17,weight='bold')
    box(ax,2.1,4.1,13.65,3.55,dashed=True,color='#F5F9FC')
    ax.text(2.35,7.27,'Asynchronous inference · NUC CPU · median 321 ms',fontsize=11,color=MUTED)
    box(ax,.05,5.8,1.65,.85,'Wrist image\n640 × 480',BLUE,size=10)
    box(ax,.05,4.45,1.65,.85,'Robot state\n9 channels',BLUE,size=10)
    box(ax,2.45,5.8,1.85,.85,'RGB tensor\nnormalization',BLUE,size=10)
    box(ax,4.8,5.8,2,.85,'ResNet18\nimage features',BLUE,size=11)
    box(ax,7.3,5.8,1.9,.85,'1 × 1 projection\n+ position tokens',BLUE,size=9)
    box(ax,3.1,4.45,2.5,.85,'Normalize + project\nstate token',BLUE,size=10)
    box(ax,6.3,4.45,2.7,.85,'Latent token\nz = 0 at inference',GRAY,size=10)
    box(ax,9.85,4.7,2.2,1.75,'ACT transformer\n2 encoder layers\n1 decoder layer',PEACH,size=10)
    box(ax,12.65,4.7,2.65,1.75,'Action head\n+ denormalization\n30 steps × 9 channels',PEACH,size=10)
    for a,b in [((1.7,6.22),(2.45,6.22)),((4.3,6.22),(4.8,6.22)),((6.8,6.22),(7.3,6.22)),
                ((9.2,6.22),(9.85,6.0)),((1.7,4.87),(3.1,4.87)),((5.6,4.87),(5.85,4.87)),
                ((9,4.87),(9.85,5.2)),((12.05,5.57),(12.65,5.57))]:arrow(ax,a,b)
    ax.plot([5.85,5.85,9.45,9.45],[4.87,4.32,4.32,5.5],color=INK,lw=.9)
    arrow(ax,(9.45,5.5),(9.85,5.5))
    ax.text(10.95,4.35,'30 learned action queries',ha='center',fontsize=9,color=MUTED)
    box(ax,.05,.8,15.7,2.9,dashed=True)
    ax.text(.3,3.3,'Controller · requested 30 Hz · keeps running during inference',fontsize=11,color=MUTED)
    nodes=[(.4,2.6,'Timestamped queue\nblend overlaps',BLUE),(3.45,2,'Raw proposal\nchecks',GRAY),
           (6,2,'α filter\n5 arm channels',PEACH),(8.55,2.3,'Bounded target\nchecks',GRAY),
           (11.5,3.55,'SO-101 arm + gripper\nbase velocities = 0',BLUE)]
    for x,w,label,color in nodes:box(ax,x,1.75,w,.9,label,color,size=10)
    for left,right in zip(nodes,nodes[1:]):arrow(ax,(left[0]+left[1],2.2),(right[0],2.2))
    ax.plot([14,15.5,15.5,1.7,1.7],[4.7,4.7,3.97,3.97,2.67],color=INK,lw=1)
    arrow(ax,(1.7,2.9),(1.7,2.67))
    ax.text(7.85,3.8,'Ready chunk: six arm/gripper channels retained',ha='center',fontsize=9,color=MUTED)
    ax.plot([5.72,5.72,9.7,9.7],[2.2,1.2,1.2,1.72],color=INK,lw=.9)
    arrow(ax,(9.7,1.4),(9.7,1.73))
    ax.text(7.7,.96,'Gripper bypasses α filter',ha='center',fontsize=10)
    ax.text(.3,.22,'Next wrist image + measured state feed the next inference; base predictions are not sent.',fontsize=10,color=MUTED)
    save(fig,'runtime_architecture')

    fig=plt.figure(figsize=(12.8,5.2))
    schedule=fig.add_axes([.10,.26,.53,.57]);schedule.set(xlim=(23.5,54),ylim=(-.6,3.9));schedule.axis('off')
    fig.text(.05,.94,'How chunks become commands',fontsize=17,weight='bold')
    fig.text(.05,.86,'(a) One arriving chunk',fontsize=12,weight='bold')
    arrow(schedule,(24,3.1),(34,3.1))
    schedule.text(24,3.45,'Observe: 24',fontsize=10)
    schedule.text(34,3.45,'Ready: 34',fontsize=10)
    schedule.text(29,2.7,'Inference',ha='center',fontsize=10,color=MUTED)
    for label,y in [('Older queue',1.8),('New chunk',.95),('Output',.1)]:
        schedule.text(23.4,y+.25,label,ha='right',va='center',fontsize=10)
        for cycle in range(24,54):
            color='white';hatch=None
            if label=='Older queue' and cycle<42:color=BLUE
            elif label=='New chunk':
                color=GRAY if cycle<34 else PEACH
                if cycle<34:hatch='///'
            elif label=='Output':
                color=BLUE if cycle<34 else '#C3C4DC' if cycle<42 else PEACH
            schedule.add_patch(Rectangle((cycle,y),.86,.5,fc=color,ec='#929FA9',lw=.4,hatch=hatch))
    schedule.plot([34,34],[-.05,2.5],ls='--',lw=.8,color=INK)
    for cycle in [24,29,34,39,42,53]:schedule.text(cycle+.4,-.22,str(cycle),ha='center',fontsize=9)
    fig.text(.10,.17,'24–33: old actions execute; 10 new predictions expire.\n34–41: overlap blends. 42–53: new-only predictions.',fontsize=10,linespacing=1.6)
    filt=fig.add_axes([.70,.22,.28,.60]);filt.set(xlim=(0,5.5),ylim=(0,4.7));filt.axis('off')
    filt.text(0,4.4,'(b) Separate arm filter',fontsize=12,weight='bold')
    box(filt,0,2.7,1.2,.8,'Queue',BLUE,size=10);box(filt,2,2.7,1.25,.8,'α filter',PEACH,size=11)
    box(filt,4,2.7,1.3,.8,'Target\nchecks',GRAY,size=10)
    arrow(filt,(1.2,3.1),(2,3.1));arrow(filt,(3.25,3.1),(4,3.1))
    filt.text(2.65,3.7,'First five arm channels',ha='center',fontsize=10)
    filt.plot([.6,.6,4.65,4.65],[2.7,2,2,2.68],color=INK,lw=.9)
    arrow(filt,(4.65,2.3),(4.65,2.7));filt.text(2.65,1.6,'Gripper bypass',ha='center',fontsize=10)
    filt.text(2.65,.8,r'$u_t=\alpha q_t+(1-\alpha)u_{t-1}$',ha='center',fontsize=12)
    filt.text(2.65,.25,'q: queued · u: filtered',ha='center',fontsize=9,color=MUTED)
    fig.text(.05,.04,'Illustrative zero-offset example · overlap weight ramps to 0.7 over six cycles; new-only weight is 1.0.',fontsize=10,color=MUTED)
    save(fig,'scheduling_filter')

    # Count-conserving flow, all connections derived from episode membership.
    records=manifest['episodes'];counts=Counter((r['group'],r['split']) for r in records)
    from flow_figures import dataset_flow
    fig,ax=plt.subplots(figsize=(12.8,7.4))
    dataset_flow(ax,counts)
    ax.text(.012,1.31,'What went into the dataset',fontsize=16,weight='bold')
    ax.text(.012,.015,'134 training · 24 validation · percentages of all 158 demos',fontsize=10,color=MUTED)
    save(fig,'dataset_flow')

    fig,axes=plt.subplots(3,2,figsize=(12,7.4),sharex=True,layout='constrained')
    titles=['Shoulder pan','Shoulder lift','Elbow flex','Wrist flex','Wrist roll','Gripper']
    for j,ax in enumerate(axes.flat):
        for r in records:
            data=np.asarray(traces[r['episode']])
            ax.plot(data[:,0],data[:,j+1],color=BLUE if r['group']=='scene' else PEACH,alpha=.3,lw=.6,rasterized=False)
        ax.set_title(f'({chr(97+j)})  {titles[j]}',loc='left',fontsize=12)
        ax.grid(color='#E2E6EA',ls=':',lw=.6);ax.tick_params(labelsize=9)
        if j%2==0:ax.set_ylabel('Command units',fontsize=10)
        if j>=4:ax.set_xlabel('Time in demonstration (s)',fontsize=10)
    fig.suptitle('The motions I demonstrated',fontsize=17,weight='bold')
    fig.legend(handles=[Patch(color=BLUE,label='Scene variation · 110 demos'),Patch(color=PEACH,label='Motion variation · 48 demos')],
               loc='outside lower center',ncol=2,frameon=False,fontsize=11)
    save(fig,'dataset_joint_commands')

    fig,ax=canvas(12.8,4.3,16,5.4)
    ax.text(.05,5.05,'Practice pipeline',fontsize=17,weight='bold')
    steps=[(.3,3,3,'1  Assemble\n& calibrate'),(4.35,3,3,'2  Define task\n& success'),(8.4,3,3,'3  Set scene\n& teleoperate'),
           (12.45,3,3,'4  Collect\n& review demos'),(12.45,.9,3,'5  Train\n& review policy'),(8.4,.9,3,'6  NUC\npreflight'),(2.4,.9,5,'7  Supervised trials\n& review')]
    for i,(x,y,w,label) in enumerate(steps):box(ax,x,y,w,1,label,BLUE if i in [0,2,3,5] else PEACH,size=11)
    for a,b in zip(steps,steps[1:]):
        if a[1]==b[1]:arrow(ax,(a[0]+a[2] if b[0]>a[0] else a[0],a[1]+.5),(b[0] if b[0]>a[0] else b[0]+b[2],b[1]+.5))
        else:arrow(ax,(a[0]+a[2]/2,a[1]),(b[0]+b[2]/2,b[1]+1))
    ax.plot([4.9,4.9,13.95],[1.9,2.5,2.5],color=MUTED,lw=.9)
    arrow(ax,(9.9,2.5),(9.9,3))
    arrow(ax,(13.95,2.5),(13.95,3))
    arrow(ax,(13.95,2.5),(13.95,1.9))
    ax.text(5.2,2.55,'Review → adjust scene, demos or policy',fontsize=9,color=MUTED)
    ax.text(.3,.15,'My hands-on workflow, from setting up the robot to finding out what needs work.',fontsize=10,color=MUTED)
    save(fig,'practice_pipeline')

    fig,ax=canvas(12.8,4.0,16,5)
    ax.text(.05,4.65,'Policy deployment',fontsize=17,weight='bold')
    box(ax,.1,2.5,5.1,1.5,dashed=True);ax.text(.35,3.75,'Mac',fontsize=10,color=MUTED)
    box(ax,5.6,.65,10.1,3.35,dashed=True);ax.text(5.85,3.75,'NUC + robot',fontsize=10,color=MUTED)
    nodes=[(.4,2.75,4.5,'Held-out review\nchoose checkpoint',BLUE),(6,2.75,4.2,'Preflight\nmodel · camera · pose · timing',BLUE),
           (11,2.75,4.2,'Shadow test\nleader moves; policy predicts',BLUE),
           (11,1.0,4.2,'Short supervised rollout',PEACH),(6,1.0,4.2,'Full task trial + review',PEACH)]
    for x,y,w,label,col in nodes:box(ax,x,y,w,.75,label,col,size=10)
    arrow(ax,(4.9,3.12),(6,3.12));ax.text(5.45,3.5,'Transfer',ha='center',fontsize=9)
    arrow(ax,(10.2,3.12),(11,3.12));arrow(ax,(13.1,2.75),(13.1,1.75));arrow(ax,(11,1.37),(10.2,1.37))
    ax.text(.25,1.2,'Review each stage\nbefore moving on',fontsize=11,color=MUTED)
    ax.text(.25,.1,'Development checks for the final ACT workflow · full trials use the overlapping queue and arm filter.',fontsize=10,color=MUTED)
    save(fig,'policy_deployment')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path)
    parser.add_argument('--snapshot-dir',type=Path,default=EVIDENCE)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    if args.dataset_root:
        source = args.dataset_root.resolve()
        for destination in [args.snapshot_dir.resolve(), args.output_dir.resolve()]:
            if destination == source or source in destination.parents:
                parser.error('Snapshot and figures must be outside the source dataset tree')
        export_dataset(args.dataset_root,args.snapshot_dir,ROOT/'evidence/2026-11-01/split.json')
    manifest,traces=load_snapshot(args.snapshot_dir)
    render(manifest,traces,args.output_dir)
    print('Rendered six PNG/SVG pairs from',len(traces),'demonstrations.')


if __name__=='__main__':main()
