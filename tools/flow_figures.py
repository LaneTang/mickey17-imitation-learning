"""Shared, count-preserving branching flows for the README's two Sankey plots."""
from __future__ import annotations


def draw_flow(ax, total, source_label, branches, source_bottom, scale):
    """Draw top-to-bottom branches; leaf bottoms are explicit layout choices.

    Each branch is {label, count, bottom, color, node, leaves}; leaves have
    those same fields except leaves. Width always equals count * scale.
    No ribbon is inflated to make a small count more visible.
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Rectangle
    if total <= 0 or sum(b['count'] for b in branches) != total:
        raise ValueError('Source and branch counts must conserve flow')
    for branch in branches:
        if sum(leaf['count'] for leaf in branch['leaves']) != branch['count']:
            raise ValueError('Branch and terminal counts must conserve flow')
    ink = '#50566F'
    width = .016
    xs = [.012, .49, .992]

    def ribbon(x0, y0, x1, y1, count, color):
        height = count * scale
        span = x1-x0
        c0,c1 = x0 + .48*span, x1 - .48*span
        vertices = [(x0,y0), (c0,y0), (c1,y1), (x1,y1),
                    (x1,y1+height), (c1,y1+height), (c0,y0+height),
                    (x0,y0+height), (x0,y0)]
        codes = [Path.MOVETO]+[Path.CURVE4]*3+[Path.LINETO]+[Path.CURVE4]*3+[Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(vertices,codes), facecolor=color, edgecolor='none'))

    def node(x,y,count,color):
        ax.add_patch(Rectangle((x,y),width,count*scale,fc=color,ec='none'))

    def label(x,y,text,count,align='right',percent=False):
        ax.text(x,y+.025,text,ha=align,va='bottom',fontsize=12.5,color=ink)
        suffix = f' · {count/total:.1%}' if percent else ''
        ax.text(x,y+.004,f'{count:,}{suffix}',ha=align,va='top',fontsize=12.5,
                color=ink,weight='bold')

    source_cursor=source_bottom+total*scale
    for branch in branches:
        source_cursor-=branch['count']*scale
        ribbon(xs[0]+width,source_cursor,xs[1],branch['bottom'],branch['count'],branch['color'])
        node(xs[1],branch['bottom'],branch['count'],branch['node'])
        label(xs[1]-.013,branch['bottom']+branch['count']*scale/2,branch['label'],branch['count'],percent=True)
        cursor=branch['bottom']+branch['count']*scale
        for leaf in branch['leaves']:
            cursor-=leaf['count']*scale
            ribbon(xs[1]+width,cursor,xs[2],leaf['bottom'],leaf['count'],leaf['color'])
            node(xs[2],leaf['bottom'],leaf['count'],leaf['node'])
            label(xs[2]-.013,leaf['bottom']+leaf['count']*scale/2,leaf['label'],leaf['count'],percent=True)
    node(xs[0],source_bottom,total,'#229EB1')
    label(xs[0]+.03,source_bottom+total*scale/2,source_label,total,align='left')
    ax.set(xlim=(0,1.025),ylim=(0,1.35))
    ax.axis('off')


def dataset_flow(ax, counts):
    scene=sum(counts['scene',p] for p in ['train','validation'])
    motion=sum(counts['motion',p] for p in ['train','validation'])
    branches=[dict(label='Scene variation',count=scene,bottom=.69,color='#B9DDE4',node='#229EB1',leaves=[
        dict(label='Training',count=counts['scene','train'],bottom=.84,color='#C1DDB7',node='#459C2B'),
        dict(label='Validation',count=counts['scene','validation'],bottom=.61,color='#F0B2BA',node='#C54D69')]),
        dict(label='Motion variation',count=motion,bottom=.14,color='#F7D0B6',node='#DE9A71',leaves=[
        dict(label='Training',count=counts['motion','train'],bottom=.31,color='#C1DDB7',node='#459C2B'),
        dict(label='Validation',count=counts['motion','validation'],bottom=.09,color='#F0B2BA',node='#C54D69')])]
    draw_flow(ax,scene+motion,'All demos',branches,.32,.004)


def outcome_flow(ax, counts):
    branches=[dict(label='Time limit',count=counts['normal'],bottom=.66,color='#B9DDE4',node='#229EB1',leaves=[
        dict(label='Placed + returned',count=counts['returned'],bottom=1.01,color='#C1DDB7',node='#459C2B'),
        dict(label='Placed, not returned',count=counts['placed_only'],bottom=.83,color='#F0B2BA',node='#C54D69'),
        dict(label='Not placed',count=counts['not_placed'],bottom=.43,color='#FBCBB4',node='#F07639')]),
        dict(label='Controller stop',count=counts['controller_stops'],bottom=.14,color='#D8D9E1',node='#9EA1B1',leaves=[
        dict(label='Phase not ranked',count=counts['controller_stops'],bottom=.10,color='#D8D9E1',node='#9EA1B1')])]
    draw_flow(ax,counts['all'],'All trials',branches,.32,.0055)
