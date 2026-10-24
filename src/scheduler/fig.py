"""Hand-built SVG figures.  Colours are the shared tokens, so both themes work."""
from __future__ import annotations
import math


def _x(v, lo, hi, x0, x1, logx):
    if logx:
        v, lo, hi = math.log10(v), math.log10(lo), math.log10(hi)
    return x0 + (v - lo) / (hi - lo) * (x1 - x0)


def _y(v, lo, hi, y0, y1):
    return y1 - (v - lo) / (hi - lo) * (y1 - y0)


def linechart(xs, series, *, ylab, xlab, w=930, h=400, ylim=None, logx=True,
              xticks=None, yticks=None, title=None, notes=(), legend_cols=4):
    ml, mr, mt, mb = 62, 16, (30 if title else 14), 78
    x0, x1, y0, y1 = ml, w - mr, mt, h - mb
    lo, hi = min(xs), max(xs)
    ymax = ylim[1] if ylim else max(max(s['ys']) for s in series) * 1.06
    ymin = ylim[0] if ylim else 0.0
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
         f'aria-label="{ylab} against {xlab}" style="font-family:var(--f-m);font-size:11px">']
    if title:
        p.append(f'<text x="{ml}" y="16" fill="var(--ink)" '
                 f'style="font-size:12px;letter-spacing:.08em">{title}</text>')
    yt = yticks or [ymin + (ymax - ymin) * i / 4 for i in range(5)]
    for v in yt:
        yy = _y(v, ymin, ymax, y0, y1)
        p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
        p.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" '
                 f'fill="var(--muted)">{v:g}</text>')
    for v in (xticks or xs):
        xx = _x(v, lo, hi, x0, x1, logx)
        p.append(f'<line x1="{xx:.1f}" y1="{y1}" x2="{xx:.1f}" y2="{y1+4}" '
                 f'stroke="var(--rule2)"/>')
        lab = f'{v/1000:g}' if v >= 1000 else f'{v/1000:g}'
        p.append(f'<text x="{xx:.1f}" y="{y1+17}" text-anchor="middle" '
                 f'fill="var(--muted)">{lab}</text>')
    p.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="var(--rule2)"/>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1+34}" text-anchor="middle" '
             f'fill="var(--muted)">{xlab}</text>')
    p.append(f'<text transform="translate(14,{(y0+y1)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" fill="var(--muted)">{ylab}</text>')
    for s in series:
        pts = " ".join(f"{_x(x,lo,hi,x0,x1,logx):.1f},{_y(v,ymin,ymax,y0,y1):.1f}"
                       for x, v in zip(xs, s['ys']))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
                 f'stroke-width="{s.get("w",2)}" stroke-linejoin="round" '
                 f'stroke-linecap="round"'
                 + (f' stroke-dasharray="{s["dash"]}"' if s.get('dash') else '') + '/>')
        if s.get('dots'):
            for x, v in zip(xs, s['ys']):
                p.append(f'<circle cx="{_x(x,lo,hi,x0,x1,logx):.1f}" '
                         f'cy="{_y(v,ymin,ymax,y0,y1):.1f}" r="2.6" fill="{s["color"]}"/>')
    for n in notes:
        xx = _x(n['x'], lo, hi, x0, x1, logx)
        p.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" '
                 f'stroke="var(--alarm)" stroke-width="1" stroke-dasharray="3 3"/>')
        p.append(f'<text x="{xx+5:.1f}" y="{y0+12}" fill="var(--alarm)">{n["label"]}</text>')
    # legend
    cw = (x1 - x0) / legend_cols
    for i, s in enumerate(series):
        cx = x0 + (i % legend_cols) * cw
        cy = h - mb + 46 + (i // legend_cols) * 15
        p.append(f'<line x1="{cx:.0f}" y1="{cy-4}" x2="{cx+16:.0f}" y2="{cy-4}" '
                 f'stroke="{s["color"]}" stroke-width="{s.get("w",2)}"'
                 + (f' stroke-dasharray="{s["dash"]}"' if s.get('dash') else '') + '/>')
        p.append(f'<text x="{cx+21:.0f}" y="{cy}" fill="var(--muted)">{s["name"]}</text>')
    p.append('</svg>')
    return "".join(p)


def barpanel(xs, groups, *, ylab, xlab, w=930, h=300, colors=None, title=None,
             ymax=None, fmt='{:g}'):
    """Grouped bars: groups = [(label, [values per x])]."""
    ml, mr, mt, mb = 62, 16, (28 if title else 12), 66
    x0, x1, y0, y1 = ml, w - mr, mt, h - mb
    ymax = ymax or max(max(v) for _, v in groups) * 1.1
    n, g = len(xs), len(groups)
    slot = (x1 - x0) / n
    bw = slot * 0.78 / g
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{ylab}" '
         f'style="font-family:var(--f-m);font-size:11px">']
    if title:
        p.append(f'<text x="{ml}" y="15" fill="var(--ink)" '
                 f'style="font-size:12px;letter-spacing:.08em">{title}</text>')
    for i in range(5):
        v = ymax * i / 4
        yy = _y(v, 0, ymax, y0, y1)
        p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="var(--rule)"/>')
        p.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" '
                 f'fill="var(--muted)">{fmt.format(v)}</text>')
    for j, (lab, vals) in enumerate(groups):
        c = (colors or ['var(--payload)'])[j % len(colors or [1])]
        for i, v in enumerate(vals):
            bx = x0 + i * slot + slot * 0.11 + j * bw
            by = _y(v, 0, ymax, y0, y1)
            p.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw-1.5:.1f}" '
                     f'height="{max(y1-by,0):.1f}" fill="{c}" opacity="0.9"/>')
        cx = x0 + (j % 4) * (x1 - x0) / 4
        cy = h - mb + 44 + (j // 4) * 14
        p.append(f'<rect x="{cx:.0f}" y="{cy-9}" width="11" height="9" fill="{c}"/>')
        p.append(f'<text x="{cx+16:.0f}" y="{cy}" fill="var(--muted)">{lab}</text>')
    for i, v in enumerate(xs):
        p.append(f'<text x="{x0+i*slot+slot/2:.1f}" y="{y1+16}" text-anchor="middle" '
                 f'fill="var(--muted)">{v/1000:g}</text>')
    p.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="var(--rule2)"/>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1+32}" text-anchor="middle" '
             f'fill="var(--muted)">{xlab}</text>')
    p.append(f'<text transform="translate(14,{(y0+y1)/2:.0f}) rotate(-90)" '
             f'text-anchor="middle" fill="var(--muted)">{ylab}</text>')
    p.append('</svg>')
    return "".join(p)
