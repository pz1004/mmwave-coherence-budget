#!/usr/bin/env python3
"""Graff et al.'s comparison, reproduced under Graff et al.'s accounting, then
re-run with the compute term added.

The rule followed: "Reproduce their comparison under their own accounting first,
confirm you match, then add the compute term and show what moves.  Overturning a
result requires first reproducing it."

Exactly one thing differs between the two runs: whether inference is free.  Same
channels, same receivers, same noise draws, same probe charge, same hardware.
So every difference below is the compute term and only the compute term.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'budget'))
import theory

PUBS = ['sweep64', 'hier6', 'topk8', 'topk16', 'topk32', 'fc16', 'fc32', 'cnn16',
        'cnn32', 'zhang-AHTBS+AHIBS']
JOINT = json.load(open(os.path.join(HERE, 'results.json')))
FREE = json.load(open(os.path.join(HERE, 'results_freecompute.json')))
GRID = JOINT['grid_us']


def curve(site, nm):
    return np.array([site['methods'][nm]['per_T'][str(T)]['mean'] for T in GRID])


def pairs(site):
    """Ordered pairs (a, b) whose sign flips between the ends of the sweep."""
    out = set()
    for i, a in enumerate(PUBS):
        for b in PUBS[i + 1:]:
            d = curve(site, a) - curve(site, b)
            if d[0] * d[-1] < 0:
                out.add((a, b))
    return out


def pairs_anywhere(site):
    """Pairs (a, b) whose rate difference changes sign ANYWHERE on the grid.

    pairs() compares only the grid's two ends, and at 0.25 ms
    seven of the ten methods deliver exactly zero under BOTH accountings, so
    no pair among them can ever register -- and those same-K pairs are exactly
    where charging inference acts (under (1) topk16/fc16/cnn16 all cost K t_p
    and are ordered by eta alone; under (2) a charged method with the higher
    eta must cross its cheaper counterpart).  This definition can see them."""
    out = set()
    for i, a in enumerate(PUBS):
        for b in PUBS[i + 1:]:
            d = curve(site, a) - curve(site, b)
            if (d > 1e-12).any() and (d < -1e-12).any():
                out.add((a, b))
    return out


def reversals(sf, sj):
    """Per (pair, deadline): cells whose strict order differs between the two
    accountings.  Ties (both methods at zero rate) are not reversals."""
    cells = []
    for i, a in enumerate(PUBS):
        for b in PUBS[i + 1:]:
            df, dj = curve(sf, a) - curve(sf, b), curve(sj, a) - curve(sj, b)
            for t, T in enumerate(GRID):
                if df[t] * dj[t] < 0:
                    cells.append({"pair": [a, b], "T_us": T})
    return cells


def tstar(site, m):
    sw = theory.Strategy('sweep64', 0, site['methods']['sweep64']['cost_us'],
                         site['methods']['sweep64']['eta'])
    L = theory.Strategy(m, 0, site['methods'][m]['cost_us'], site['methods'][m]['eta'])
    bp = theory.breakpoint(L, sw)
    return float(bp) if bp else None


out = {"accounting_joint": JOINT.get('accounting', 'joint'),
       "accounting_free": FREE['accounting'], "grid_us": GRID, "sites": []}
for sj, sf in zip(JOINT['sites'], FREE['sites']):
    assert sj['site'] == sf['site']
    # --- 1. does the published conclusion reproduce under its own accounting? --
    sweep_zero = [T for T, v in zip(GRID, curve(sf, 'sweep64')) if v == 0]
    best_free = [max(PUBS, key=lambda m: curve(sf, m)[i]) for i in range(len(GRID))]
    best_joint = [max(PUBS, key=lambda m: curve(sj, m)[i]) for i in range(len(GRID))]
    # --- 2. which inversions are Graff's, and which need the compute term? -----
    pf, pj = pairs(sf), pairs(sj)
    af, aj = pairs_anywhere(sf), pairs_anywhere(sj)
    rev = reversals(sf, sj)
    npairs = len(PUBS) * (len(PUBS) - 1) // 2
    # --- 3. how far does the compute term move the crossover? -----------------
    ts = []
    for m in ('topk8', 'topk16', 'topk32', 'fc16', 'fc32', 'cnn16', 'cnn32', 'hier6'):
        a, b = tstar(sf, m), tstar(sj, m)
        ts.append({"method": m, "tau_us": round(sj['methods'][m]['cost_us'] -
                                                sf['methods'][m]['cost_us'], 2),
                   "T_star_free_us": round(a, 1) if a else None,
                   "T_star_joint_us": round(b, 1) if b else None,
                   "shift_pct": (round((b / a - 1) * 100, 2) if a and b else None)})
    # --- 4. rate gap at each deadline: what does charging compute cost? -------
    gap = []
    for i, T in enumerate(GRID):
        rows = [(m, curve(sf, m)[i], curve(sj, m)[i]) for m in PUBS]
        worst = max(rows, key=lambda r: r[1] - r[2])
        gap.append({"T_us": T, "worst_method": worst[0],
                    "free": round(worst[1], 4), "joint": round(worst[2], 4),
                    "loss_pct": round((1 - worst[2] / worst[1]) * 100, 2)
                    if worst[1] > 0 else None,
                    "winner_free": best_free[i], "winner_joint": best_joint[i]})
    # --- 5. FEASIBILITY: methods their accounting calls usable at a deadline
    #        that cannot finish at all once their own inference is charged. ----
    flips = []
    for m in PUBS:
        cf, cj = curve(sf, m), curve(sj, m)
        for i, T in enumerate(GRID):
            if cf[i] > 0 and cj[i] == 0:
                flips.append({"method": m, "T_ms": T / 1000,
                              "free_rate": round(float(cf[i]), 4),
                              "cost_free_us": round(sf['methods'][m]['cost_us'], 1),
                              "cost_joint_us": round(sj['methods'][m]['cost_us'], 1)})
    out['sites'].append({
        "feasibility_flips": flips,
        "n_feasibility_flips": len(flips),
        "site": sj['site'],
        "sweep_infeasible_below_ms": (max(sweep_zero) / 1000 if sweep_zero else None),
        "winner_free": best_free, "winner_joint": best_joint,
        "winner_changes": [{"T_ms": T / 1000, "free": a, "joint": b}
                           for T, a, b in zip(GRID, best_free, best_joint) if a != b],
        "inversions_under_their_accounting": sorted(map(list, pf)),
        "inversions_under_joint": sorted(map(list, pj)),
        "inversions_needing_compute": sorted(map(list, pj - pf)),
        "inversions_removed_by_compute": sorted(map(list, pf - pj)),
        # the four keys above use pairs(): a sign flip between the grid's two
        # ENDS only, so they are blind to every pair tied at zero at 0.25 ms.
        "inversion_definition_endpoint": "sign flip between T = 0.25 ms and T = 10 ms",
        "inversions_anywhere_their": sorted(map(list, af)),
        "inversions_anywhere_joint": sorted(map(list, aj)),
        "inversions_anywhere_created_by_compute": sorted(map(list, aj - af)),
        "inversions_anywhere_removed_by_compute": sorted(map(list, af - aj)),
        "reversal_cells": rev,
        "n_reversal_cells": len(rev), "n_pair_deadline_cells": npairs * len(GRID),
        "reversals_by_deadline": {str(T): sum(1 for c in rev if c['T_us'] == T)
                                  for T in GRID},
        "zero_at_tightest_under_both": [m for m in PUBS
                                        if curve(sf, m)[0] == 0 and curve(sj, m)[0] == 0],
        "t_star": ts, "gap": gap})

json.dump(out, open(os.path.join(HERE, 'accounting.json'), 'w'), indent=1)

for s in out['sites']:
    print("=" * 92)
    print(f"{s['site']}   sweep delivers zero at or below "
          f"{s['sweep_infeasible_below_ms']} ms under their accounting")
    print(f"  inversions under THEIR accounting : {len(s['inversions_under_their_accounting'])}")
    print(f"  inversions under JOINT accounting : {len(s['inversions_under_joint'])}")
    print(f"  ... that require the compute term : {len(s['inversions_needing_compute'])}"
          f"  {s['inversions_needing_compute']}")
    print(f"  ... that the compute term removes : {len(s['inversions_removed_by_compute'])}")
    print(f"  deadlines where the WINNER changes: {len(s['winner_changes'])}"
          f"  {[(c['T_ms'], c['free'], '->', c['joint']) for c in s['winner_changes']]}")
    print(f"  {'method':>10} {'tau us':>8} {'T* their':>10} {'T* joint':>10} {'shift':>8}")
    for t in s['t_star']:
        print(f"  {t['method']:>10} {t['tau_us']:8.1f} "
              f"{(t['T_star_free_us']/1000 if t['T_star_free_us'] else 0):10.2f} "
              f"{(t['T_star_joint_us']/1000 if t['T_star_joint_us'] else 0):10.2f} "
              f"{(str(t['shift_pct'])+'%' if t['shift_pct'] is not None else '-'):>8}")
    w = max(s['gap'], key=lambda g: g['loss_pct'] or 0)
    print(f"  largest rate loss from charging compute: {w['loss_pct']}% "
          f"({w['worst_method']} at {w['T_us']/1000:g} ms)")
    print(f"  FEASIBILITY FLIPS (positive rate under their accounting, exactly "
          f"zero under ours): {s['n_feasibility_flips']}")
    for f in s['feasibility_flips']:
        print(f"     {f['method']:>18} at {f['T_ms']:>5}ms : "
              f"{f['free_rate']:.3f} -> 0   ({f['cost_free_us']} -> "
              f"{f['cost_joint_us']} us against a {f['T_ms']*1000:.0f} us deadline)")
