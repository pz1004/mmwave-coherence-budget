#!/usr/bin/env python3
"""How much does the controller-variant choice depend on WHICH site validates it?

The manuscript's limitation section says one validation site carries that choice
and that the two leading candidates differ by a few per cent on test in opposite
directions.  That is a statement about variance, and variance can be measured:
rotate the validating site over every site that is not the fit site, and for each
choice record what it would have cost on all the others.

Nothing is retrained.  The fit site, the predictor and the DP tables are fixed;
only which site is consulted to pick the variant changes.  So this isolates the
selection step, which is the thing the limitation is about.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# --suffix _rr reads and writes the re-solved runs' files; '' is the original
SUF = sys.argv[sys.argv.index('--suffix') + 1] if '--suffix' in sys.argv else ''
FAMILIES = {"with_compute": ("sched", "sched-pi", "sched-noreorder", "sched-noreorder-pi"),
            "without_compute": ("sched-nocompute", "sched-nocompute-pi")}

sites = {}
for f in ('validation', 'results', 'heldout'):
    j = json.load(open(os.path.join(HERE, f'{f}{SUF}.json')))
    grid = j['grid_us']
    for s in j['sites']:
        sites[s['site']] = s
names = sorted(sites)
print(f"{len(names)} sites available (the fit site is not among them)\n")


def mean_goodput(site, nm):
    return float(np.mean([sites[site]['methods'][nm]['per_T'][str(T)]['mean']
                          for T in grid]))


rec = {"grid_us": grid, "sites": names, "families": {}}
for fam, cands in FAMILIES.items():
    perf = {s: {c: mean_goodput(s, c) for c in cands} for s in names}
    best_on = {s: max(cands, key=lambda c: perf[s][c]) for s in names}
    rows = []
    for v in names:                       # v validates
        pick = best_on[v]
        others = [s for s in names if s != v]
        # regret: how much worse the pick is than that site's own best
        reg = [(perf[s][best_on[s]] - perf[s][pick]) / perf[s][best_on[s]]
               for s in others]
        rows.append({"validator": v, "picked": pick,
                     "agrees_with_site_best": sum(1 for s in others
                                                  if best_on[s] == pick),
                     "n_others": len(others),
                     "mean_regret": round(float(np.mean(reg)), 5),
                     "max_regret": round(float(np.max(reg)), 5)})
    votes = {}
    for v in names:
        votes[best_on[v]] = votes.get(best_on[v], 0) + 1
    rec['families'][fam] = {"candidates": list(cands), "rows": rows, "votes": votes,
                            "per_site_best": best_on,
                            "mean_regret_over_validators":
                                round(float(np.mean([r['mean_regret'] for r in rows])), 5),
                            "worst_regret_any_pair":
                                round(float(max(r['max_regret'] for r in rows)), 5)}
    print(f"--- {fam} ---")
    print(f"  which variant each site would pick: {votes}")
    print(f"  {'validator':>16} {'picks':>22} {'agrees':>7} {'mean regret':>12} {'worst':>8}")
    for r in rows:
        print(f"  {r['validator'].split('_',2)[2][:-3]:>16} {r['picked']:>22} "
              f"{r['agrees_with_site_best']}/{r['n_others']:<5} "
              f"{r['mean_regret']*100:11.2f}% {r['max_regret']*100:7.2f}%")
    print(f"  mean regret over all choices of validator: "
          f"{rec['families'][fam]['mean_regret_over_validators']*100:.2f}%"
          f"   worst single pairing: {rec['families'][fam]['worst_regret_any_pair']*100:.2f}%\n")

# does the CONCLUSION (compute pays) survive every choice of validator?
flips = []
for v in names:
    w = max(FAMILIES['with_compute'], key=lambda c: mean_goodput(v, c))
    o = max(FAMILIES['without_compute'], key=lambda c: mean_goodput(v, c))
    gains = [(s, mean_goodput(s, w) / mean_goodput(s, o) - 1)
             for s in names if s != v]
    flips.append({"validator": v, "with": w, "without": o,
                  "min_gain": round(min(g for _, g in gains), 5),
                  "max_gain": round(max(g for _, g in gains), 5),
                  "n_negative": sum(1 for _, g in gains if g <= 0)})
rec['conclusion_robustness'] = flips
print("does 'the compute action pays' survive every choice of validating site?")
print(f"  {'validator':>16} {'with':>22} {'without':>24} {'gain range':>20} {'sites <= 0':>11}")
for f in flips:
    print(f"  {f['validator'].split('_',2)[2][:-3]:>16} {f['with']:>22} {f['without']:>24} "
          f"{f['min_gain']*100:+8.1f}% to {f['max_gain']*100:+6.1f}% {f['n_negative']:>10}")
print(f"\n  worst case over all validators: min gain "
      f"{min(f['min_gain'] for f in flips)*100:+.1f}%, and "
      f"{sum(f['n_negative'] for f in flips)} of {len(flips)*(len(names)-1)} "
      f"site-validator pairs are non-positive")
json.dump(rec, open(os.path.join(HERE, f'selection_stability{SUF}.json'), 'w'), indent=1)
