#!/usr/bin/env python3
"""Why the published comparator loses rate as the deadline grows.

A controller that is given more time should not deliver less. The paper's
reimplementation of Zhang et al.'s AHTBS+AHIBS does so on most evaluation
sites between 0.5 and 0.7 ms. This script records the evidence for a
mechanism, without testing it by ablation, and writes comparator.json:

  * rate and mean hierarchy layer against T, for AHIBS alone and for
    AHTBS+AHIBS. The layer comes from evaluate.py's diagnostics, and T enters
    only AHIBS's layer rule (methods.zhang);
  * each drop in rate as T grows, split into the four prespecified
    oracle-SNR strata, weighted by stratum size, so the stratum that carries
    the loss can be read directly;
  * the rate of fixed-depth hierarchical search (hier3..hier6) in the top
    stratum at the loosest deadline. AHIBS descends whenever it predicts one
    more layer doubles the SNR, a single-path assumption. Deeper fixed search
    losing rate there means that prediction is wrong on the site.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ZH = ('zhang-AHIBS', 'zhang-AHTBS+AHIBS')


def site_record(s, grid):
    M, frac = s['methods'], s['strata_frac']
    rec = {"site": s['site']}
    for nm in ZH:
        rate = [M[nm]['per_T'][str(T)]['mean'] for T in grid]
        layer = [s['diagnostics'][f'{nm}@{T}'][1] for T in grid]
        slots = [s['diagnostics'][f'{nm}@{T}'][0] for T in grid]
        drops = []
        for i in range(len(grid) - 1):
            if rate[i + 1] < rate[i]:
                a = M[nm]['per_T'][str(grid[i])]['strata']
                b = M[nm]['per_T'][str(grid[i + 1])]['strata']
                contrib = [None if (x is None or y is None) else (y - x) * f
                           for x, y, f in zip(a, b, frac)]
                drops.append({"from_us": grid[i], "to_us": grid[i + 1],
                              "rel": (rate[i + 1] / rate[i] - 1) if rate[i] else None,
                              "layer_from": layer[i], "layer_to": layer[i + 1],
                              "stratum_contribution": contrib})
        rec[nm] = {"rate": rate, "mean_layer": layer, "mean_slots": slots, "drops": drops}
    top = {f'hier{L}': M[f'hier{L}']['per_T'][str(grid[-1])]['strata'][3]
           for L in (3, 4, 5, 6) if f'hier{L}' in M}
    rec["fixed_depth_top_stratum_at_loosest"] = top
    return rec


def main():
    out = {"sites": []}
    for f in ('results.json', 'heldout.json'):
        R = json.load(open(os.path.join(HERE, f)))
        for s in R['sites']:
            out['sites'].append(site_record(s, R['grid_us']))
    json.dump(out, open(os.path.join(HERE, 'comparator.json'), 'w'), indent=1)
    for r in out['sites']:
        z = r['zhang-AHTBS+AHIBS']
        ds = [f"{d['from_us']/1000:g}->{d['to_us']/1000:g} ms {100*d['rel']:+.1f}% "
              f"(layer {d['layer_from']:.2f}->{d['layer_to']:.2f}; top stratum "
              f"{d['stratum_contribution'][3]:+.3f})" for d in z['drops']]
        print(f"{r['site']:24s} drops: {'; '.join(ds) or 'none'}")
        print(f"{'':24s} fixed depth, top stratum at 10 ms: "
              + ", ".join(f"{k} {v:.2f}" for k, v in r['fixed_depth_top_stratum_at_loosest'].items()))


if __name__ == '__main__':
    main()
