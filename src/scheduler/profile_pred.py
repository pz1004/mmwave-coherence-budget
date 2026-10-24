#!/usr/bin/env python3
"""Accuracy and efficiency of every predictor, on the VALIDATION city only.

This is the (cost, eta) menu the Phase 2 envelope consumes.  It is measured
where model selection is allowed to look; the test cities are not touched.
"""
import json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
import obs, feats, sim, methods
import models as M

VAL = 'city_1_losangeles_28'
import protocol
SEEDS = protocol.seed_arg()   # the ten seeds in force; --seeds overrides it
KS = (2, 4, 8, 16, 32)
L6 = obs.LAYER6_OFFSET

lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
ch = {r['component']: r['charge_us'] for r in lat['rows']}
# One-shot strategies: k probes, then a SINGLE invocation to exit e.  That is
# one fresh pass, tau_fresh[e] -- not tau_restart[e], which is what a controller
# pays to climb to exit e through e separate invocations.
TAU = [0.0] + [lat['tau_fresh_us'][str(i)] for i in (1, 2, 3)]
TP = obs.T_P_US + ch['probe_update']
site = sim.Site(VAL)
orc = float(site.oracle_se.mean())
rows = []
for seed in SEEDS:
    ee = sim.load_predictor(seed)
    gen = torch.Generator(device=sim.DEV).manual_seed(seed)
    y = sim.measure(site.pbar, gen)
    for k in KS:
        d, c = methods.topk(site, y, k, TP)
        se = site.se.gather(1, c[:, None])[:, 0]
        rows.append(dict(name=f'topk{k}', k=k, tau=0.0, seed=seed,
                         acc=float((c - L6 == site.oracle_beam).double().mean()),
                         se=float(se.mean()), cost=float(d[0])))
        for e in (1, 2, 3):
            d, c = methods.fixed_net(site, y, k, f'ee{e}', ee, TAU[e], TP)
            se = site.se.gather(1, c[:, None])[:, 0]
            rows.append(dict(name=f'ee{e}k{k}', k=k, tau=TAU[e], seed=seed,
                             acc=float((c - L6 == site.oracle_beam).double().mean()),
                             se=float(se.mean()), cost=float(d[0])))
    for k in (4, 8, 16, 32, 64):
        for kind, tau in (('fc', ch[f'fc_k{k}']), ('cnn', ch['cnn'])):
            m = (M.BeamMLP(n_in=k) if kind == 'fc' else M.BeamCNN()).to(sim.DEV)
            m.load_state_dict(torch.load(os.path.join(HERE, 'models', f'{kind}_k{k}_{seed}.pt')))
            d, c = methods.fixed_net(site, y, k, kind, m.eval(), tau, TP)
            se = site.se.gather(1, c[:, None])[:, 0]
            rows.append(dict(name=f'{kind}{k}', k=k, tau=tau, seed=seed,
                             acc=float((c - L6 == site.oracle_beam).double().mean()),
                             se=float(se.mean()), cost=float(d[0])))
agg = {}
for r in rows:
    a = agg.setdefault(r['name'], dict(k=r['k'], tau=r['tau'], cost=r['cost'],
                                       acc=[], se=[]))
    a['acc'].append(r['acc']); a['se'].append(r['se'])
out = {"city": VAL, "n": site.n, "oracle_se": round(orc, 4), "methods": {}}
for n, a in agg.items():
    out['methods'][n] = {"k": a['k'], "tau_us": round(a['tau'], 3),
                         "cost_us": round(a['cost'], 3),
                         "top1_acc": round(float(np.mean(a['acc'])), 4),
                         "se": round(float(np.mean(a['se'])), 4),
                         "eta": round(float(np.mean(a['se'])) / orc, 4),
                         "seed_spread_se": round(float(np.ptp(a['se'])), 4)}
json.dump(out, open(os.path.join(HERE, 'predictor_profile.json'), 'w'), indent=1)
print(f"{VAL}  n={site.n}  oracle SE {orc:.4f}")
print(f"{'method':>9} {'K':>3} {'tau us':>8} {'cost us':>8} {'top-1':>7} {'SE':>7} {'eta':>7}")
for n in sorted(out['methods'], key=lambda x: (out['methods'][x]['k'], x)):
    m = out['methods'][n]
    print(f"{n:>9} {m['k']:3d} {m['tau_us']:8.2f} {m['cost_us']:8.1f} "
          f"{m['top1_acc']:7.4f} {m['se']:7.4f} {m['eta']:7.4f}")
