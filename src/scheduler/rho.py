#!/usr/bin/env python3
"""T* as a function of the hardware ratio, and the latency budget a deployment
must meet.  The answer to "inference latency is hardware-specific".

Accuracy is a property of the method, cost is a property of the method AND the
machine.  So eta is measured once per city and held fixed while the inference
charge is scaled by rho; the four measured profiles are marked at their own rho.

Two budgets are reported.  The ACCURACY-FREE one is Phase 2's tau* = (M-K) t_p:
the probe time the method saves, and an upper bound on what it may spend however
good it is.  The REALISED one uses the method's measured eta and is therefore
tighter: tau_max(T) = T - K t_p - (eta_S/eta_L)(T - M t_p)_+.

The positive part is load-bearing.  Below the sweep's completion time M t_p the
sweep delivers nothing, so the binding constraint is feasibility alone,
tau < T - K t_p; without the clip the negative (T - M t_p) is scaled by
eta_S/eta_L > 1 and ADDED, which would report a realised budget larger than the
accuracy-free cap tau* it is supposed to sharpen.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'budget'))
import theory

PROFILES = ('cpu1t_pcore', 'cpu1t_ecore', 'cpu8t_pcore', 'gpu')
LAT = {p: json.load(open(os.path.join(HERE, f'lat_{p}.json'))) for p in PROFILES}
RES = json.load(open(os.path.join(HERE, 'results.json')))
FREE = json.load(open(os.path.join(HERE, 'results_freecompute.json')))
GRID = RES['grid_us']
M_BEAMS = 64
CH = {p: {r['component']: r['charge_us'] for r in LAT[p]['rows']} for p in PROFILES}
TP0 = 20.0 + CH['cpu1t_pcore']['probe_update']

# method -> (probes K, inference charge on the primary profile, latency key)
METHODS = {'fc16': (16, CH['cpu1t_pcore']['fc_k16'], 'fc_k16'),
           'fc32': (32, CH['cpu1t_pcore']['fc_k32'], 'fc_k32'),
           'cnn16': (16, CH['cpu1t_pcore']['cnn'], 'cnn'),
           'cnn32': (32, CH['cpu1t_pcore']['cnn'], 'cnn')}
# published architectures with no trainable observation here: cost only, from the
# Phase 1 atlas.  They enter through the accuracy-free budget, which needs no eta.
ATLAS = json.load(open(os.path.join(HERE, '..', 'latency', 'p_cpu1t_pcore.json')))
HEAVY = {r['component']: r['charge_us'] for r in ATLAS['rows']
         if r['component'] in ('transformer_small', 'resnet18_vision', 'vit_b16_vision',
                               'lstm_cached', 'graff_eig')}

# R4: the same architectures on EVERY measured profile.  Beside the
# single-core CPU charges above, the atlas holds the GPU-resident and 8-thread
# ones -- and on the GPU ResNet-18 is feasible at half the deadlines.  An
# exclusion is stated per profile.
ATLAS_ALL = json.load(open(os.path.join(HERE, '..', 'latency', 'atlas.json')))['charge_us']
# The GPU-resident charges of the two vision models were re-timed at the FULL
# G3 counts (calibrate.py --full).  The atlas holds them at the reduced
# counts Phase 1 used for slow components; the full-count value replaces them.
_FULL = os.path.join(HERE, '..', 'latency', 'p_gpu_heavy_full.json')
HEAVY_FULL = ({r['component']: r['charge_us'] for r in json.load(open(_FULL))['rows']}
              if os.path.exists(_FULL) else {})
for _c, _v in HEAVY_FULL.items():
    ATLAS_ALL[f'{_c}|gpu-resident'] = _v
ATLAS_PROFILES = {'cpu-1t-pcore': 'cpu1t_pcore', 'cpu-1t-ecore': 'cpu1t_ecore',
                  'cpu-8t-pcore': 'cpu8t_pcore', 'gpu-resident': 'gpu'}
K_VISION = 4
heavy_by_profile = []
for comp in ('resnet18_vision', 'vit_b16_vision', 'transformer_small'):
    for ap, lp in ATLAS_PROFILES.items():
        tau = ATLAS_ALL.get(f'{comp}|{ap}')
        if tau is None:
            continue
        tp = 20.0 + CH[lp]['probe_update']
        feas = [T for T in GRID if T - K_VISION * tp > tau]
        tstar = (M_BEAMS - K_VISION) * tp
        heavy_by_profile.append({
            "component": comp, "profile": ap, "tau_us": tau, "t_p_us": round(tp, 3),
            "counts": ("full G3 (re-timed)" if (ap == 'gpu-resident' and comp in HEAVY_FULL)
                       else "as in the Phase 1 atlas"),
            "K": K_VISION, "feasible_deadlines_us": feas,
            "n_feasible": len(feas), "n_deadlines": len(GRID),
            "zero_rate_at_every_deadline": not feas,
            "tau_star_us": round(tstar, 1),
            "cannot_beat_sweep_anywhere": tau >= tstar,
            "margin_over_tau_star": round(tau / tstar, 3)})

out = {"t_p_us": TP0, "profiles": {}, "sites": [], "heavy_us": HEAVY,
       "heavy_by_profile": heavy_by_profile}
for p in PROFILES:
    out['profiles'][p] = {
        "rho_vs_pcore": round(LAT[p]['tau_restart_us']['3'] /
                              LAT['cpu1t_pcore']['tau_restart_us']['3'], 3),
        "tau3_us": LAT[p]['tau_restart_us']['3'],
        "t_probe_us": round(20.0 + CH[p]['probe_update'], 3),
        "contended": LAT[p]['contention']['contended']}

RHO = np.concatenate([np.linspace(0.25, 4, 46), np.linspace(4.5, 20, 32)])
for sj, sf in zip(RES['sites'], FREE['sites']):
    eta_s = sf['methods']['sweep64']['eta']
    d_s = M_BEAMS * TP0
    rows, curves, budget = [], {}, []
    for m, (K, tau0, key) in METHODS.items():
        eta_l = sf['methods'][m]['eta']          # eta is hardware-independent
        ys = []
        for r in RHO:
            L = theory.Strategy(m, 0, K * TP0 + r * tau0, eta_l)
            S = theory.Strategy('sweep', 0, d_s, eta_s)
            w = theory.window(L, S)          # None once c_L >= c_S: no window
            ys.append(round(float(w[1]), 1) if w and np.isfinite(w[1]) else None)
        curves[m] = ys
        prof = {}
        for p in PROFILES:
            tp = 20.0 + CH[p]['probe_update']
            L = theory.Strategy(m, 0, K * tp + CH[p][key], eta_l)
            S = theory.Strategy('sweep', 0, M_BEAMS * tp, eta_s)
            w = theory.window(L, S)
            prof[p] = round(float(w[1]), 1) if w and np.isfinite(w[1]) else None
        rows.append({"method": m, "K": K, "eta": round(eta_l, 4),
                     "tau_pcore_us": round(tau0, 2), "T_star_by_profile": prof,
                     # beyond this hardware ratio the method costs more than the
                     # sweep and the operating window is empty at every deadline
                     "rho_crit": round((M_BEAMS - K) * TP0 / tau0, 3)})
        # realised budget: the most inference this method may spend at each deadline
        for T in GRID:
            tmax = T - K * TP0 - (eta_s / eta_l) * max(0.0, T - d_s)
            fbp = {}
            for p in PROFILES:
                tp = 20.0 + CH[p]['probe_update']
                tm = T - K * tp - (eta_s / eta_l) * max(0.0, T - M_BEAMS * tp)
                fbp[p] = bool(tm > CH[p][key])
            budget.append({"method": m, "T_us": T,
                           "tau_max_us": round(float(tmax), 1),
                           "tau_star_accuracy_free_us": round((M_BEAMS - K) * TP0, 1),
                           "measured_tau_us": round(tau0, 2),
                           "fits": bool(tmax > tau0), "fits_by_profile": fbp})
    out['sites'].append({"site": sj['site'], "eta_sweep": round(eta_s, 4),
                         "rho_grid": [round(float(r), 3) for r in RHO],
                         "T_star_vs_rho": curves, "by_profile": rows,
                         "budget": budget})

json.dump(out, open(os.path.join(HERE, 'rho.json'), 'w'), indent=1)

print("hardware ratios, measured (rho = tau_3 relative to the pinned P-core)")
for p, v in out['profiles'].items():
    print(f"  {p:14s} rho={v['rho_vs_pcore']:6.2f}  tau_3={v['tau3_us']:8.2f} us"
          f"  t_p={v['t_probe_us']:.2f} us"
          + ("   [guard: contended]" if v['contended'] else ""))
for s in out['sites']:
    print(f"\n{s['site']}  (eta_sweep {s['eta_sweep']})")
    print(f"  {'method':>7} {'K':>3} {'eta':>7} {'tau us':>8} " +
          " ".join(f"{p.replace('cpu','').replace('_',''):>10}" for p in PROFILES))
    for r in s['by_profile']:
        print(f"  {r['method']:>7} {r['K']:3d} {r['eta']:7.4f} {r['tau_pcore_us']:8.2f} " +
              " ".join(f"{(r['T_star_by_profile'][p]/1000 if r['T_star_by_profile'][p] else 0):10.2f}"
                       for p in PROFILES) + "   (T* in ms)")
    b = [x for x in s['budget'] if x['method'] == 'cnn16']
    print(f"  latency budget for a 16-probe method, {s['site'].split('_')[2]}:")
    print(f"    {'T_c ms':>7} {'tau_max us':>11} {'accuracy-free tau* us':>22} {'cnn fits?':>10}")
    for x in b[:6]:
        print(f"    {x['T_us']/1000:7.2f} {x['tau_max_us']:11.1f} "
              f"{x['tau_star_accuracy_free_us']:22.1f} {str(x['fits']):>10}")
print("\npublished architectures with no trainable observation here (Phase 1 charges):")
for k, v in sorted(HEAVY.items(), key=lambda kv: kv[1]):
    excl = [T for T in GRID if v > (M_BEAMS - 4) * TP0 or v > T]
    print(f"  {k:20s} {v:10.1f} us   excluded at {len(excl)}/{len(GRID)} deadlines "
          f"by the accuracy-free budget alone")
print("\nper profile (R4), as a 4-probe method:")
for r in heavy_by_profile:
    print(f"  {r['component']:18s} {r['profile']:13s} {r['tau_us']:10.1f} us  feasible at "
          f"{r['n_feasible']:2d}/{r['n_deadlines']}  cannot beat the sweep: "
          f"{r['cannot_beat_sweep_anywhere']}  ({r['margin_over_tau_star']} x tau*)")
