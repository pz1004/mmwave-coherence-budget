#!/usr/bin/env python3
"""Robust exclusion margins from the recorded timings.  Declared in protocol.json (analysis
'csirs_margins_calib') before it ran; nothing is timed.

Proposition 1(b) in completion-time form excludes an implementation for every T <= T_max when
d_L >= (1 - eta_S) T_max + eta_S d_S.  Its right side grows with d_S and, for T_max > d_S, falls
as eta_S grows, so the condition holds whenever a lower bound dL_lo <= d_L, a lower bound
eta_lo <= eta_S and an upper bound dS_hi >= d_S satisfy
    m = dL_lo - (1 - eta_lo) T_max - eta_lo dS_hi >= 0,
the exclusion margin.  Here
  dL_lo   K t_air + the smallest, over the implementation's runtimes, of a lower value for its
          95th-percentile latency.  The uncompiled runtime stored every timed call
          (src/latency/p_<profile>_raw.npz): its value is the smallest over the three timing seeds
          of the distribution-free lower confidence bound X_(r) of the seed's 95th percentile, r
          the largest rank with P(Binomial(n, 0.95) >= r) >= 1 - 0.05/3, so that the three hold
          together with probability at least 95% for calls drawn independently within a run.  The
          TorchScript and the full-count GPU runtimes stored only per-seed 95th percentiles: their
          value is the smallest of the three, a point value, flagged as such.  t_air = 20 us; the
          per-probe processing is left out.
  dS_hi   64 (t_air + the largest per-seed 95th percentile of the per-probe processing recorded
          on any profile).
  eta_lo  csirs.json's lower bound on the sweep's efficiency: the probe model (coherent, N =
          1000) for Table I, each CSI-RS density for the procedure.
Under the CSI-RS procedure (make_draft._nr_eta) d_L is the end of the 4th resource, less one slot,
plus tau, and d_S the end of the 64th, at 12 one-symbol resources per 125 us slot with every
common delay zero.  The bounds cover the sampling of calls within a timing run, not drift between
runs, which reached 9.9% on the shared GPU (p_gpu_heavy_full.json).

Gates: calibrate.nearest_rank_p95 of every stored raw array equals the stored per-seed 95th
percentile; every runtime's stored charge is the largest of its per-seed values and equals R5's.

usage:  python3 margins.py [--check-only]       -> margins.json
"""
import hashlib, json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = os.path.normpath(os.path.join(HERE, '..', 'latency'))
sys.path.insert(0, HERE)
sys.path.insert(0, LAT)
import obs
from calibrate import nearest_rank_p95

PROF = {'cpu-1t-pcore': 'cpu1t_pcore', 'cpu-1t-ecore': 'cpu1t_ecore',
        'cpu-8t-pcore': 'cpu8t_pcore', 'gpu-resident': 'gpu'}
EAGER, FULL, TS = 'eager, Phase-1 counts', 'eager, full counts', 'TorchScript, whole decision'
ALPHA, Q, SEEDS = 0.05, 0.95, (17, 29, 43)
SLOT, NSYM = 125.0, 12
J = lambda *p: json.load(open(os.path.join(*p)))


def rank(n, alpha):
    """The largest r with P(Binomial(n, Q) >= r) >= 1 - alpha: then the r-th smallest of n calls
    is below the Q-quantile of their distribution with probability at least 1 - alpha."""
    r = np.arange(1, n + 1)
    ok = r[stats.binom.sf(r - 1, n, Q) >= 1 - alpha]
    assert ok.size, (n, alpha)
    return int(ok.max())


def runtime_values(row, raw, p, o2, full):
    """For one R5 row: each runtime's charge and lower value, and the gates on them."""
    comp, prof = row['component'], row['profile']
    out, gates = {}, []
    pr = next(x for x in p['rows'] if x['component'] == comp)
    p95, lcb, ns = [], [], []
    for s in pr['per_seed']:
        a = raw[f"{comp}.s{s['seed']}"]
        assert a.size == s['n_timed'], (comp, prof, s['seed'])
        gates.append(nearest_rank_p95(a) == s['p95_ns'])
        r = rank(a.size, ALPHA / len(pr['per_seed']))
        lcb.append(float(np.sort(a)[r - 1]) / 1e3)
        p95.append(s['p95_ns'] / 1e3)
        ns.append(int(a.size))
    gates.append(abs(max(p95) - pr['charge_us']) < 1e-9 and
                 abs(pr['charge_us'] - row['charges_us'][EAGER]) < 1e-9)
    out[EAGER] = {"charged_us": pr['charge_us'], "lower_us": min(lcb), "seed_p95_us": p95,
                  "seed_lower_us": lcb, "n_timed": ns, "kind": "order-statistic bound"}
    for name, src in ((TS, o2), (FULL, full)):
        if name not in row['charges_us']:
            continue
        x = next(x for x in src['rows'] if x['component'] == comp)
        sp = [s['p95_ns'] / 1e3 for s in x['per_seed']]
        gates.append(abs(max(sp) - x['charge_us']) < 1e-9 and
                     abs(x['charge_us'] - row['charges_us'][name]) < 1e-9)
        out[name] = {"charged_us": x['charge_us'], "lower_us": min(sp), "seed_p95_us": sp,
                     "n_timed": [s['n_timed'] for s in x['per_seed']],
                     "kind": "smallest seed p95 (point value)"}
    assert set(out) == set(row['charges_us']), (comp, prof, sorted(out))
    return out, gates


def nr_end(j, n=NSYM):
    return (j // n) * SLOT + (14 - n + j % n + 1) * SLOT / 14


def main():
    check = '--check-only' in sys.argv
    r5 = J(HERE, 'rules_rx.json')['R5']
    K = r5['K']
    t_max = max(J(HERE, 'protocol.json')['coherence_grid_us'])
    proc = max(max(next(r for r in J(HERE, f)['rows'] if r['component'] == 'probe_update')
                   ['per_seed_us'])
               for f in ('lat_cpu1t_pcore.json', 'lat_cpu1t_ecore.json', 'lat_cpu8t_pcore.json',
                         'lat_gpu.json'))
    ds_hi = 64 * (obs.T_P_US + proc)
    full = J(LAT, 'p_gpu_heavy_full.json')
    rows, gates = [], []
    for row in r5['rows']:
        prof = PROF[row['profile']]
        raw = np.load(os.path.join(LAT, f'p_{prof}_raw.npz'))
        p = J(LAT, f'p_{prof}.json')
        o2 = J(LAT, f'o2_{prof}.json') if os.path.exists(os.path.join(LAT, f'o2_{prof}.json')) \
            else {'rows': []}
        rt, g = runtime_values(row, raw, p, o2, full)
        gates += g
        fastest = min(rt, key=lambda k: rt[k]['charged_us'])
        tau_lo = min(v['lower_us'] for v in rt.values())
        rows.append({"component": row['component'], "profile": row['profile'],
                     "runtimes": rt, "fastest_runtime": fastest,
                     "tau_us": rt[fastest]['charged_us'], "tau_lo_us": tau_lo,
                     "tau_lo_runtime": min(rt, key=lambda k: rt[k]['lower_us']),
                     "tau_lo_kind": min(rt.values(), key=lambda v: v['lower_us'])['kind'],
                     "dL_us": K * row['t_p_us'] + rt[fastest]['charged_us'],
                     "dL_lo_us": K * obs.T_P_US + tau_lo,
                     "nr_dL_lo_us": nr_end(K - 1) - SLOT + tau_lo})
    gate = {"n_checks": len(gates), "holds": bool(all(gates))}
    print(json.dumps(gate))
    if not gate['holds']:
        raise SystemExit("gate failed: nothing written")
    if check:
        print("gates hold; nothing written")
        return
    cs = J(HERE, 'csirs.json')
    eta_probe = cs['configs']['coherent|1000']['eta_lb_min']
    eta_nr = {d: cs['configs'][f'coherent|{n}']['eta_lb_min']
              for d, n in cs['density_samples'].items()}
    ds_nr = nr_end(63)
    m = lambda dl, eta, ds: dl - (1 - eta) * t_max - eta * ds
    for r in rows:
        r["margin_us"] = m(r['dL_lo_us'], eta_probe, ds_hi)
        r["excluded_robust"] = r["margin_us"] >= 0
        r["infeasible_robust"] = r['dL_lo_us'] >= t_max
        r["nr_margin_us"] = {d: m(r['nr_dL_lo_us'], e, ds_nr) for d, e in eta_nr.items()}
    out = {"script_sha256": hashlib.sha256(
               open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "alpha": ALPHA, "quantile": Q, "K": K, "t_max_us": t_max, "t_air_us": obs.T_P_US,
           "probe_processing_max_us": proc, "dS_hi_us": ds_hi, "nr_dS_us": ds_nr,
           "eta_lo": {"probe_model": eta_probe, "csirs": eta_nr}, "gate": gate, "rows": rows}
    pth = os.path.join(HERE, 'margins.json')
    assert not os.path.exists(pth), "margins.json exists: it is not rewritten"
    json.dump(out, open(pth, 'w'), indent=1)
    for r in rows:
        print(f"{r['component']:18s} {r['profile']:13s} tau {r['tau_us']:10.1f} lo "
              f"{r['tau_lo_us']:10.1f} ({r['tau_lo_kind'][:5]}) margin {r['margin_us']:10.1f}")
    print("wrote margins.json")


if __name__ == '__main__':
    main()
