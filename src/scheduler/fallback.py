#!/usr/bin/env python3
"""Rule R27: a procedure that sends data on the best probed beam while the predictor infers
(protocol.json analysis 'scope_checks', declared before this script ran; R27 itself was set after
every earlier result and a review of the paper had been seen).

Nothing is timed, trained or simulated: the script reads stored efficiencies and latencies.

Procedure.  A 4-probe method probes K = 4 narrow beams in Top-4's order.  From K t_p it sends data
on the best of them (Top-4's selection) and from d = K t_p + tau on the predicted beam; switching
costs nothing, as in the per-probe model under block fading.  For a perfect predictor its rate over
a period T, in units of the genie-aided rate G times time, is

  phi_fb(T)   = eta_4 (min(T, d) - K t_p)^+ + (T - d)^+,

and for the paper's procedure, which sends nothing until the inference ends,

  phi_idle(T) = (T - d)^+.

Reference.  The envelope E(T) = max_j eta_j (T - d_j)^+ of the fixed methods that need no training:
Top-K (K = 1, 2, 4, 8, 16, 32), the exhaustive sweep and Hier-3 to Hier-6, each at its measured
efficiency in the scenario (results.json, heldout.json: the six evaluation scenarios, ten seeds)
and d_j = K_j t_p, t_p the per-probe time of the implementation's hardware profile (rho.json), as
Table IV takes it for the sweep.

Verdict.  Per scenario, implementation and latency reading, a procedure is excluded if
max over 0 < T <= T_max of phi(T) - E(T) is at most eps T_max (eps = 1e-6, rule R26's), checked
exactly: phi - E is piecewise linear with kinks at K t_p, d, every d_j and every crossing of two
reference lines, so its maximum over (0, T_max] lies at one of them or at T_max.  The advantage is
max over T of (phi(T) - E(T)) / T, a share of G, with the T that attains it.

Latency readings (reconcile.json): Table IV's tau, the 95th percentile of the fastest runtime; its
lower bound, Table IV's adverse input; and, for the three implementations timed in repeated runs,
the fastest recorded call.  phi_fb does not increase in d, so an exclusion at the fastest call
holds for every recorded call.

Also recorded: Top-4's crossover with the sweep alone (Proposition 2), below which Top-4, and so
phi_fb, delivers more than the sweep: against the sweep alone no such procedure can be excluded.

Gates:
  G1_pins   the inputs are those protocol.json records, and every reference's stored completion
            time equals K_j t_p (P-core) within 0.01 us;
  G2_idle   phi_idle against the sweep alone gives, in every scenario at its measured efficiency,
            the verdict of Proposition 1(a) at that efficiency: excluded exactly when
            d >= eta_S d_S + (1 - eta_S) T_max;
  G3_exact  no point of a scan every 1 us exceeds the exact maximum, which exceeds the scan's by at
            most 1 us (every slope of phi - E lies in [-1, 1]).

usage:  python3 fallback.py [--check-only]      -> fallback.json (written once)
"""
import hashlib, itertools, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol

OUT = os.path.join(HERE, 'fallback.json')
J = lambda f: json.load(open(os.path.join(HERE, f)))
K_FB = 4
EPS = 1e-6
# the fixed methods that need no training, with their probe counts
REFS = {**{f'topk{k}': k for k in (1, 2, 4, 8, 16, 32)}, 'sweep64': 64,
        **{f'hier{l}': 2 * l for l in (3, 4, 5, 6)}}
PROFILE = {'cpu-1t-pcore': 'cpu1t_pcore', 'cpu-1t-ecore': 'cpu1t_ecore',
           'cpu-8t-pcore': 'cpu8t_pcore', 'gpu-resident': 'gpu'}
INPUTS = ['results.json', 'heldout.json', 'reconcile.json', 'rho.json']


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def env(T, refs):
    return max([e * max(T - d, 0.0) for d, e in refs] + [0.0])


def phi(T, d, kind, eta4, d4):
    if kind == 'idle':
        return max(T - d, 0.0)
    return eta4 * max(min(T, d) - d4, 0.0) + max(T - d, 0.0)


def kinks(refs, d, d4, t_max):
    pts = {d4, d, t_max} | {dj for dj, _ in refs}
    for (d1, e1), (d2, e2) in itertools.combinations(refs, 2):
        if e1 != e2:
            pts.add((e1 * d1 - e2 * d2) / (e1 - e2))
    return sorted(t for t in pts if 0 < t <= t_max)


def worst(d, kind, refs, eta4, d4, t_max):
    """max over (0, T_max] of phi - E, exactly, with the share of G and its T."""
    best = (-np.inf, None, None)
    for T in kinks(refs, d, d4, t_max):
        g = phi(T, d, kind, eta4, d4) - env(T, refs)
        if g > best[0]:
            best = (g, T, g / T)
    return best


def scan(d, kind, refs, eta4, d4, t_max):
    Ts = np.arange(1.0, t_max + 0.5, 1.0)
    E = np.max([e * np.clip(Ts - dj, 0, None) for dj, e in refs] + [np.zeros_like(Ts)], axis=0)
    F = np.clip(Ts - d, 0, None)
    if kind == 'fb':
        F = F + eta4 * np.clip(np.minimum(Ts, d) - d4, 0, None)
    return float(np.max(F - E))


def main():
    check = '--check-only' in sys.argv
    assert check or not os.path.exists(OUT), "fallback.json exists: it is not rewritten"
    P = protocol.load()
    rec = protocol.analysis('scope_checks', P)
    gate = {}
    gate['G1_pins'] = all(sha(os.path.join(HERE, f)) == rec['inputs_sha256'].get(f) for f in INPUTS)
    t_max = float(max(P['coherence_grid_us']))
    rho = J('rho.json')['profiles']
    tp_p = rho['cpu1t_pcore']['t_probe_us']
    sites = J('results.json')['sites'] + J('heldout.json')['sites']
    rc = J('reconcile.json')
    rep = {v['component'] + '|' + v['profile']: k for k, v in rc['repeated'].items()}
    ok_cost = True
    for s in sites:
        for m, k in REFS.items():
            ok_cost &= abs(s['methods'][m]['cost_us'] - k * tp_p) < 0.01
    gate['G1_pins'] = gate['G1_pins'] and ok_cost

    out_sites, g2, g3 = {}, True, True
    for s in sites:
        M = s['methods']
        eta = {m: M[m]['eta'] for m in REFS}
        eS, e4 = eta['sweep64'], eta['topk4']
        cross = (eS * 64 * tp_p - e4 * K_FB * tp_p) / (eS - e4)
        rows = {}
        for r in rc['margin_rows']:
            key = r['component'] + '|' + r['profile']
            tp = rho[PROFILE[r['profile']]]['t_probe_us']
            refs = [(k * tp, eta[m]) for m, k in REFS.items()]
            sweep = [(64 * tp, eS)]
            d4 = K_FB * tp
            reads = {'p95': r['tau_us'], 'lower_bound': r['tau_lo_us']}
            if key in rep:
                reads['fastest_call'] = rc['fastest_call_repeated_us'][rep[key]]
            row = {'t_p_us': tp, 'readings': {}}
            for rd, tau in reads.items():
                d = d4 + tau
                res = {'tau_us': tau, 'd_us': round(d, 3)}
                for kind, rf, lab in (('idle', sweep, 'idle_vs_sweep'), ('idle', refs, 'idle_vs_fixed'),
                                      ('fb', refs, 'fallback_vs_fixed'), ('fb', sweep, 'fallback_vs_sweep')):
                    g, T, adv = worst(d, kind, rf, e4, d4, t_max)
                    res[lab] = {'excluded': bool(g <= EPS * t_max),
                                'max_gap_us': round(g, 4), 'at_T_us': round(T, 3),
                                'advantage_share_of_G': round(adv, 6)}
                    sc = scan(d, kind, rf, e4, d4, t_max)
                    g3 &= sc <= g + 1e-9 * t_max and g - sc <= 1.0
                p1a = d >= eS * 64 * tp + (1 - eS) * t_max
                g2 &= res['idle_vs_sweep']['excluded'] == p1a
                row['readings'][rd] = res
            rows[key] = row
        out_sites[s['site']] = {'eta_topk4': e4, 'eta_sweep': eS,
                                'topk4_crossover_with_sweep_us': round(cross, 3),
                                'eta_refs': eta, 'implementations': rows}
    gate['G2_idle'] = bool(g2)
    gate['G3_exact'] = bool(g3)
    gate['holds'] = all(gate.values())

    keys = [r['component'] + '|' + r['profile'] for r in rc['margin_rows']]
    summary = {}
    for k in keys:
        summary[k] = {}
        for rd in ('p95', 'lower_bound', 'fastest_call'):
            if rd not in out_sites[sites[0]['site']]['implementations'][k]['readings']:
                continue
            summary[k][rd] = {lab: sum(out_sites[s['site']]['implementations'][k]['readings'][rd][lab]
                                       ['excluded'] for s in sites)
                              for lab in ('idle_vs_sweep', 'idle_vs_fixed', 'fallback_vs_fixed',
                                          'fallback_vs_sweep')}
    print(json.dumps(gate))
    for s in sites:
        o = out_sites[s['site']]
        print(f"{s['site'][:18]:18s} eta4 {o['eta_topk4']:.4f}  Top-4 beats the sweep below "
              f"{o['topk4_crossover_with_sweep_us'] / 1e3:.2f} ms")
    for k in keys:
        print(f"{k:32s}", '  '.join(f"{rd}: {v['idle_vs_sweep']}/{v['idle_vs_fixed']}/"
                                     f"{v['fallback_vs_fixed']}/{v['fallback_vs_sweep']}"
                                     for rd, v in summary[k].items()))
    if not gate['holds']:
        raise SystemExit("gate failed: nothing written")
    if check:
        print("gates hold; nothing written")
        return
    out = {'rule': 'R27', 'script_sha256': sha(os.path.abspath(__file__)),
           'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in INPUTS},
           'T_max_us': t_max, 'eps': EPS, 'K': K_FB, 'references': REFS, 'gate': gate,
           'summary_excluded_scenarios': summary, 'sites': out_sites}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
