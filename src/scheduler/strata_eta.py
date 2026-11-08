#!/usr/bin/env python3
"""Rule R28: the exhaustive sweep's efficiency within each prespecified SNR stratum, against every
exclusion's threshold (protocol.json analysis 'scope_checks', declared before this script ran; R28
itself was set after every earlier result and a review of the paper had been seen).

Nothing is simulated.  The evaluation's stored per-stratum means (validation.json, results.json,
heldout.json: the seven non-training scenarios, ten seeds, each user weighted equally) give, at each
period T of the grid, the sweep's mean effective rate R_k(T) and the genie-aided rate G_k in each
stratum k of the best beam's SNR (protocol.json strata: edges 0, 10 and 20 dB).  Under block fading
the sweep's efficiency in a stratum,

  eta_k = R_k(T) / (G_k (1 - d_S / T)),   d_S the sweep's stored completion time,

does not depend on T > d_S.  It is read at T_max, where the stored rounding (five decimals) moves it
least.  The stored means are point estimates; no bound is computed.

Thresholds (reconcile.json, margins.json, rho.json).  For each implementation finishing within
T_max that Proposition 1(a) can exclude, the lowest efficiency at which it excludes it up to T_max:
at the measured inputs, (T_max - d_L) / (T_max - 64 t_p), t_p its profile's per-probe time, and at
the adverse inputs, (T_max - d_L_lo) / (T_max - d_S_hi).  Reported per scenario and stratum: the
share of users, the share of the genie-aided rate, eta_k and every threshold above it.

Gates:
  G1_pins     the inputs are those protocol.json records, and the strata are the protocol's;
  G2_pooled   the strata, weighted by their stored user counts, reproduce each scenario's stored
              mean rate at T_max (five places) and the sweep's stored efficiency (four places);
  G3_T_free   at every period of the grid from 4 ms, eta_k agrees with its value at T_max within
              the stored rounding.

usage:  python3 strata_eta.py [--check-only]      -> strata_eta.json (written once)
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol

OUT = os.path.join(HERE, 'strata_eta.json')
J = lambda f: json.load(open(os.path.join(HERE, f)))
INPUTS = ['validation.json', 'results.json', 'heldout.json', 'reconcile.json', 'margins.json',
          'rho.json']
PROFILE = {'cpu-1t-pcore': 'cpu1t_pcore', 'cpu-1t-ecore': 'cpu1t_ecore',
           'cpu-8t-pcore': 'cpu8t_pcore', 'gpu-resident': 'gpu'}
HALF = 0.5e-5                      # half a unit in the stored means' fifth decimal


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    check = '--check-only' in sys.argv
    assert check or not os.path.exists(OUT), "strata_eta.json exists: it is not rewritten"
    P = protocol.load()
    rec = protocol.analysis('scope_checks', P)
    edges = P['strata']['edges_dB']
    gate = {'G1_pins': all(sha(os.path.join(HERE, f)) == rec['inputs_sha256'].get(f)
                           for f in INPUTS) and edges[1:4] == [0, 10, 20]}
    t_max = float(max(P['coherence_grid_us']))
    grid = [float(t) for t in P['coherence_grid_us']]
    sites = J('validation.json')['sites'] + J('results.json')['sites'] + J('heldout.json')['sites']
    rho = J('rho.json')['profiles']
    mg = J('margins.json')
    d_hi = mg['dS_hi_us']
    assert mg['t_max_us'] == t_max

    thr = {}
    for r in J('reconcile.json')['margin_rows']:
        tp = rho[PROFILE[r['profile']]]['t_probe_us']
        meas = (t_max - r['dL_us']) / (t_max - 64 * tp)
        if r['dL_us'] < t_max and meas <= 1:
            thr[r['component'] + '|' + r['profile']] = {
                'measured': meas, 'adverse': (t_max - r['dL_lo_us']) / (t_max - d_hi)}

    g2 = g3 = True
    out_sites = {}
    for s in sites:
        M, n = s['methods'], s['n']
        cnt = s['strata_counts']
        assert sum(cnt) == n
        dS = M['sweep64']['cost_us']
        pt = lambda m, T: M[m]['per_T'][str(int(T))]
        Rk = pt('sweep64', t_max)['strata']
        Gk = pt('oracle', t_max)['strata']
        # pooled from the strata, each weighted by its count
        R_pool = sum(c * r for c, r in zip(cnt, Rk)) / n
        G_pool = sum(c * g for c, g in zip(cnt, Gk)) / n
        g2 &= abs(R_pool - pt('sweep64', t_max)['mean']) <= 2 * HALF
        g2 &= abs(R_pool / (G_pool * (1 - dS / t_max)) - M['sweep64']['eta']) <= 6e-5
        rows = []
        for k, (c, r, g) in enumerate(zip(cnt, Rk, Gk)):
            if not c:
                rows.append(None)
                continue
            eta_k = r / (g * (1 - dS / t_max))
            for T in grid:
                if T < 4000:
                    continue
                q = pt('sweep64', T)['strata'][k] / (pt('oracle', T)['strata'][k] * (1 - dS / T))
                tol = 2 * (HALF / (g * (1 - dS / T)) + HALF / (g * (1 - dS / t_max)) + HALF / g)
                g3 &= abs(q - eta_k) <= tol
            rows.append({'users': c, 'share_users': round(c / n, 4),
                         'share_G': round(c * g / (n * G_pool), 4), 'G_k': g,
                         'eta': round(eta_k, 5),
                         'below': sorted(f"{key}:{kind}" for key, v in thr.items()
                                         for kind, x in v.items() if eta_k < x)})
        out_sites[s['site']] = {'n': n, 'd_S_us': dS, 'eta_pooled': M['sweep64']['eta'],
                                'strata': rows}
    gate['G2_pooled'] = bool(g2)
    gate['G3_T_free'] = bool(g3)
    gate['holds'] = all(gate.values())

    low = {k: min((v['strata'][k]['eta'], s) for s, v in out_sites.items() if v['strata'][k])
           for k in range(4)}
    summary = {'lowest_eta_by_stratum': {str(k): {'eta': e, 'site': s} for k, (e, s) in low.items()},
               'thresholds': {k: {kk: round(x, 5) for kk, x in v.items()} for k, v in thr.items()},
               'below': sorted({b + ' @ ' + s + f' stratum {k}' for s, v in out_sites.items()
                                for k, r in enumerate(v['strata']) if r for b in r['below']})}
    print(json.dumps(gate))
    for s, v in out_sites.items():
        print(f"{s[:20]:20s}", '  '.join(f"[{k}] {r['share_users']:.3f}u {r['share_G']:.3f}G "
                                         f"eta {r['eta']:.4f}" for k, r in enumerate(v['strata']) if r))
    print('thresholds', summary['thresholds'])
    print('below', summary['below'])
    if not gate['holds']:
        raise SystemExit("gate failed: nothing written")
    if check:
        print("gates hold; nothing written")
        return
    out = {'rule': 'R28', 'script_sha256': sha(os.path.abspath(__file__)),
           'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in INPUTS},
           'T_max_us': t_max, 'edges_dB': edges, 'gate': gate, 'summary': summary,
           'sites': out_sites}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
