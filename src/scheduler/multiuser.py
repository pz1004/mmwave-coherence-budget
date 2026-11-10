#!/usr/bin/env python3
"""Rule R48: one base-station panel shared by users that acquire together (protocol.json analysis
'multiuser', declared before it ran; R48 is post hoc).

The model of Proposition 4.  One analog panel serves one beam at a time.  U cold users start
acquiring at time 0; W warm users already hold beams.  Probing is shared: every cold user measures
the same probes (the sweep's M or the K-probe method's K), and the panel carries no data while it
probes.  The K-probe method infers on one processor, one cold user after another, each call taking
tau from the end of the probes, so cold user u's beam is chosen at K t_p + u tau; under the sweep
every cold user's beam is chosen at M t_p.  A warm user is ready once the probes end, a cold user
once its beam is chosen; the panel is shared equally among the users ready at each instant (fluid
round robin, so it idles only when no user is ready), each transmitting at its beam's rate.

Rates.  The decision audit's two cities (data/<city>.npz and audit_pop.npz, as covpop.audit reads
them), rule R37's covered users: the genie-aided rate g (the best of the 64 beams) and the sweep's
rate s (the beam it selects with pilots of 1000 samples, averaged over realizations 1-10).  A
perfect predictor's cold user and every warm user transmit at g, the sweep's cold user at s.
Timing: the audit's reference, t_p = 20.966 us (d_S = 64 t_p), K = 4, T_max = 10 ms.

Draws.  Per city, U in (1, 2, 4, 8, 16) and W in (0, 1, 8): 200 draws of U + W distinct covered
users, numpy default_rng(evaluate.stable_seed('multiuser', city, U, W)).choice(n, U + W,
replace=False), the first U cold.  Latencies: the paper's implementations (the lowest charge of each
FP32 one in reconcile.json, the lowest session 95th percentile of each reduced-precision build in
matched.json and deploy.json) and a grid.  Per draw and latency: whether the cell's data under the
perfect predictor are at most the sweep's at every T in (0, T_max] (the cell excluded), and the same
for the cold users' data alone.  Both differences are piecewise linear in T with kinks at the ready
times, so the check at every kink and at T_max is exact.

Gates, before anything is written:
  M1  U = 1, W = 0: the data equal g (T - K t_p - tau)^+ and s (T - M t_p)^+ to 1e-9 relative at every
      T of the protocol's grid, and the cell verdict equals Proposition 1(a)'s per-link condition
      tau >= tau* + (1 - s/g)(T_max - M t_p) on every draw;
  M2  K = M, tau = 0 and the predictor's rates set to the sweep's: the two cells' data are equal at
      every grid T, for every U and W;
  M3  the panel's idle time after the probes, from the simulation, is zero when W >= 1 and
      K t_p + tau - K t_p = tau (the first cold user's inference) when W = 0, under the predictor.

usage:  python3 multiuser.py [--gate]      -> multiuser.json (refuses to overwrite it)
"""
import hashlib, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'multiuser.json')
KEY = 'multiuser'
US = (1, 2, 4, 8, 16)
WS = (0, 1, 8)
DRAWS = 200
M_, K_ = 64, 4
NS = 1000                          # the pilot whose selections give the sweep's rates
T_MAX = 10_000.0
GRID_TAU = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def J(name):
    return json.load(open(os.path.join(HERE, name)))


def rates():
    """Per city: (g, s) of the covered users."""
    import covcurve, covpop, audit_screen as AS
    arr, _ = covcurve.arrays()
    out = {}
    for c in AS.CITIES:
        g = arr[c + '.g']
        s = arr[c + '.s'][AS.SAMPLES.index(NS)].mean(0)
        m = covpop.keep(g, 'cqi1')
        out[c] = (g[m], s[m])
    return out


def implementations():
    RC = J('reconcile.json')
    lat = {}
    for r in RC['r5_rows']:
        key = {('resnet18_vision', 'cpu-8t-pcore'): 'rn_cpu8t_fp32',
               ('resnet18_vision', 'gpu-resident'): 'rn_gpu_fp32',
               ('vit_b16_vision', 'gpu-resident'): 'vit_gpu_fp32'}.get((r['component'], r['profile']))
        if key:
            lat[key] = min(r['charges_us'].values())
    lat['vit_gpu_fp16'] = min(v['builds']['fp16_graph']['p95_us']
                              for v in J('matched.json')['sessions'].values())
    D = J('deploy.json')['groups']
    lat['rn_gpu_fp16c'] = min(v['builds']['fp16_cmax']['p95_us'] for v in D['rn_gpu']['sessions'].values())
    lat['rn_cpu8t_int8'] = min(v['builds']['int8']['p95_us'] for v in D['rn_cpu']['sessions'].values())
    return lat


def timing():
    import audit_screen as AS
    tp = AS.d_s_us(NS) / M_
    return tp, M_ * tp


def simulate(ready, r, Ts):
    """Fluid round robin: each user's data at each T (users x Ts), and the panel's idle time
    after `start` up to T_max is computed by the caller from the ready times."""
    fin = np.isfinite(ready)
    ev = np.unique(ready[fin])
    ends = np.append(ev[1:], np.inf)
    mask = ready[None, :] <= ev[:, None]                     # interval j: users ready at its start
    n = mask.sum(1)
    share = np.where(mask, r[None, :] / n[:, None], 0.0)     # (J, users)
    L = np.clip(np.minimum(ends[None, :], Ts[:, None]) - ev[None, :], 0.0, None)   # (T, J)
    return L @ share                                         # (T, users)


def cell(ready_L, r_L, ready_S, r_S, cold):
    """Excluded (cell, cold users): the predictor's data at most the sweep's at every T in
    (0, T_max], checked at every kink of either method and at T_max."""
    kinks = np.unique(np.concatenate([ready_L[np.isfinite(ready_L)], ready_S[np.isfinite(ready_S)],
                                      [T_MAX]]))
    Ts = kinks[(kinks > 0) & (kinks <= T_MAX)]
    DL, DS = simulate(ready_L, r_L, Ts), simulate(ready_S, r_S, Ts)
    tol = 1e-12 * max(1.0, float(np.abs(DS).max()))
    ex_cell = bool(np.all(DL.sum(1) - DS.sum(1) <= tol))
    ex_cold = bool(np.all(DL[:, cold].sum(1) - DS[:, cold].sum(1) <= tol))
    return ex_cell, ex_cold


def ready_times(U, W, tau, tp, dS, K=K_):
    a_L = K * tp
    rl = np.concatenate([a_L + tau * np.arange(1, U + 1), np.full(W, a_L)])
    rs = np.concatenate([np.full(U, dS), np.full(W, dS)])
    return rl, rs


def idle_after_probes(ready, start):
    """The panel's idle time on (start, T_max]: before the first ready user."""
    first = ready[np.isfinite(ready)].min()
    return max(0.0, min(first, T_MAX) - start)


def gate(R, tp, dS, lat):
    import evaluate
    P = J('protocol.json')
    grid = np.array([float(t) for t in P['coherence_grid_us']])
    bad, g = [], {}
    taus = sorted(set(GRID_TAU) | set(lat.values()))
    for c, (gr, sr) in R.items():
        rng = np.random.default_rng(evaluate.stable_seed(KEY, 'gate', c))
        for d in range(50):
            u = rng.integers(gr.size)
            for tau in taus:
                rl, rs = ready_times(1, 0, tau, tp, dS)
                DL = simulate(rl, gr[[u]], grid)[:, 0]
                DS = simulate(rs, sr[[u]], grid)[:, 0]
                cl = gr[u] * np.maximum(grid - K_ * tp - tau, 0.0)
                cs = sr[u] * np.maximum(grid - dS, 0.0)
                if not (np.allclose(DL, cl, rtol=1e-9, atol=1e-9) and
                        np.allclose(DS, cs, rtol=1e-9, atol=1e-9)):
                    bad.append(('M1 data', c, d, tau))
                ex, _ = cell(rl, gr[[u]], rs, sr[[u]], np.array([0]))
                eta = sr[u] / gr[u]
                lhs, rhs = tau, (M_ - K_) * tp + (1 - eta) * (T_MAX - dS)
                if abs(lhs - rhs) > 1e-6 and ex != (lhs >= rhs):
                    bad.append(('M1 verdict', c, d, tau, ex))
    g['M1_single_link'] = not [b for b in bad if b[0].startswith('M1')]
    for c, (gr, sr) in R.items():
        rng = np.random.default_rng(evaluate.stable_seed(KEY, 'gate2', c))
        for U in US:
            for W in WS:
                idx = rng.choice(gr.size, U + W, replace=False)
                r = np.concatenate([sr[idx[:U]], gr[idx[U:]]])
                rl, rs = ready_times(U, W, 0.0, tp, dS, K=M_)
                if not np.allclose(simulate(rl, r, grid), simulate(rs, r, grid), rtol=1e-12, atol=1e-12):
                    bad.append(('M2', c, U, W))
                for tau in taus:
                    rl, _ = ready_times(U, W, tau, tp, dS)
                    want = 0.0 if W else tau
                    if abs(idle_after_probes(rl, K_ * tp) - min(want, T_MAX - K_ * tp)) > 1e-9:
                        bad.append(('M3', c, U, W, tau))
    g['M2_equal_methods_equal_data'] = not [b for b in bad if b[0] == 'M2']
    g['M3_idle_time'] = not [b for b in bad if b[0] == 'M3']
    g['holds'] = not bad
    return g, bad


def main():
    import evaluate
    R = rates()
    tp, dS = timing()
    lat = implementations()
    g, bad = gate(R, tp, dS, lat)
    print(f"gate: {g}")
    if bad:
        sys.exit(f"multiuser: gate failed: {bad[:10]}")
    if '--gate' in sys.argv:
        return
    assert not os.path.exists(OUT), f"{OUT} exists: written once"
    taus = {**{f"grid_{int(t)}": t for t in GRID_TAU}, **lat}
    res = {}
    for c, (gr, sr) in R.items():
        for U in US:
            for W in WS:
                rng = np.random.default_rng(evaluate.stable_seed(KEY, c, U, W))
                draws = [rng.choice(gr.size, U + W, replace=False) for _ in range(DRAWS)]
                for name, tau in taus.items():
                    rl, rs = ready_times(U, W, tau, tp, dS)
                    n_cell = n_cold = n_link = 0
                    for idx in draws:
                        r_L = gr[idx]
                        r_S = np.concatenate([sr[idx[:U]], gr[idx[U:]]])
                        ex_cell, ex_cold = cell(rl, r_L, rs, r_S, np.arange(U))
                        eta1 = sr[idx[0]] / gr[idx[0]]
                        n_link += tau >= (M_ - K_) * tp + (1 - eta1) * (T_MAX - dS)
                        n_cell += ex_cell
                        n_cold += ex_cold
                    res[f"{c}|{U}|{W}|{name}"] = {
                        'city': c, 'U': U, 'W': W, 'latency': name, 'tau_us': tau,
                        'cell_excluded_share': n_cell / DRAWS,
                        'cold_excluded_share': n_cold / DRAWS,
                        'first_user_link_excluded_share': n_link / DRAWS,
                        'last_user_ready_us': K_ * tp + U * tau, 'sweep_ready_us': dS}
    out = {'rule': 'R48', 'script_sha256': sha(os.path.abspath(__file__)), 'gate': g,
           't_p_us': tp, 'd_S_us': dS, 'K': K_, 'M': M_, 'tau_star_us': (M_ - K_) * tp,
           'T_max_us': T_MAX, 'users': {c: int(v[0].size) for c, v in R.items()},
           'latencies_us': taus, 'draws': DRAWS, 'U': list(US), 'W': list(WS), 'results': res,
           'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in (
               'reconcile.json', 'matched.json', 'deploy.json', 'audit_pop.npz')}}
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, v in res.items():
        if v['latency'] in lat:
            print(k, v['cell_excluded_share'], v['cold_excluded_share'],
                  v['first_user_link_excluded_share'])
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
