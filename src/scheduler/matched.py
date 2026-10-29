#!/usr/bin/env python3
"""Rule R22: reading the matched implementation-profiling study (protocol.json analysis 'matched').

  python3 matched.py --freeze   applies R22 (ii) to the pilot (../latency/profile_matched/pilot.*)
                                and writes the analysis's frozen record into protocol.json, once,
                                before any main session runs
  python3 matched.py            reads the main sessions (R22 (iii)-(vii)) into matched.json

The candidate is an image-only ViT-B/16 (K = 0) timed from the frame in pinned host memory to the
beam index on the host (../latency/profile_matched.py).  The reference is the per-probe exhaustive
sweep at adverse inputs, d_S_hi = 64 (20 us + p) and eta_lo from margins.json, T_max = 10 ms, on
the protocol's 12-period grid.  For each session read, build and period T:

  psi_adv(T) = mean_i (T - D_i)^+ - eta_lo (T - d_S_hi)^+

with its one-sided upper bound from batch means (batches of b consecutive calls, Student t with
B - 1 degrees of freedom at 1 - 0.05/12, Bonferroni over the grid within one session and build).
Excluded if the upper bound is at most zero, inconclusive if psi_adv <= 0 < bound, not excluded if
psi_adv > 0.  Every reading is for the recorded completion times of that session.
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAT = os.path.join(HERE, '..', 'latency')
OUTD = os.path.join(LAT, 'profile_matched')
PROT = os.path.join(HERE, 'protocol.json')
OUT = os.path.join(HERE, 'matched.json')
KEY = 'matched'
CANDIDATES = ('fp16_eager', 'fp16_graph', 'fp16_compile')
BATCHES = (100, 200, 500, 1000)
LAST = 10_000                      # the pilot calls each rule reads, the last of each window
ALPHA = 0.05
W_STEP, W_FIRST, W_LAST_J, W_CAP, W_TOL = 30, 2, 8, 300, 1.0
RHO_MAX = 0.10
T_AIR_US = 20.0
LOG_COLS = ('timestamp', 'util_gpu', 'util_mem', 'util_enc', 'clock_sm', 'clock_mem', 'temp',
            'power', 'pstate', 'event_reasons', 'enc_sessions', 'enc_fps')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def rel(p):
    return os.path.relpath(p, HERE)


def gpu_log(path):
    """The nvidia-smi log: one dict of arrays, timestamps as epoch seconds."""
    rows = []
    for line in open(path):
        f = [x.strip() for x in line.split(',')]
        if len(f) != len(LOG_COLS):
            continue
        try:
            ts, frac = f[0].split('.')
            t = time.mktime(time.strptime(ts, '%Y/%m/%d %H:%M:%S')) + float('0.' + frac)
            rows.append((t, *(float(x) if k not in ('pstate', 'event_reasons') else x
                              for k, x in zip(LOG_COLS[1:], f[1:]))))
        except ValueError:
            continue
    cols = list(zip(*rows)) if rows else [[] for _ in LOG_COLS]
    return {k: (np.array(c, dtype=float) if k not in ('pstate', 'event_reasons') else list(c))
            for k, c in zip(LOG_COLS, cols)}


def window_log(L, t0, t1):
    m = (L['timestamp'] >= t0) & (L['timestamp'] <= t1)
    idx = np.nonzero(m)[0]
    if idx.size == 0:
        return None
    take = lambda k: L[k][m]
    fps, enc = take('enc_fps'), take('util_enc')
    return {'samples': int(idx.size),
            'temp_c': [float(take('temp').min()), float(take('temp').max())],
            'clock_sm_mhz': [float(take('clock_sm').min()), float(np.median(take('clock_sm'))),
                             float(take('clock_sm').max())],
            'power_w_mean': float(take('power').mean()),
            'util_gpu_mean': float(take('util_gpu').mean()),
            'pstates': sorted({L['pstate'][i] for i in idx}),
            'event_reasons': sorted({L['event_reasons'][i] for i in idx}),
            'encoder_sessions_max': float(take('enc_sessions').max()),
            'encoder_share_of_samples_active': float(np.mean((fps > 0) | (enc > 0))),
            'encoder_util_max': float(enc.max())}


def protocol():
    P = json.load(open(PROT))
    i = [k for k, a in enumerate(P['analyses']) if a['key'] == KEY]
    assert len(i) == 1, f"protocol.json declares '{KEY}' exactly once"
    return P, i[0]


def write_protocol(P):
    s = json.dumps(P, indent=1, sort_keys=True) + "\n"
    s.encode('ascii')
    open(PROT, 'w').write(s)


def lag1(x):
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def freeze():
    P, i = protocol()
    A = P['analyses'][i]
    assert 'frozen' not in A, "the frozen record is written once"
    pj, pz, pc = (os.path.join(OUTD, f) for f in ('pilot.json', 'pilot.npz', 'pilot_gpu.csv'))
    rec, raw, L = json.load(open(pj)), np.load(pz), gpu_log(pc)
    assert rec['mode'] == 'pilot' and not rec['smoke'] and rec['failed'] is None
    W = rec['windows']
    assert [w['build'] for w in W] == ['fp32', 'fp16_eager', 'fp16_graph', 'fp16_compile', 'fp32']
    t_max = P['coherence_grid_us'][-1]

    def calls(j, w):
        return raw[f"{j}.{w['build']}.ns"] / 1e3          # microseconds

    # (ii) the optimized build: eligible candidates, lowest mean over the last 10,000 calls
    selection = {}
    for j, w in enumerate(W):
        if w['build'] not in CANDIDATES:
            continue
        s = {'built': w['built'], 'runs': w.get('runs', False),
             'execution_checks_pass': bool(w.get('execution_checks', {}).get('passes'))}
        if w['build'] == 'fp16_compile' and w['built']:
            dc = w.get('dynamo_counters', {})
            s['graph_breaks'] = sum(dc.get('graph_break', {}).values())
            s['unique_graphs'] = dc.get('stats', {}).get('unique_graphs')
            s['single_graph'] = s['graph_breaks'] == 0 and s['unique_graphs'] == 1
        s['eligible'] = (s['built'] and s['runs'] and s['execution_checks_pass']
                         and s.get('single_graph', True) and w['n_calls'] >= LAST)
        if s['built'] and s['runs']:
            s['mean_us_last_10000'] = float(calls(j, w)[-LAST:].mean())
        selection[w['build']] = s
    eligible = [b for b, s in selection.items() if s['eligible']]
    assert eligible, "no candidate is eligible: the study reports FP32 only (R22 (ii))"
    build = min(eligible, key=lambda b: selection[b]['mean_us_last_10000'])

    # (ii) the warm-up: FP32 from idle and from hot, and the chosen build
    read = [(0, W[0]), (4, W[4])] + [(j, w) for j, w in enumerate(W) if w['build'] == build]
    warm = {}
    for j, w in read:
        t0 = w['window_start_time']
        m = []
        for k in range(10):
            sel = (L['timestamp'] >= t0 + W_STEP * k) & (L['timestamp'] < t0 + W_STEP * (k + 1))
            m.append(float(L['temp'][sel].mean()) if sel.any() else float('nan'))
        stable = [jj for jj in range(W_FIRST, W_LAST_J + 1)
                  if all(abs(m[k + 1] - m[k]) <= W_TOL for k in range(jj, 9))]
        warm[f"{j}.{w['build']}"] = {'block_means_c': m,
                                     'warmup_s': W_STEP * stable[0] if stable else W_CAP}
    warmup_s = max(v['warmup_s'] for v in warm.values())

    # (ii) the batch size: lag-one autocorrelation of the batch means of min(D, T_max)
    rho = {}
    for j, w in read:
        c = np.minimum(calls(j, w)[-LAST:], t_max)
        rho[f"{j}.{w['build']}"] = {str(b): lag1(c.reshape(-1, b).mean(axis=1)) for b in BATCHES}
    ok = [b for b in BATCHES if all(r[str(b)] <= RHO_MAX for r in rho.values())]
    batch = ok[0] if ok else BATCHES[-1]

    A['frozen'] = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'build': build, 'warmup_s': warmup_s, 'batch': batch,
                   'selection': selection, 'warmup_by_window': warm,
                   'lag1_by_window': rho,
                   'pilot_sha256': {rel(p): sha(p) for p in (pj, pz, pc)},
                   'script_sha256': sha(os.path.abspath(__file__))}
    write_protocol(P)
    print(f"frozen: build {build}, warm-up {warmup_s} s, batch {batch}")


def rank(n, alpha):
    import margins
    return margins.rank(n, alpha)


def read_sessions(fz):
    out = {}
    for k in (1, 2, 3):
        attempts = []
        for stem in (f"s{k}", f"s{k}_a2"):
            pj = os.path.join(OUTD, stem + '.json')
            if not os.path.exists(pj):
                continue
            r = json.load(open(pj))
            assert r['mode'] == 'session' and r['session'] == k and not r['smoke']
            assert r['frozen'] == fz, f"{stem} ran under a different frozen record"
            attempts.append((stem, r))
        assert attempts, f"session {k} has no record"
        read = [(s, r) for s, r in attempts if not r['contended'] and r['failed'] is None]
        out[k] = {'attempts': [{'stem': s, 'contended': r['contended'], 'failed': r['failed']}
                               for s, r in attempts],
                  'read': read[0] if read else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    a = ap.parse_args()
    if a.freeze:
        return freeze()

    P, i = protocol()
    A = P['analyses'][i]
    fz = A['frozen']
    MG = json.load(open(os.path.join(HERE, 'margins.json')))
    grid = [float(t) for t in P['coherence_grid_us']]
    t_max = grid[-1]
    eta_lo = MG['eta_lo']['probe_model']
    proc_stored = MG['probe_processing_max_us']
    lat_gpu = json.load(open(os.path.join(HERE, 'lat_gpu.json')))
    proc_gpu = [r for r in lat_gpu['rows'] if r['component'] == 'probe_update'][0]['charge_us']
    eta_meas = min(s['methods']['sweep64']['eta'] for f in
                   ('results_rx.json', 'heldout_rx.json', 'validation_rx.json')
                   for s in json.load(open(os.path.join(HERE, f)))['sites'])
    d_s_meas = 64 * (T_AIR_US + proc_gpu)

    S = read_sessions(fz)
    read = {k: v['read'] for k, v in S.items() if v['read'] is not None}
    # (iv) the matched reference check: the sweep's per-probe processing in each session
    proc_sessions = {k: r['probe_processing']['p95_us'] for k, (_, r) in read.items()}
    proc = max([proc_stored] + list(proc_sessions.values()))
    d_s_hi = 64 * (T_AIR_US + proc)
    rhs = (1 - eta_lo) * t_max + eta_lo * d_s_hi
    b = fz['batch']
    tq = float(stats.t.ppf(1 - ALPHA / len(grid), LAST // b - 1))

    def psi(D, T, eta, ds):
        return float(np.maximum(T - D, 0).mean() - eta * max(T - ds, 0.0))

    sessions, per_build = {}, {}
    for k, (stem, r) in read.items():
        raw = np.load(os.path.join(OUTD, stem + '.npz'))
        L = gpu_log(os.path.join(OUTD, stem + '_gpu.csv'))
        rows = {}
        for w in r['windows']:
            name = w['build']
            D = raw[f"{name}.ns"] / 1e3
            assert D.size == LAST and D.size % b == 0
            Dr = raw[f"{name}.recheck.ns"] / 1e3
            rk = rank(D.size, ALPHA / 3)
            per_T = []
            for T in grid:
                pos = np.maximum(T - D, 0)
                bm = pos.reshape(-1, b).mean(axis=1)
                up = float(bm.mean() + tq * bm.std(ddof=1) / math.sqrt(bm.size))
                pa = psi(D, T, eta_lo, d_s_hi)
                pu = up - eta_lo * max(T - d_s_hi, 0.0)
                pm = psi(D, T, eta_meas, d_s_meas)
                dec = ('excluded' if pu <= 0 else 'inconclusive' if pa <= 0
                       else 'not excluded')
                per_T.append({'T_us': T, 'psi_adv_us': pa, 'psi_adv_upper_us': pu,
                              'psi_measured_us': pm, 'decision': dec,
                              'excluded_at_measured_inputs': pm <= 0,
                              'sweep_delivers_nothing': T <= d_s_hi,
                              'share_before_T': float((D < T).mean())})
            rows[name] = {
                'n': int(D.size), 'mean_us': float(D.mean()), 'median_us': float(np.median(D)),
                'p95_us': float(np.sort(D)[math.ceil(0.95 * D.size) - 1]),
                'p99_us': float(np.sort(D)[math.ceil(0.99 * D.size) - 1]),
                'min_us': float(D.min()), 'max_us': float(D.max()),
                'lag1_autocorrelation': lag1(D),
                'recheck_p95_us': float(np.sort(Dr)[math.ceil(0.95 * Dr.size) - 1]),
                'drift': w['drift'],
                'order_statistic_rank': rk, 'tau_lo_us': float(np.sort(D)[rk - 1]),
                'capped_mean_us': float(np.minimum(D, t_max).mean()),
                'share_before_dS_hi': float((D < d_s_hi).mean()),
                'share_before_dS_measured': float((D < d_s_meas).mean()),
                'per_T': per_T,
                'gpu': window_log(L, w['window_start_time'], w['window_end_time']),
                'warmup_calls': w['warmup_calls'],
                'execution_checks': w['execution_checks']}
            per_build.setdefault(name, {})[k] = D
        sessions[str(k)] = {'stem': stem, 'builds': rows,
                            'probe_processing_p95_us': proc_sessions[k]}

    opt = fz['build']
    n_read = len(read)
    outcome = []
    for t_i, T in enumerate(grid):
        d32 = [sessions[str(k)]['builds']['fp32']['per_T'][t_i]['decision'] for k in read]
        dop = [sessions[str(k)]['builds'][opt]['per_T'][t_i]['decision'] for k in read]
        if n_read < 2:
            o = 'no controlled outcome'
        elif all(x == 'excluded' for x in d32) and all(x == 'not excluded' for x in dop):
            o = 'optimization removes the exclusion'
        elif all(x == 'excluded' for x in d32) and all(x == 'excluded' for x in dop):
            o = 'optimization preserves the exclusion'
        else:
            o = 'unresolved'
        outcome.append({'T_us': T, 'outcome': o, 'fp32': dict(zip(map(str, read), d32)),
                        opt: dict(zip(map(str, read), dop)),
                        'sweep_delivers_nothing': T <= d_s_hi})

    # (vii) secondary, kept separate: the fixed-p95 margin and calculations on the recorded
    # distributions (pooled over the sessions read only where named)
    secondary = {}
    for name, by_k in per_build.items():
        tau_lo = min(sessions[str(k)]['builds'][name]['tau_lo_us'] for k in by_k)
        pooled = np.concatenate([by_k[k] for k in sorted(by_k)])

        def all_grid(eta, ds, D=pooled):
            return all(psi(D, T, eta, ds) <= 0 for T in grid)

        # the exclusion horizon: the largest T up to which psi_adv <= 0, scanned every 10 us
        mean_d, max_d = float(pooled.mean()), float(pooled.max())
        t_lin = (mean_d - eta_lo * d_s_hi) / (1 - eta_lo)
        top = max(t_lin, max_d) + 1000.0
        horizon = None
        for T in np.arange(10.0, top + 10.0, 10.0):
            if psi(pooled, T, eta_lo, d_s_hi) > 0:
                horizon = float(T - 10.0)
                break
        # the lowest eta_S that keeps the exclusion over the grid
        need = 0.0
        for T in grid:
            m = float(np.maximum(T - pooled, 0).mean())
            if T <= d_s_hi:
                need = need if m == 0 else math.inf
            else:
                need = max(need, m / (T - d_s_hi))
        # the largest per-probe airtime at which the exclusion over the grid holds
        if all_grid(eta_lo, 64 * proc):
            lo, hi = 0.0, 1000.0
            if all_grid(eta_lo, 64 * (hi + proc)):
                air = hi
            else:
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if all_grid(eta_lo, 64 * (mid + proc)) else (lo, mid)
                air = lo
        else:
            air = None
        secondary[name] = {'tau_lo_us': tau_lo, 'fixed_p95_margin_us': tau_lo - rhs,
                           'fixed_p95_excluded': tau_lo - rhs >= 0,
                           'pooled_mean_us': mean_d, 'pooled_capped_mean_us':
                           float(np.minimum(pooled, t_max).mean()),
                           'horizon_us': horizon,
                           'eta_needed': None if need > 1 else need,
                           'probe_airtime_max_us': air}

    out = {'rule': 'R22', 'script_sha256': sha(os.path.abspath(__file__)),
           'frozen': fz, 'grid_us': grid, 'T_max_us': t_max,
           'reference': {'eta_lo': eta_lo, 'probe_processing_stored_max_us': proc_stored,
                         'probe_processing_used_us': proc, 'd_S_hi_us': d_s_hi,
                         'd_S_hi_recomputed': proc > proc_stored,
                         'fixed_p95_boundary_us': rhs, 'eta_measured': eta_meas,
                         'd_S_measured_us': d_s_meas, 'probe_processing_gpu_us': proc_gpu},
           'batch': b, 'batches': LAST // b, 't_quantile': tq,
           'sessions_read': sorted(map(str, read)), 'attempts': {str(k): v['attempts']
                                                                 for k, v in S.items()},
           'sessions': sessions, 'outcome': outcome, 'secondary': secondary,
           'optimized_build': opt,
           'inputs_sha256': {rel(p): sha(p) for p in
                             [os.path.join(HERE, f) for f in
                              ('margins.json', 'lat_gpu.json', 'results_rx.json',
                               'heldout_rx.json', 'validation_rx.json')]
                             + [os.path.join(OUTD, s + e) for s, _ in read.values()
                                for e in ('.json', '.npz', '_gpu.csv')]}}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}: optimized build {opt}, sessions read {sorted(read)}")
    for o in outcome:
        print(f"  T = {o['T_us'] / 1e3:6.2f} ms  {o['outcome']}")


if __name__ == '__main__':
    main()
