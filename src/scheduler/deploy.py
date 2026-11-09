#!/usr/bin/env python3
"""Rule R38: reading the deployment-runtime study (protocol.json analysis 'deploy'), matched.py's
rules applied to each group of ../latency/profile_deploy.py.

  python3 deploy.py --freeze   applies R38 (ii) to the three pilots (profile_deploy/pilot_<G>.*)
                               and writes the analysis's frozen record into protocol.json, once,
                               before any main session runs
  python3 deploy.py            reads the main sessions (R38 (iii)-(v)) into deploy.json

Groups: rn_gpu (ResNet-18, GPU), vit_gpu (ViT-B/16, GPU), rn_cpu (ResNet-18, eight P-core threads),
each an FP32 comparator and candidates (profile_deploy.GROUPS), timed from the frame to the beam
index on the host.  Per group:
  (ii) the optimized build: among the candidates that built, ran, passed the execution checks, made
       at least 10,000 calls and, if compiled, compiled to one graph without breaks, the lowest
       mean over the window's last 10,000 calls; the warm-up: matched.py's rule (the GPU's
       temperature in 30 s blocks of the FP32 windows from idle and from hot and the chosen
       build's) for the GPU groups, 60 s for the CPU group; the batch size: matched.py's rule (the
       smallest of 100, 200, 500, 1,000 calls whose batch means of min(D, T_max) have a lag-one
       autocorrelation of at most 0.10 in every window read).
  (iii) per session read (the first attempt that is neither contended nor failed), build and period
       T of the protocol's grid, matched.py's expected-rate reading against the per-probe sweep at
       adverse inputs, d_S_hi = 64 (20 us + p), p the larger of margins.json's stored per-probe
       processing and the sessions' (timed in group rn_gpu's processes), and eta_lo from
       margins.json: psi_adv(T) = mean_i (T - D_i)^+ - eta_lo (T - d_S_hi)^+, with its one-sided
       batch-means upper bound at 1 - 0.05/12; excluded if the bound is at most zero.
  (iv) per group and T, the outcome over the sessions read: the optimization removes the exclusion
       (FP32 excluded and the build not excluded in every session), preserves it, or neither.
  (v) per build, the p95 of each session read, their least and largest, the pooled mean, matched.py's
       secondary quantities; make_draft.py screens each frozen build at its fastest session's p95
       against every reference of Table IV, as Table V's needs are computed.
"""
import argparse, json, math, os, sys, time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matched as MT

LAT = os.path.join(HERE, '..', 'latency')
OUTD = os.path.join(LAT, 'profile_deploy')
PROT = os.path.join(HERE, 'protocol.json')
OUT = os.path.join(HERE, 'deploy.json')
KEY = 'deploy'
GROUPS = {'rn_gpu': ('fp16_eager', 'fp16_graph', 'fp16_cmax'),
          'vit_gpu': ('fp16_graph', 'fp16_cmax'),
          'rn_cpu': ('cpu_compile', 'int8')}
COMPILED = ('fp16_cmax', 'cpu_compile')
CPU_WARMUP_S = 60
LAST, ALPHA, BATCHES, RHO_MAX, T_AIR_US = MT.LAST, MT.ALPHA, MT.BATCHES, MT.RHO_MAX, MT.T_AIR_US
W_STEP, W_FIRST, W_LAST_J, W_CAP, W_TOL = MT.W_STEP, MT.W_FIRST, MT.W_LAST_J, MT.W_CAP, MT.W_TOL


def write_protocol(P):
    """The protocol file as it is serialized now (indent 1, keys in their order), ASCII only."""
    s = json.dumps(P, indent=1) + "\n"
    s.encode('ascii')
    open(PROT, 'w').write(s)


def protocol():
    P = json.load(open(PROT))
    i = [k for k, a in enumerate(P['analyses']) if a['key'] == KEY]
    assert len(i) == 1, f"protocol.json declares '{KEY}' exactly once"
    return P, i[0]


def freeze():
    P, i = protocol()
    A = P['analyses'][i]
    assert 'frozen' not in A, "the frozen record is written once"
    t_max = P['coherence_grid_us'][-1]
    fz = {}
    for g, cands in GROUPS.items():
        pj, pz, pc = (os.path.join(OUTD, f"pilot_{g}{e}") for e in ('.json', '.npz', '_gpu.csv'))
        rec, raw, L = json.load(open(pj)), np.load(pz), MT.gpu_log(pc)
        assert rec['mode'] == 'pilot' and not rec['smoke'] and rec['failed'] is None
        W = rec['windows']
        assert [w['build'] for w in W] == ['fp32', *cands, 'fp32']
        calls = lambda j, w: raw[f"{j}.{w['build']}.ns"] / 1e3
        selection = {}
        for j, w in enumerate(W):
            if w['build'] not in cands:
                continue
            s = {'built': w['built'], 'runs': w.get('runs', False),
                 'execution_checks_pass': bool(w.get('execution_checks', {}).get('passes'))}
            if w['build'] in COMPILED and w['built']:
                dc = w.get('dynamo_counters', {})
                s['graph_breaks'] = sum(dc.get('graph_break', {}).values())
                s['unique_graphs'] = dc.get('stats', {}).get('unique_graphs')
                s['single_graph'] = s['graph_breaks'] == 0 and s['unique_graphs'] == 1
            s['eligible'] = (s['built'] and s['runs'] and s['execution_checks_pass']
                             and s.get('single_graph', True) and w.get('n_calls', 0) >= LAST)
            if s['built'] and s['runs']:
                s['mean_us_last_10000'] = float(calls(j, w)[-LAST:].mean())
            selection[w['build']] = s
        eligible = [b for b, s in selection.items() if s['eligible']]
        if not eligible:
            fz[g] = {'build': None, 'selection': selection,
                     'note': 'no candidate is eligible: the group reports FP32 only'}
            continue
        build = min(eligible, key=lambda b: selection[b]['mean_us_last_10000'])
        last = len(W) - 1
        read = [(0, W[0]), (last, W[last])] + [(j, w) for j, w in enumerate(W)
                                              if w['build'] == build]
        warm = {}
        if g == 'rn_cpu':
            warmup_s = CPU_WARMUP_S
        else:
            for j, w in read:
                t0 = w['window_start_time']
                m = []
                for kk in range(10):
                    sel = (L['timestamp'] >= t0 + W_STEP * kk) & (L['timestamp'] < t0 + W_STEP * (kk + 1))
                    m.append(float(L['temp'][sel].mean()) if sel.any() else float('nan'))
                stable = [jj for jj in range(W_FIRST, W_LAST_J + 1)
                          if all(abs(m[kk + 1] - m[kk]) <= W_TOL for kk in range(jj, 9))]
                warm[f"{j}.{w['build']}"] = {'block_means_c': m,
                                             'warmup_s': W_STEP * stable[0] if stable else W_CAP}
            warmup_s = max(v['warmup_s'] for v in warm.values())
        rho = {}
        for j, w in read:
            c = np.minimum(calls(j, w)[-LAST:], t_max)
            rho[f"{j}.{w['build']}"] = {str(b): MT.lag1(c.reshape(-1, b).mean(axis=1))
                                        for b in BATCHES}
        ok = [b for b in BATCHES if all(r[str(b)] <= RHO_MAX for r in rho.values())]
        fz[g] = {'build': build, 'warmup_s': warmup_s, 'batch': ok[0] if ok else BATCHES[-1],
                 'selection': selection, 'warmup_by_window': warm, 'lag1_by_window': rho,
                 'pilot_sha256': {MT.rel(p): MT.sha(p) for p in (pj, pz, pc)}}
    fz['_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    fz['_script_sha256'] = MT.sha(os.path.abspath(__file__))
    A['frozen'] = fz
    write_protocol(P)
    for g in GROUPS:
        f = fz[g]
        print(f"frozen {g}: build {f['build']}, warm-up {f.get('warmup_s')} s, batch {f.get('batch')}")


def read_sessions(g, fz):
    out = {}
    for k in (1, 2, 3):
        attempts = []
        for stem in (f"s{k}_{g}", f"s{k}_{g}_a2"):
            pj = os.path.join(OUTD, stem + '.json')
            if not os.path.exists(pj):
                continue
            r = json.load(open(pj))
            assert r['mode'] == 'session' and r['session'] == k and not r['smoke']
            assert r['frozen'] == fz, f"{stem} ran under a different frozen record"
            attempts.append((stem, r))
        assert attempts, f"session {k} of {g} has no record"
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
    FZ = A['frozen']
    assert A['scripts_sha256']['deploy.py'] == MT.sha(os.path.abspath(__file__)), \
        "deploy.py is not the declared script"
    MG = json.load(open(os.path.join(HERE, 'margins.json')))
    grid = [float(t) for t in P['coherence_grid_us']]
    t_max = grid[-1]
    eta_lo = MG['eta_lo']['probe_model']
    proc_stored = MG['probe_processing_max_us']
    S = {g: read_sessions(g, FZ[g]) for g in GROUPS if FZ[g]['build']}
    proc_sessions = {}
    for k, v in S.get('rn_gpu', {}).items():
        if v['read']:
            proc_sessions[k] = v['read'][1]['probe_processing']['p95_us']
    proc = max([proc_stored] + list(proc_sessions.values()))
    d_s_hi = 64 * (T_AIR_US + proc)
    rhs = (1 - eta_lo) * t_max + eta_lo * d_s_hi

    def psi(D, T, eta, ds):
        return float(np.maximum(T - D, 0).mean() - eta * max(T - ds, 0.0))

    out = {'rule': 'R38', 'script_sha256': MT.sha(os.path.abspath(__file__)), 'frozen': FZ,
           'grid_us': grid, 'T_max_us': t_max,
           'reference': {'eta_lo': eta_lo, 'probe_processing_stored_max_us': proc_stored,
                         'probe_processing_sessions_us': {str(k): v for k, v in
                                                          proc_sessions.items()},
                         'probe_processing_used_us': proc, 'd_S_hi_us': d_s_hi,
                         'fixed_p95_boundary_us': rhs},
           'groups': {}, 'inputs_sha256': {MT.rel(os.path.join(HERE, 'margins.json')):
                                           MT.sha(os.path.join(HERE, 'margins.json'))}}
    for g, Sg in S.items():
        fz = FZ[g]
        b = fz['batch']
        tq = float(stats.t.ppf(1 - ALPHA / len(grid), LAST // b - 1))
        read = {k: v['read'] for k, v in Sg.items() if v['read'] is not None}
        sessions, per_build = {}, {}
        for k, (stem, r) in read.items():
            raw = np.load(os.path.join(OUTD, stem + '.npz'))
            L = MT.gpu_log(os.path.join(OUTD, stem + '_gpu.csv'))
            rows = {}
            for w in r['windows']:
                name = w['build']
                D = raw[f"{name}.ns"] / 1e3
                assert D.size == LAST and D.size % b == 0
                Dr = raw[f"{name}.recheck.ns"] / 1e3
                rk = MT.rank(D.size, ALPHA / 3)
                per_T = []
                for T in grid:
                    bm = np.maximum(T - D, 0).reshape(-1, b).mean(axis=1)
                    up = float(bm.mean() + tq * bm.std(ddof=1) / math.sqrt(bm.size))
                    pa = psi(D, T, eta_lo, d_s_hi)
                    pu = up - eta_lo * max(T - d_s_hi, 0.0)
                    dec = ('excluded' if pu <= 0 else 'inconclusive' if pa <= 0
                           else 'not excluded')
                    per_T.append({'T_us': T, 'psi_adv_us': pa, 'psi_adv_upper_us': pu,
                                  'decision': dec, 'sweep_delivers_nothing': T <= d_s_hi})
                srt = np.sort(D)
                rows[name] = {
                    'n': int(D.size), 'mean_us': float(D.mean()), 'median_us': float(np.median(D)),
                    'p95_us': float(srt[math.ceil(0.95 * D.size) - 1]),
                    'p99_us': float(srt[math.ceil(0.99 * D.size) - 1]),
                    'min_us': float(D.min()), 'max_us': float(D.max()),
                    'lag1_autocorrelation': MT.lag1(D),
                    'recheck_p95_us': float(np.sort(Dr)[math.ceil(0.95 * Dr.size) - 1]),
                    'drift': w['drift'], 'order_statistic_rank': rk,
                    'tau_lo_us': float(srt[rk - 1]),
                    'per_T': per_T, 'gpu': MT.window_log(L, w['window_start_time'],
                                                         w['window_end_time']),
                    'warmup_calls': w['warmup_calls'], 'execution_checks': w['execution_checks']}
                per_build.setdefault(name, {})[k] = D
            sessions[str(k)] = {'stem': stem, 'builds': rows}
        opt = fz['build']
        outcome = []
        for t_i, T in enumerate(grid):
            d32 = [sessions[str(k)]['builds']['fp32']['per_T'][t_i]['decision'] for k in read]
            dop = [sessions[str(k)]['builds'][opt]['per_T'][t_i]['decision'] for k in read]
            if len(read) < 2:
                o = 'no controlled outcome'
            elif all(x == 'excluded' for x in d32) and all(x == 'not excluded' for x in dop):
                o = 'optimization removes the exclusion'
            elif all(x == 'excluded' for x in d32) and all(x == 'excluded' for x in dop):
                o = 'optimization preserves the exclusion'
            else:
                o = 'unresolved'
            outcome.append({'T_us': T, 'outcome': o, 'fp32': dict(zip(map(str, read), d32)),
                            opt: dict(zip(map(str, read), dop))})
        summary = {}
        for name, by_k in per_build.items():
            p95 = {str(k): sessions[str(k)]['builds'][name]['p95_us'] for k in by_k}
            pooled = np.concatenate([by_k[k] for k in sorted(by_k)])
            need = 0.0
            for T in grid:
                m = float(np.maximum(T - pooled, 0).mean())
                if T <= d_s_hi:
                    need = need if m == 0 else math.inf
                else:
                    need = max(need, m / (T - d_s_hi))
            tau_lo = min(sessions[str(k)]['builds'][name]['tau_lo_us'] for k in by_k)
            summary[name] = {'p95_us': p95, 'p95_min_us': min(p95.values()),
                             'p95_max_us': max(p95.values()), 'pooled_mean_us': float(pooled.mean()),
                             'pooled_min_us': float(pooled.min()), 'tau_lo_us': tau_lo,
                             'fixed_p95_margin_us': tau_lo - rhs,
                             'eta_needed_per_probe_adverse': None if need > 1 else need}
        out['groups'][g] = {'optimized_build': opt, 'batch': b, 't_quantile': tq,
                            'sessions_read': sorted(map(str, read)),
                            'attempts': {str(k): v['attempts'] for k, v in Sg.items()},
                            'sessions': sessions, 'outcome': outcome, 'summary': summary}
        out['inputs_sha256'].update({MT.rel(os.path.join(OUTD, s + e)): MT.sha(
            os.path.join(OUTD, s + e)) for s, _ in read.values() for e in ('.json', '.npz',
                                                                         '_gpu.csv')})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")
    for g, v in out['groups'].items():
        print(f"  {g}: {v['optimized_build']}, sessions {v['sessions_read']}")
        for name, s in v['summary'].items():
            print(f"    {name}: p95 {s['p95_min_us']:.0f}-{s['p95_max_us']:.0f} us")


if __name__ == '__main__':
    main()
