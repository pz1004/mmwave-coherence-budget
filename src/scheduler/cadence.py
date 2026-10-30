#!/usr/bin/env python3
"""Rule R24: reading the cadence study (protocol.json analysis 'cadence').

  python3 cadence.py --freeze       writes the analysis's frozen record into protocol.json, once,
                                    before any session runs: the session order, the scripts, the
                                    software, the weights and the reference inputs
  python3 cadence.py                reads the 18 sessions (R24 (v)-(vii)) into cadence.json
  python3 cadence.py --check-only   recomputes cadence.json and compares, writing nothing

The candidate is the matched study's image-only ViT-B/16 (K = 0), eager FP32 or the FP16 build in
one CUDA graph, timed from the frame in pinned host memory to the beam index on the host
(../latency/profile_cadence.py) at one of three invocation spacings: back to back, or a minimum
start-to-start spacing of 10 or 50 ms.  The reference is the matched study's: the per-probe
exhaustive sweep at R22 (iv)'s adverse inputs as matched.json records them, eta_lo and d_S_hi,
on the protocol's 12-period grid with T_max = 10 ms.  For each session and period T:

  psi(T) = mean_i (T - D_i)^+ - eta_lo (T - d_S_hi)^+

over the completion times D_i of the 120 s window's calls.  This is Proposition 1(d)'s psi for a
perfect predictor under the session's empirical distribution: excluded at T if psi(T) <= 0, not
excluded if psi(T) > 0.  Each reading is exact for that distribution, conditional on the
reference inputs and the model, and says nothing about future executions.  Across sessions the
readings are descriptive.  If the fastest call is at least d_S_hi and psi(T_max) <= 0, Proposition
1(d) excludes the build at every T up to T_max for that distribution.
"""
import argparse, hashlib, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAT = os.path.join(HERE, '..', 'latency')
OUTD = os.path.join(LAT, 'profile_cadence')
MOUTD = os.path.join(LAT, 'profile_matched')
PROT = os.path.join(HERE, 'protocol.json')
OUT = os.path.join(HERE, 'cadence.json')
KEY = 'cadence'
BUILDS = ('fp32', 'fp16_graph')
SPACINGS_MS = (0, 10, 50)
CONDITIONS = [(b, ms) for b in BUILDS for ms in SPACINGS_MS]
N_BLOCKS = 3
PHASES_S = {'idle': 60, 'warmup': 60, 'window': 120}
SCRIPTS = ('../latency/profile_cadence.py', '../latency/run_profile_cadence.sh', 'cadence.py')
SOFTWARE = ('torch', 'torchvision', 'triton', 'cuda', 'cudnn', 'driver', 'python', 'numpy')
DRIFT_S, EDGE_S = 10, 30                  # block length of the drift means; first and last span


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def rel(p):
    return os.path.relpath(p, HERE)


def protocol():
    P = json.load(open(PROT))
    i = [k for k, a in enumerate(P['analyses']) if a['key'] == KEY]
    assert len(i) == 1, f"protocol.json declares '{KEY}' exactly once"
    return P, i[0]


def write_protocol(P):
    s = json.dumps(P, indent=1, sort_keys=True) + "\n"
    s.encode('ascii')
    open(PROT, 'w').write(s)


def order():
    """Three blocks, each a permutation of the six conditions."""
    from evaluate import stable_seed
    out = []
    for blk in range(1, N_BLOCKS + 1):
        for j in np.random.default_rng(stable_seed('cadence', blk)).permutation(len(CONDITIONS)):
            b, ms = CONDITIONS[int(j)]
            out.append({'slot': len(out) + 1, 'block': blk, 'build': b, 'spacing_ms': ms})
    return out


def freeze():
    P, i = protocol()
    A = P['analyses'][i]
    assert 'frozen' not in A, "the frozen record is written once"
    for k in range(1, len(CONDITIONS) * N_BLOCKS + 1):
        for x in ('', '_a2'):
            assert not os.path.exists(os.path.join(OUTD, f"c{k:02d}{x}.json")), \
                "a session ran before the freeze"
    MT = json.load(open(os.path.join(HERE, 'matched.json')))
    s1 = json.load(open(os.path.join(MOUTD, MT['sessions']['1']['stem'] + '.json')))
    wts = {w['build']: w['weights_sha256'] for w in s1['windows']}
    assert wts['fp32'] == s1['weights_fp32_sha256'] and set(BUILDS) <= set(wts)
    A['frozen'] = {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'order': order(),
        'order_rule': "three blocks; block k is the six conditions [(b, ms) for b in ('fp32', "
                      "'fp16_graph') for ms in (0, 10, 50)] permuted by numpy.random."
                      "default_rng(evaluate.stable_seed('cadence', k)).permutation(6)",
        'builds': list(BUILDS), 'spacings_ms': list(SPACINGS_MS), 'phases_s': PHASES_S,
        'software': {k: s1['env'][k] for k in SOFTWARE},
        'software_source': rel(os.path.join(MOUTD, MT['sessions']['1']['stem'] + '.json')),
        'weights_sha256': {b: wts[b] for b in BUILDS},
        'reference': {k: MT['reference'][k] for k in
                      ('eta_lo', 'd_S_hi_us', 'eta_measured', 'd_S_measured_us')},
        'grid_us': [float(t) for t in P['coherence_grid_us']],
        'scripts_sha256': {s: sha(os.path.join(HERE, s)) for s in SCRIPTS}}
    write_protocol(P)
    print("frozen: " + ", ".join(f"{o['slot']}:{o['build']}@{o['spacing_ms']}"
                                 for o in A['frozen']['order']))


def psi(D, T, eta, ds):
    """Proposition 1(d)'s psi for a perfect predictor under the empirical distribution of D."""
    return float(np.maximum(T - D, 0).mean() - eta * max(T - ds, 0.0))


def lag1(x):
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def p_rank(x, q):
    return float(np.sort(x)[math.ceil(q * x.size) - 1])


def gate_matched(ref, grid):
    """psi as cadence.py computes it, on R22's six session-build records, against matched.json."""
    MT = json.load(open(os.path.join(HERE, 'matched.json')))
    worst = 0.0
    for k, ses in MT['sessions'].items():
        raw = np.load(os.path.join(MOUTD, ses['stem'] + '.npz'))
        for b, row in ses['builds'].items():
            D = raw[f"{b}.ns"] / 1e3
            for t, r in zip(grid, row['per_T']):
                assert r['T_us'] == t
                worst = max(worst, abs(psi(D, t, ref['eta_lo'], ref['d_S_hi_us'])
                                       - r['psi_adv_us']))
    return {'psi_vs_matched_max_abs_us': worst, 'holds': worst <= 1e-9}


def read_session(rec, raw, L, ref, grid, ms):
    """One session's readings, R24 (v) and (vii)."""
    import matched
    w = rec['runs'][0]
    s, e = raw['start_ns'], raw['end_ns']
    w0 = int(raw['window_from_ns'])
    win = s >= w0
    D = (e[win] - s[win]) / 1e3                                   # us
    S = s[win] / 1e9                                              # s from the warm-up's start
    t_max, eta, ds = grid[-1], ref['eta_lo'], ref['d_S_hi_us']
    fastest = float(D.min())
    per_T = []
    for T in grid:
        p = psi(D, T, eta, ds)
        pm = psi(D, T, ref['eta_measured'], ref['d_S_measured_us'])
        per_T.append({'T_us': T, 'psi_us': p, 'excluded': p <= 0,
                      'sweep_delivers_nothing': T <= ds,
                      'neither_delivers': T <= min(ds, fastest),
                      'share_before_T': float((D < T).mean()),
                      'psi_measured_us': pm, 'excluded_at_measured_inputs': pm <= 0})
    gap = np.diff(s[win]) / 1e3
    t_win, span = w0 / 1e9, rec['protocol']['window_s']
    blocks = [D[(S >= t_win + DRIFT_S * j) & (S < t_win + DRIFT_S * (j + 1))]
              for j in range(math.ceil(span / DRIFT_S))]
    bmeans = [float(x.mean()) for x in blocks if x.size]
    first = D[S < t_win + EDGE_S]
    last = D[S >= t_win + span - EDGE_S]
    delta_us = ms * 1e3
    return {
        'n': int(D.size), 'mean_us': float(D.mean()), 'median_us': float(np.median(D)),
        'p95_us': p_rank(D, 0.95), 'p99_us': p_rank(D, 0.99), 'min_us': fastest,
        'max_us': float(D.max()), 'lag1_autocorrelation': lag1(D),
        'share_longer_than_spacing': float((D > delta_us).mean()) if ms > 0 else None,
        'start_to_start_us': {'median': float(np.median(gap)), 'p95': p_rank(gap, 0.95),
                              'min': float(gap.min()), 'max': float(gap.max())},
        'drift': {'block_means_us': bmeans,
                  'block_mean_range_rel': (max(bmeans) - min(bmeans)) / min(bmeans),
                  'first_30s_mean_us': float(first.mean()), 'last_30s_mean_us': float(last.mean()),
                  'last_vs_first_rel': float(last.mean() / first.mean() - 1)},
        'psi_Tmax_us': per_T[-1]['psi_us'],
        'share_before_dS_hi': float((D < ds).mean()),
        'excluded_every_period': all(r['excluded'] for r in per_T),
        'not_excluded_at_T_us': [r['T_us'] for r in per_T if not r['excluded']],
        'excluded_delivering': sum(r['excluded'] for r in per_T
                                   if not r['sweep_delivers_nothing']),
        'n_delivering': sum(not r['sweep_delivers_nothing'] for r in per_T),
        'all_periods_up_to_Tmax': bool(fastest >= ds and per_T[-1]['psi_us'] <= 0),
        'per_T': per_T,
        'gpu_window': matched.window_log(L, w['window_start_time'], w['window_end_time']),
        'gpu_idle': matched.window_log(L, w['idle_start_time'], w['idle_end_time']),
        'temp_at_warmup_start_c': (lambda m: float(L['temp'][m][-1]) if m.any() else None)(
            L['timestamp'] <= w['warmup_start_time']),
        'execution_checks_pass': bool(w['execution_checks']['passes']),
        'new_compute_apps': rec['new_compute_apps'],
        'loadavg_before': w['state_before']['loadavg'],
        'loadavg_after': w['state_after']['loadavg']}


def condition_label(rows):
    """R24 (vi), per build and spacing over its sessions."""
    if all(r['excluded_every_period'] for r in rows):
        return 'excluded at every period in every session'
    if all(not r['excluded_every_period'] for r in rows):
        return 'not excluded at some period in every session'
    return 'the sessions differ'


def build_label(labels):
    """R24 (vi), per build over the spacings."""
    every = 'excluded at every period in every session'
    none = 'not excluded at some period in every session'
    if all(x == every for x in labels):
        return 'the exclusion persists across the tested spacings'
    if any(x == every for x in labels) and any(x == none for x in labels):
        return 'the exclusion depends on the spacing'
    if all(x == none for x in labels):
        return 'not excluded at any tested spacing'
    return 'the verdict varies with the sessions'


def compute():
    import matched
    P, i = protocol()
    A = P['analyses'][i]
    fz = A['frozen']
    MT = json.load(open(os.path.join(HERE, 'matched.json')))
    ref = fz['reference']
    assert all(ref[k] == MT['reference'][k] for k in ref), "matched.json's reference changed"
    grid = [float(t) for t in P['coherence_grid_us']]
    assert grid == fz['grid_us']
    gate = {'matched_psi': gate_matched(ref, grid)}
    assert gate['matched_psi']['holds'], gate
    fz_sha = hashlib.sha256(json.dumps(fz, sort_keys=True).encode()).hexdigest()
    sessions, attempts, inputs = {}, {}, {}
    for o in fz['order']:
        k = o['slot']
        att = []
        for stem in (f"c{k:02d}", f"c{k:02d}_a2"):
            pj = os.path.join(OUTD, stem + '.json')
            if not os.path.exists(pj):
                continue
            r = json.load(open(pj))
            assert r['mode'] == 'session' and r['slot'] == k and not r['smoke'], stem
            assert r['frozen'] == fz and r['frozen_sha256'] == fz_sha, \
                f"{stem} ran under another frozen record"
            assert r['env']['script_sha256'] == fz['scripts_sha256'][SCRIPTS[0]], stem
            assert all((w['build'], w['spacing_ms']) == (o['build'], o['spacing_ms'])
                       for w in r['runs']), stem
            att.append((stem, r))
        attempts[str(k)] = [{'stem': s, 'failed': r['failed']} for s, r in att]
        read = [(s, r) for s, r in att if r['failed'] is None]
        if not read:
            continue
        stem, r = read[0]
        files = [os.path.join(OUTD, stem + x) for x in ('.json', '.npz', '_gpu.csv')]
        inputs.update({rel(p): sha(p) for p in files})
        raw = np.load(files[1])
        L = matched.gpu_log(files[2])
        row = read_session(r, raw, L, ref, grid, o['spacing_ms'])
        sessions[str(k)] = {'stem': stem, 'block': o['block'], 'build': o['build'],
                            'spacing_ms': o['spacing_ms'], **row}

    conditions = {}
    for b, ms in CONDITIONS:
        ks = [str(o['slot']) for o in fz['order']
              if (o['build'], o['spacing_ms']) == (b, ms) and str(o['slot']) in sessions]
        rows = [sessions[k] for k in ks]
        if not rows:
            conditions[f"{b}|{ms}"] = {'sessions': [], 'label': 'no session read'}
            continue
        rng = lambda f: [min(r[f] for r in rows), max(r[f] for r in rows)]
        conditions[f"{b}|{ms}"] = {
            'build': b, 'spacing_ms': ms, 'sessions': ks,
            'mean_us': rng('mean_us'), 'p95_us': rng('p95_us'), 'min_us': rng('min_us'),
            'psi_Tmax_us': rng('psi_Tmax_us'),
            'excluded_delivering': [r['excluded_delivering'] for r in rows],
            'not_excluded_at_T_us': {k: sessions[k]['not_excluded_at_T_us'] for k in ks},
            'all_periods_up_to_Tmax': [r['all_periods_up_to_Tmax'] for r in rows],
            'share_longer_than_spacing': ([r['share_longer_than_spacing'] for r in rows]
                                          if ms > 0 else None),
            'temp_c': [min(r['gpu_window']['temp_c'][0] for r in rows),
                       max(r['gpu_window']['temp_c'][1] for r in rows)],
            'clock_sm_mhz': [min(r['gpu_window']['clock_sm_mhz'][0] for r in rows),
                             max(r['gpu_window']['clock_sm_mhz'][2] for r in rows)],
            'label': condition_label(rows),
            'complete': len(rows) == N_BLOCKS}
    builds = {}
    for b in BUILDS:
        labs = [conditions[f"{b}|{ms}"]['label'] for ms in SPACINGS_MS]
        r22 = [MT['sessions'][k]['builds'][b] for k in MT['sessions']]
        means, psis = [r['mean_us'] for r in r22], [r['per_T'][-1]['psi_adv_us'] for r in r22]
        builds[b] = {'label': build_label(labs),
                     'by_spacing': dict(zip(map(str, SPACINGS_MS), labs)),
                     'needs_accuracy_evaluation': {
                         k: s['not_excluded_at_T_us'] for k, s in sessions.items()
                         if s['build'] == b and s['not_excluded_at_T_us']},
                     'matched_R22': {'mean_us': [min(means), max(means)],
                                     'psi_Tmax_us': [min(psis), max(psis)]}}
    inputs.update({rel(os.path.join(HERE, 'matched.json')):
                   sha(os.path.join(HERE, 'matched.json'))})
    return {'rule': 'R24', 'script_sha256': sha(os.path.abspath(__file__)),
            'script_as_frozen': sha(os.path.abspath(__file__)) ==
            fz['scripts_sha256']['cadence.py'],
            'runner_as_frozen': sha(os.path.join(HERE, SCRIPTS[1])) ==
            fz['scripts_sha256'][SCRIPTS[1]],
            'frozen': fz, 'grid_us': grid, 'T_max_us': grid[-1], 'reference': ref,
            'gate': gate, 'attempts': attempts, 'sessions_read': sorted(sessions, key=int),
            'complete': len(sessions) == len(fz['order']),
            'sessions': sessions, 'conditions': conditions, 'builds': builds,
            'inputs_sha256': inputs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.freeze:
        return freeze()
    out = compute()
    if a.check_only:
        old = json.load(open(OUT))
        same = json.loads(json.dumps(out)) == old
        print(f"cadence.json {'reproduces' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    assert not os.path.exists(OUT), f"{OUT} exists: written once (use --check-only)"
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}: {len(out['sessions'])} of {len(out['frozen']['order'])} sessions read")
    for b, v in out['builds'].items():
        print(f"  {b}: {v['label']}  {v['by_spacing']}")


if __name__ == '__main__':
    main()
