#!/usr/bin/env python3
"""Rule R23: the matched study's readings restated, and the calculations added after its outputs
(protocol.json analysis 'matched_post', declared before this script ran; R23 itself was set after
the matched study's outputs and a review had been seen).

Nothing is timed and no stored file changes.  For each session rule R22 read and each build, from
the session's 10,000 timed completion times D_i (../latency/profile_matched/<stem>.npz):

  psi(T; eta, d, a) = mean_i (T - D_i / a)^+ - eta (T - d)^+

is Proposition 1(d)'s psi for a perfect predictor under the empirical distribution of the D_i, each
divided by a.  At a = 1 and the adverse inputs (eta_lo, d_S_hi) it is R22's psi_adv, exact for that
distribution; at the measured inputs (the lowest measured efficiency, 64 t_p) it is R22's
descriptive reading.  Per session read and build the script computes:

  - R22 (vii)'s three calculations, which R22 makes on the pooled record only: the exclusion horizon
    (the largest T up to which psi_adv <= 0, scanned every 10 us as matched.py scans, and in closed
    form, (mean - eta d) / (1 - eta), where no call finishes before d and every call before it), the
    lowest eta_S that keeps the exclusion over the grid and the largest per-probe airtime at which
    it holds;
  - the uniform scaling factor, at the adverse and at the measured inputs: the critical factor, the
    infimum of the a for which psi(T; ., ., a) > 0 at some period of the grid, every timed call
    (copy, normalization, network, argmax and return) divided by a; the period at which it turns;
    and, where every call divided by the critical factor finishes by T_max and after d, the closed
    form mean / ((1 - eta) T_max + eta d): psi is then zero up to d and convex beyond, so T_max
    turns first.  psi does not decrease in a, so every larger factor removes the exclusion and
    bisection finds the critical factor;
  - the infimum of the a for which R22's own procedure, applied to the D_i / a, does not read
    'excluded' at some period at which the sweep delivers (its batch-means bound above zero
    there).  Its 'not excluded' begins above the critical factor, where psi itself is positive.

Every factor holds the regime, the reference and the other inputs fixed; at the critical factor
itself psi is zero at the period that turns, so the exclusion still holds there, weakly.  The value
recorded is the bisection's upper end, at which it no longer holds.

Gates:
  G1_pins         matched.json and the session records are those protocol.json records (analysis
                  'matched'), matched.json was written by the matched.py recorded there, and
                  margins.json is the one matched.json read;
  G2_reproduced   at a = 1, every psi_adv, bound, decision and measured-input reading equals
                  matched.json's, and the pooled horizon, efficiency and airtime equal its
                  secondary ones;
  G3_closed_form  where its conditions hold, each closed form agrees with the scan or bisection;
  G4_monotone     on a scan of a in steps of 0.001, R22's classification leaves 'excluded' once.

usage:  python3 matched_post.py [--check-only]      -> matched_post.json (written once)
"""
import hashlib, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protocol

OUTD = os.path.join(HERE, '..', 'latency', 'profile_matched')
OUT = os.path.join(HERE, 'matched_post.json')
J = lambda *p: json.load(open(os.path.join(*p)))
T_AIR_US = 20.0                    # matched.py's per-probe airtime
STEP = 0.001                       # G4's scan of a
TOL = 1e-9


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def psi(D, T, eta, d, a=1.0):
    return float(np.maximum(T - D / a, 0).mean() - eta * max(T - d, 0.0))


def bound(D, T, eta, d, b, tq, a=1.0):
    """R22 (v)'s one-sided batch-means upper bound on psi(T), the D_i divided by a."""
    bm = np.maximum(T - D / a, 0).reshape(-1, b).mean(axis=1)
    return float(bm.mean() + tq * bm.std(ddof=1) / math.sqrt(bm.size)) - eta * max(T - d, 0.0)


def first_turn(pred, hi=2.0):
    """The infimum of the a >= 1 at which pred holds, pred false at 1 and monotone in a, by
    bisection to float precision; the value returned is the bracket's upper end, at which pred
    holds."""
    assert not pred(1.0)
    while not pred(hi):
        hi *= 2
        assert hi < 1e3
    lo = 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return hi


def vii(D, grid, eta, d, proc):
    """R22 (vii)'s horizon, lowest efficiency and largest airtime, as matched.py computes them."""
    mean_d, max_d = float(D.mean()), float(D.max())
    t_lin = (mean_d - eta * d) / (1 - eta)
    horizon = None
    for T in np.arange(10.0, max(t_lin, max_d) + 1000.0 + 10.0, 10.0):
        if psi(D, T, eta, d) > 0:
            horizon = float(T - 10.0)
            break
    need = 0.0
    for T in grid:
        m = float(np.maximum(T - D, 0).mean())
        if T <= d:
            need = need if m == 0 else math.inf
        else:
            need = max(need, m / (T - d))
    all_grid = lambda ds: all(psi(D, T, eta, ds) <= 0 for T in grid)
    if all_grid(64 * proc):
        lo, hi = 0.0, 1000.0
        if all_grid(64 * (hi + proc)):
            air = hi
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if all_grid(64 * (mid + proc)) else (lo, mid)
            air = lo
    else:
        air = None
    closed = t_lin if float(D.min()) > d and t_lin >= max_d else None
    return {'horizon_us': horizon, 'horizon_closed_us': closed,
            'eta_needed': None if need > 1 else need, 'probe_airtime_max_us': air}


def scaling(D, grid, t_max, eta, d):
    """The critical factor: the infimum of the uniform scaling factors a for which psi is positive
    at some period of the grid."""
    a = first_turn(lambda a: max(psi(D, T, eta, d, a) for T in grid) > 0)
    at = max(grid, key=lambda T: psi(D, T, eta, d, a))
    within, after = float(D.max()) / a <= t_max, float(D.min()) / a > d
    cf = float(D.mean()) / ((1 - eta) * t_max + eta * d) if within and after else None
    return {'factor': a, 'turns_at_us': at, 'scaled_calls_within_T_max': within,
            'scaled_calls_after_d_S': after, 'closed_form': cf}


def r22_leaves(D, grid, eta, d, b, tq):
    """The infimum of the a for which R22's procedure, on the D_i / a, does not read 'excluded' at
    some period at which the sweep delivers, and G4's scan."""
    contest = [T for T in grid if T > d]
    excluded = lambda a: all(bound(D, T, eta, d, b, tq, a) <= 0 for T in contest)
    a = first_turn(lambda a: not excluded(a))
    at = max(contest, key=lambda T: bound(D, T, eta, d, b, tq, a))
    flags = [excluded(x) for x in np.arange(1.0, a + 0.05, STEP)]
    k = flags.index(False)
    monotone = all(flags[:k]) and not any(flags[k:]) and 1.0 + (k - 1) * STEP < a <= 1.0 + k * STEP
    return {'factor': a, 'turns_at_us': at,
            'decision': 'inconclusive' if psi(D, at, eta, d, a) <= 0 else 'not excluded'}, monotone


def main():
    check = '--check-only' in sys.argv
    assert check or not os.path.exists(OUT), "matched_post.json exists: it is not rewritten"
    MA = protocol.analysis('matched')
    MT = J(HERE, 'matched.json')
    MG = J(HERE, 'margins.json')
    gates = {'G1_pins': True, 'G2_reproduced': True, 'G3_closed_form': True, 'G4_monotone': True}

    # ---- G1: the stored study, as recorded and as matched.py wrote it ------------------------------
    for f, h in MA['files_sha256'].items():
        gates['G1_pins'] &= sha(os.path.join(HERE, f)) == h
    gates['G1_pins'] &= MT['script_sha256'] == MA['scripts_sha256']['matched.py']
    gates['G1_pins'] &= sha(os.path.join(HERE, 'margins.json')) == MT['inputs_sha256']['margins.json']

    ref = MT['reference']
    grid, t_max = [float(t) for t in MT['grid_us']], float(MT['T_max_us'])
    eta, d, proc = ref['eta_lo'], ref['d_S_hi_us'], ref['probe_processing_used_us']
    eta_m, d_m = ref['eta_measured'], ref['d_S_measured_us']
    assert eta == MG['eta_lo']['probe_model'] and d == 64 * (T_AIR_US + proc)
    b, tq = MT['batch'], MT['t_quantile']
    assert b == MT['frozen']['batch'] and MT['batches'] == 10_000 // b
    opt = MT['optimized_build']
    builds = ('fp32', opt)

    sessions, Ds = {}, {}
    for k in MT['sessions_read']:
        stem = MT['sessions'][k]['stem']
        raw = np.load(os.path.join(OUTD, stem + '.npz'))
        rows = {}
        for name in builds:
            D = raw[f"{name}.ns"] / 1e3
            S = MT['sessions'][k]['builds'][name]
            # ---- G2: R22's readings, reproduced at a = 1 -----------------------------------------
            gates['G2_reproduced'] &= D.size == S['n'] and abs(float(D.mean()) - S['mean_us']) < TOL
            for T, x in zip(grid, S['per_T']):
                pa, pu = psi(D, T, eta, d), bound(D, T, eta, d, b, tq)
                dec = 'excluded' if pu <= 0 else 'inconclusive' if pa <= 0 else 'not excluded'
                gates['G2_reproduced'] &= (abs(pa - x['psi_adv_us']) < TOL
                                           and abs(pu - x['psi_adv_upper_us']) < TOL
                                           and dec == x['decision']
                                           and abs(psi(D, T, eta_m, d_m) - x['psi_measured_us']) < TOL)
            r = {'n': int(D.size), 'mean_us': float(D.mean()), 'min_us': float(D.min()),
                 'max_us': float(D.max())}
            r.update(vii(D, grid, eta, d, proc))
            r['scaling_adverse'] = scaling(D, grid, t_max, eta, d)
            r['scaling_measured'] = scaling(D, grid, t_max, eta_m, d_m)
            r['r22_leaves_excluded'], mono = r22_leaves(D, grid, eta, d, b, tq)
            gates['G4_monotone'] &= mono
            # ---- G3: the closed forms, where their conditions hold ---------------------------------
            if r['horizon_closed_us'] is not None:
                gates['G3_closed_form'] &= 0 <= r['horizon_closed_us'] - r['horizon_us'] < 10 + 1e-6
            for s in (r['scaling_adverse'], r['scaling_measured']):
                if s['closed_form'] is not None:
                    gates['G3_closed_form'] &= (abs(s['factor'] - s['closed_form']) < TOL * s['factor']
                                                and s['turns_at_us'] == t_max)
            rows[name] = r
            Ds.setdefault(name, {})[k] = D
        sessions[k] = rows

    # ---- G2: R22 (vii)'s pooled calculations, reproduced ----------------------------------------------
    pooled = {}
    for name in builds:
        P = np.concatenate([Ds[name][k] for k in sorted(Ds[name], key=int)])
        pooled[name] = vii(P, grid, eta, d, proc)
        sec = MT['secondary'][name]
        gates['G2_reproduced'] &= (pooled[name]['horizon_us'] == sec['horizon_us']
                                   and abs(pooled[name]['eta_needed'] - sec['eta_needed']) < TOL
                                   and abs(pooled[name]['probe_airtime_max_us']
                                           - sec['probe_airtime_max_us']) < 1e-6)

    gate = dict(gates, holds=all(gates.values()))
    print(json.dumps(gate))
    for k in MT['sessions_read']:
        for name in builds:
            r = sessions[k][name]
            sa, sm, sr = r['scaling_adverse'], r['scaling_measured'], r['r22_leaves_excluded']
            print(f"s{k} {name:11s} horizon {r['horizon_us'] / 1e3:7.2f} ms  eta {r['eta_needed']:.4f}  "
                  f"air {r['probe_airtime_max_us']:6.2f} us  scale {sa['factor']:.4f} "
                  f"(T {sa['turns_at_us'] / 1e3:g})  measured {sm['factor']:.4f} "
                  f"(T {sm['turns_at_us'] / 1e3:g})  R22 {sr['factor']:.4f} ({sr['decision']})")
    if not gate['holds']:
        raise SystemExit("gate failed: nothing written")
    if check:
        print("gates hold; nothing written")
        return
    out = {'rule': 'R23', 'script_sha256': sha(os.path.abspath(__file__)),
           'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in
                             ['matched.json', 'margins.json']
                             + [f"../latency/profile_matched/{MT['sessions'][k]['stem']}.npz"
                                for k in MT['sessions_read']]},
           'grid_us': grid, 'T_max_us': t_max, 'batch': b, 't_quantile': tq,
           'reference': {'adverse': {'eta': eta, 'd_S_us': d,
                                     'boundary_us': (1 - eta) * t_max + eta * d},
                         'measured': {'eta': eta_m, 'd_S_us': d_m,
                                      'boundary_us': (1 - eta_m) * t_max + eta_m * d_m}},
           'sessions_read': MT['sessions_read'], 'optimized_build': opt, 'gate': gate,
           'sessions': sessions, 'pooled': pooled}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
