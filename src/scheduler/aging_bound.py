#!/usr/bin/env python3
"""The perfect-predictor exclusion bound under channel aging.

Proposition 1(b) bounds, under block fading, how far a method that selects among the M
narrow beams can outperform the exhaustive sweep whatever its accuracy.  This script gives
the bound on a channel that varies within the period.  Let G(t) be the
user-mean rate of the best of the M beams at instant t, and r_S(t) <= G(t) that of the
beam the sweep holds from its completion d_S on.  A method with completion time
d_L >= d_S holds a beam whose rate is at most G(t), so for d_L < T <= T_max
    T (R_S(T) - R_L(T)) >= int_{d_S}^{d_L} G dt - int_{d_S}^{T} (G - r_S) dt,
and the right side only falls as T grows, because G - r_S >= 0.  The method therefore
outperforms the sweep for no T <= T_max if
    int_{d_S}^{d_L} G dt >= int_{d_S}^{T_max} [G(t) - r_S(t)] dt.
Under block fading G is constant and r_S = eta_S G, which is Proposition 1(b)'s condition.
The SLACK is the smallest d_L - d_S that satisfies it.  It is a certificate, not a tight
bound: the best beam may change during the transmission, and no fixed beam follows it.

aging.py computes both curves on its t_p/4 grid and discards them after
summarizing.  This script re-runs aging.run_site unchanged.  It wraps aging.summarize,
the module global run_site calls, to keep G(t) = ba[1] (summed over users, divided by
their number) and r_S(t) (the seed mean of the sweep's held-beam curve), and then calls
the original summarize.  Every row the re-run returns must reproduce aging.json: the
GATE.  The script writes aging_bound.json and nothing else.

DECLARED in protocol.json (analysis 'aging_bound') before it ran:
  integrals  trapezoid sums on aging.py's grid, the slack inverted exactly on the
             piecewise-linear G the trapezoid rule assumes
  gate       every numeric leaf of every re-run row within 1e-5 of aging.json, and
             every boolean and string equal
  v = 0      the slack equals Proposition 1(b)'s (1 - eta_S)(T_max - d_S) to eta_S's
             4-decimal rounding, with eta_S as results.json and heldout.json store it
  criterion  an exclusion of rules.json R5 holds under aging at a speed if its
             tau - tau* (fastest runtime, its own profile's tau*) exceeds the largest
             slack over the six evaluation scenarios at that speed

--check-only re-runs one scenario at v = 0 only, where the bound is Proposition 1(b)'s
and nothing new is computed, to check the wrapper and the gate before the run; it writes
nothing.

usage: python3 aging_bound.py [--selftest | --check-only] [--out aging_bound.json]
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOL = 1e-5                 # the gate, as aging.py's v = 0 regression
ETA_ROUND = 5e-5           # eta_S is stored to four decimals


def slack(t, G, rS, dS, Tmax):
    """Smallest Delta >= 0 with int_{dS}^{dS+Delta} G >= int_{dS}^{Tmax} (G - rS).

    Trapezoid integrals on the grid t, which contains dS and Tmax exactly.  Within a
    grid interval G is linear, so its integral is quadratic and the crossing is solved
    in closed form.  Returns (Delta, shortfall, int_{dS}^{Tmax} G); Delta is None if
    no d_L <= Tmax satisfies the condition."""
    t, G, rS = (np.asarray(x, float) for x in (t, G, rS))
    i0, i1 = int(np.searchsorted(t, dS)), int(np.searchsorted(t, Tmax))
    assert abs(t[i0] - dS) < 1e-6 and abs(t[i1] - Tmax) < 1e-6, (dS, Tmax)
    s, g, r = t[i0:i1 + 1], G[i0:i1 + 1], rS[i0:i1 + 1]
    assert (g >= 0).all() and (g - r >= -1e-12).all(), "needs 0 <= r_S <= G"
    h = np.diff(s)
    cg = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * h)])
    short = float(np.sum(0.5 * ((g - r)[1:] + (g - r)[:-1]) * h))
    if short <= 0:
        return 0.0, short, float(cg[-1])
    k = int(np.searchsorted(cg, short))          # first k with cg[k] >= short
    if k >= len(cg):
        return None, short, float(cg[-1])
    a, b, w = g[k - 1], g[k], h[k - 1]
    need = short - cg[k - 1]
    c2 = (b - a) / (2 * w)                       # need = a x + c2 x^2 on [0, w]
    x = 2 * need / (a + math.sqrt(a * a + 4 * c2 * need))
    return float(s[k - 1] + x - dS), short, float(cg[-1])


def _maxdev(a, b):
    """Largest absolute difference over the numeric leaves of two JSON trees; inf if
    their structure, or any boolean or string, differs."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else math.inf
    if a is None or b is None:
        return 0.0 if a is b else math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf
        return max((_maxdev(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max((_maxdev(x, y) for x, y in zip(a, b)), default=0.0)
    return math.inf


def selftest():
    """Synthetic checks of slack(), without the GPU or the channel data."""
    tp, Tmax = 20.339, 10000.0
    dS = 64 * tp
    t = np.array(sorted(set(np.round(np.arange(0.0, Tmax + tp / 8, tp / 4), 9).tolist())
                        | {dS, Tmax}))
    # 1. block fading: G constant, r_S = eta G -> Proposition 1(b)'s slack exactly
    for eta in (0.9993, 0.95, 1.0):
        d, sh, _ = slack(t, np.full(t.size, 5.0), np.full(t.size, 5.0 * eta), dS, Tmax)
        assert abs(d - (1 - eta) * (Tmax - dS)) < 1e-9 * Tmax, (eta, d)
    # 2. a planted time-varying case against brute force over every T of the grid
    G = 5.0 + 0.3 * np.sin(2 * np.pi * t / 700.0)
    rS = G * (0.98 + 0.01 * np.cos(2 * np.pi * t / 1300.0))
    d, sh, _ = slack(t, G, rS, dS, Tmax)
    assert d is not None and 0 < d < Tmax - dS

    def F(x, y):
        """int_0^x of the piecewise-linear interpolant of y on t."""
        i = int(np.searchsorted(t, x, side='right')) - 1
        c = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])
        if i >= t.size - 1:
            return c[-1]
        u = x - t[i]
        sl = (y[i + 1] - y[i]) / (t[i + 1] - t[i])
        return c[i] + y[i] * u + 0.5 * sl * u * u

    def worst(dL):
        """max over grid T in (dL, Tmax] of T (R_L,perfect - R_S)."""
        return max(F(T, G) - F(dL, G) - (F(T, rS) - F(dS, rS)) for T in t if dL < T <= Tmax)
    assert abs(F(dS + d, G) - F(dS, G) - sh) < 1e-6, "the crossing is not where it was solved"
    assert worst(dS + d + 1e-6) <= 1e-6, "the slack does not exclude the perfect predictor"
    assert worst(dS + d - 0.5) > 0, "a smaller slack would also exclude it"
    # 3. the gate's tree comparison
    assert _maxdev({"a": [1.0, True, "x"]}, {"a": [1.0 + 1e-7, True, "x"]}) < 2e-7
    assert _maxdev({"a": [1.0]}, {"a": [1.0, 2.0]}) == math.inf
    assert _maxdev({"a": True}, {"a": False}) == math.inf
    print(f"selftest: block-fading slack exact; planted slack {d:.3f} us confirmed by brute "
          f"force over {t.size} periods; gate comparison checked")


def run(out, check_only=False):
    import aging                                   # not modified
    t0 = time.time()
    cap = {}
    _orig = aging.summarize

    def _capture(name, v_kmh, t, ix, names, dj, df_, curves, ba, bas, ns):
        cap[(name, float(v_kmh))] = {
            "t": np.array(t, float), "G": np.array(ba[1], float),
            "rS": np.array(curves.mean(0)[names.index('sweep64')], float),
            "dS": float(dj['sweep64'])}
        return _orig(name, v_kmh, t, ix, names, dj, df_, curves, ba, bas, ns)

    aging.summarize = _capture

    AJ = json.load(open(os.path.join(HERE, 'aging.json')))
    lat = json.load(open(os.path.join(HERE, aging.evaluate.LAT_FILE)))
    seeds = aging.protocol.seeds()
    speeds = sorted(set([0.0] + list(aging.SPEEDS_KMH)))     # as aging.py's main
    assert AJ['sites'] == aging.EVAL and AJ['speeds_kmh'] == speeds \
        and tuple(AJ['seeds']) == tuple(seeds), "aging.json was not run on this design"
    ref = {(r['site'], r['speed_kmh']): r for r in AJ['rows']}
    eta = {}
    for f in ('results.json', 'heldout.json'):
        for s_ in json.load(open(os.path.join(HERE, f)))['sites']:
            eta[s_['site']] = s_['methods']['sweep64']['eta']
    gate, rows, v0 = [], [], []
    for name in (aging.EVAL[:1] if check_only else aging.EVAL):
        for rec in aging.run_site(name, [0.0] if check_only else speeds, seeds, lat):
            rec.pop('_choice', None)
            key = (name, rec['speed_kmh'])
            dev = _maxdev(rec, ref[key])
            gate.append({"site": name, "speed_kmh": rec['speed_kmh'], "max_abs_dev": dev,
                         "holds": bool(dev <= TOL)})
            c = cap.pop(key)
            d, short, intg = slack(c['t'], c['G'], c['rS'], c['dS'], aging.T_MAX_US)
            rows.append({"site": name, "speed_kmh": rec['speed_kmh'], "d_S_us": c['dS'],
                         "T_max_us": aging.T_MAX_US, "int_G": intg, "shortfall": short,
                         "slack_us": d, "grid_points": int(c['t'].size)})
            if rec['speed_kmh'] == 0.0:
                g, r = c['G'], c['rS']
                assert np.ptp(g) < 1e-9 and np.ptp(r[c['t'] >= c['dS']]) < 1e-9, \
                    "the v = 0 curves are not constant"
                b_ = (1 - eta[name]) * (aging.T_MAX_US - c['dS'])
                v0.append({"site": name, "slack_us": d, "prop1b_us": b_,
                           "eta_S_stored": eta[name],
                           "holds": bool(abs(d - b_) <= ETA_ROUND * (aging.T_MAX_US - c['dS'])
                                         + 1e-9)})
            print(f"  {name} {rec['speed_kmh']:5.1f} km/h: gate {dev:.2e}  slack {d:.3f} us",
                  flush=True)
    assert not cap, f"captured but not returned: {sorted(cap)}"
    if check_only:
        assert all(g['holds'] for g in gate) and all(x['holds'] for x in v0), (gate, v0)
        print(f"check-only: {gate[0]['site']} at v = 0 reproduces aging.json within "
              f"{gate[0]['max_abs_dev']:.1e}; its slack {v0[0]['slack_us']:.3f} us equals "
              f"Proposition 1(b)'s {v0[0]['prop1b_us']:.3f} us to eta_S's rounding; nothing written")
        return
    mx = {f"{v:g}": max(r['slack_us'] for r in rows if r['speed_kmh'] == v) for v in speeds}
    R5 = json.load(open(os.path.join(HERE, 'rules.json')))['R5']
    excl = []
    for r in R5['rows']:
        if not r['excluded']:
            continue
        tau = min(r['charges_us'].values())
        ex = tau - r['tau_star_us']
        excl.append({"component": r['component'], "profile": r['profile'], "tau_us": tau,
                     "tau_star_us": r['tau_star_us'], "excess_us": ex,
                     "holds_at": {k: bool(ex > v) for k, v in mx.items()}})
    res = {"script_sha256": hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "aging_json_sha256": hashlib.sha256(
               open(os.path.join(HERE, 'aging.json'), 'rb').read()).hexdigest(),
           "aging_py_sha256": hashlib.sha256(
               open(os.path.join(HERE, 'aging.py'), 'rb').read()).hexdigest(),
           "bound": "int_{d_S}^{d_L} G >= int_{d_S}^{T_max} (G - r_S); slack = least d_L - d_S",
           "grid": AJ['grid_step'], "tolerance": TOL, "sites": aging.EVAL,
           "speeds_kmh": speeds, "seeds": list(seeds),
           "gate": gate, "gate_holds": all(g['holds'] for g in gate),
           "v0": v0, "v0_holds": all(x['holds'] for x in v0),
           "rows": rows, "max_slack_us": mx,
           "exclusions": excl,
           "all_exclusions_hold": all(all(e['holds_at'].values()) for e in excl),
           "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(HERE, out), 'w'), indent=1)
    print(f"wrote {out} in {res['seconds']}s: gate {'holds' if res['gate_holds'] else 'FAILS'}, "
          f"v = 0 {'holds' if res['v0_holds'] else 'FAILS'}, largest slack "
          + ", ".join(f"{v} us at {k} km/h" for k, v in
                      ((k, f"{x:.2f}") for k, x in mx.items())))
    assert res['gate_holds'] and res['v0_holds'], "see aging_bound.json"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--out', default='aging_bound.json')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.out, check_only=a.check_only)
