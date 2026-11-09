#!/usr/bin/env python3
"""Rule R42: the decision audit's screen at its decision boundary (protocol.json analysis 'boundary',
declared before it ran; R42 is post hoc).

The audit's cases lie far from the screen's threshold, so it could not have shown a wrong exclusion
(Section VI-D).  R42 places synthetic cases at the threshold.  For each audit cell (city, pilot of
n samples, k blocks) and population (all eligible users, as audit_bounds.json and audit_direct.json
hold them; rule R37's covered population, as covpop.json and covpop_direct.json hold it), a perfect
predictor with a deterministic completion time d_L has the auditor's advantage at H = 10 ms

    A = (1 - d_L/H) - eta_pop (1 - d_S/H),            a share of the genie-aided rate G,

and the screen's statistic on a draw with bound eta_lb is phi(H) = (1 - d_L/H) - eta_lb (1 - d_S/H)
= A - Delta, Delta = (eta_lb - eta_pop)(1 - d_S/H) (audit_screen.phi with a one-call record).  The
screen excludes iff phi < -eps, that is iff Delta > A + eps; a draw without a bound excludes nothing.
Every quantity is therefore a property of the stored draws' Delta, whatever d_L.

Per cell: the coverage (the share of draws with a bound at most eta_pop, audit_direct.py's
definition); the largest opportunity any draw excludes, max Delta - eps, as a share of G and in
microseconds of the sweep's slack, Delta H; the share of draws that would exclude an opportunity
just above zero (Delta > eps); the smallest loss excluded in at least 95% of draws (+inf when more
than 5% of draws have no bound); and the share of draws excluding at A = +-1e-5, +-1e-4, +-1e-3,
+-1e-2 and +-0.1.

Gates, before anything is written: B1 the coverage of every cell reproduces audit_direct.json (all
users) and covpop_direct.json (covered) exactly, and B2 the covered cells' largest overshoot in
microseconds reproduces covpop_direct.json's to 1e-9 us.

usage:  python3 boundary.py [--gate]      -> boundary.json   (--gate runs B1-B2 and writes nothing)
"""
import hashlib, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'boundary.json')
INPUTS = ('audit_bounds.json', 'audit_direct.json', 'covpop.json', 'covpop_direct.json')
GRID = (-0.1, -1e-2, -1e-3, -1e-4, -1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 0.1)
POWER = 0.95


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load():
    J = {n: json.load(open(os.path.join(HERE, n))) for n in INPUTS}
    AD, CD = J['audit_direct.json'], J['covpop_direct.json']
    eps, H = AD['eps'], AD['H_us']
    assert CD['eps'] == eps and CD['H_us'] == H
    pops = {}
    # all users: the screen's bounds and the auditor's efficiency per city and pilot
    lb_all = {key: [x['eta_lb'] for x in v] for key, v in J['audit_bounds.json']['bounds'].items()}
    lb_cov = {key: [x['eta_lb'] for x in v] for key, v in J['covpop.json']['audit']['bounds']['cqi1'].items()}
    pops['all'] = (lb_all, {c: AD['populations'][c]['eta_pop'] for c in AD['populations']
                            if not c.startswith('_')}, AD['cells'])
    pops['covered'] = (lb_cov, {c: v['eta_pop'] for c, v in CD['populations']['cqi1'].items()},
                       CD['cells']['cqi1'])
    return J, eps, H, pops


def cell_stats(lbs, eta_pop, d_s, H, eps):
    have = [x for x in lbs if x is not None]
    n = len(lbs)
    f = 1.0 - d_s / H
    delta = np.array([(x - eta_pop) * f for x in have])
    excl = lambda a: float(np.sum(delta > a + eps)) / n
    # the smallest loss b with P(exclude at A = -b) >= POWER: Delta > -b + eps for >= POWER of all draws
    need = int(math.ceil(POWER * n))
    if len(have) < need:
        b95 = math.inf
    else:
        srt = np.sort(delta)[::-1]                     # descending; the need-th largest decides
        b95 = float(max(eps - srt[need - 1], 0.0))
    top = float(delta.max()) if delta.size else -math.inf
    return {'draws': n, 'bounds': len(have),
            'coverage': float(np.mean([x is not None and x <= eta_pop for x in lbs])),
            'largest_excluded_opportunity': max(top - eps, 0.0),
            'largest_overshoot_slack_us': max(top, 0.0) * H,
            'false_exclusion_share_at_zero': excl(0.0),
            'loss_excluded_95': b95,
            'loss_excluded_95_slack_us': b95 * H if math.isfinite(b95) else None,
            'exclusion_share': {f'{a:g}': excl(a) for a in GRID}}


def run():
    J, eps, H, pops = load()
    out = {'rule': 'R42', 'script_sha256': sha(os.path.abspath(__file__)), 'eps': eps, 'H_us': H,
           'grid': list(GRID), 'power': POWER, 'cells': {}, 'gate': {}}
    b1, b2 = [], []
    for pop, (lbs, etas, cells) in pops.items():
        for key, lb in sorted(lbs.items()):
            city, ns, k = key.split('|')
            ref = [c for c in cells.values() if c['city'] == city and str(c['samples']) == ns]
            assert ref and len({c['d_S_us'] for c in ref}) == 1
            d_s, ep = ref[0]['d_S_us'], etas[city][ns]
            assert all(c['eta_pop'] == ep for c in ref)
            st = cell_stats(lb, ep, d_s, H, eps)
            st.update({'population': pop, 'city': city, 'samples': int(ns), 'blocks': int(k),
                       'd_S_us': d_s, 'eta_pop': ep})
            out['cells'][f'{pop}|{key}'] = st
            b1.append(all(c['screen'][k]['coverage'] == st['coverage'] for c in ref))
            if pop == 'covered':
                b2.append(all(abs(c['screen'][k]['max_overshoot_slack_us']
                                  - (H - d_s) * max(max((x - ep for x in lb if x is not None),
                                                        default=0.0), 0.0)) < 1e-9 for c in ref))
    out['gate'] = {'B1_coverage_reproduced': all(b1), 'B2_overshoot_reproduced': all(b2),
                   'cells_checked': len(b1)}
    return out, J


def main():
    gate_only = '--gate' in sys.argv
    t0 = time.time()
    out, J = run()
    print(json.dumps(out['gate']))
    assert out['gate']['B1_coverage_reproduced'] and out['gate']['B2_overshoot_reproduced'], out['gate']
    if gate_only:
        print('gates pass; nothing written')
        return
    assert not os.path.exists(OUT), f'{OUT} exists: written once'
    out['inputs_sha256'] = {n: sha(os.path.join(HERE, n)) for n in INPUTS}
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    for key, c in sorted(out['cells'].items()):
        if c['blocks'] == 8:
            print(f"{key}: cov {c['coverage']:.3f}  max opp {c['largest_excluded_opportunity']:.2e} "
                  f"({c['largest_overshoot_slack_us']:.2f} us)  false@0+ {c['false_exclusion_share_at_zero']:.3f}"
                  f"  loss95 {c['loss_excluded_95']:.2e}")
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
