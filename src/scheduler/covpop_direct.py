#!/usr/bin/env python3
"""Rule R37, the auditor's side: rule R26's direct calculation on the operating population
(protocol.json analysis 'covpop', declared before it ran; R37 is post hoc).

Independence.  Like audit_direct.py, this file shares no code with the screen: it imports
audit_direct alone, which imports no module of this repository, and reads audit_rates.npz (the
auditor's own rates and selections), the frozen record, the timing sessions and, for the comparison
only, covpop.json, audit_bounds.json and audit_decisions.json.

The population.  The users whose genie-aided rate C*(u) = max_b C(u, b), from the auditor's own
rates, reaches CQI 1 of TS 38.214 V19.5.0's CQI Table 2 (Table 5.2.2.1-3), 2 x 78/1024 bit/s/Hz;
sensitivity, an effective SNR of at least -6 dB.  On it, per city and pilot, eta_pop with
audit_direct.eta's arithmetic on the population's users (realizations 1-10); per cell (city, pilot,
build), A(H) with the calibration record's weights, and its label (dominated if A(H) < -eps, an
opportunity if A(H) > eps, a tie otherwise).  Against covpop.json's screen on the same population,
per budget: the draws excluded, unresolved and not excluded; the contradictions (an excluded
opportunity) and the misses (a dominated case not excluded); the coverage (the share of draws whose
bound is at most eta_pop, a failed bound counting as not covering); and the largest overshoot of a
bound above eta_pop, with the slack it hides at H, (H - d_S) times the overshoot.

Gate D1: on all users, with the stored screen (audit_bounds.json, audit_decisions.json), the
arithmetic reproduces audit_direct.json's eta_pop, labels, coverage and counts exactly and A(H) to
1e-12: A(H)'s dot product changes in its last bits with the BLAS threads, so that audit_direct.py
--check-only itself no longer reproduces those bits of its stored file.

usage:  python3 covpop_direct.py [--gate]      -> covpop_direct.json (after covpop.py)
"""
import hashlib, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_direct as AD

OUT = os.path.join(HERE, 'covpop_direct.json')
CQI1 = 2 * 78 / 1024
SNR_M6 = math.log2(1 + 10 ** -0.6)
POPS = {'all': None, 'cqi1': CQI1, 'snr_m6': SNR_M6}
A_TOL = 1e-12


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def members(C, pop):
    g = C.max(1)
    return np.ones(g.size, bool) if POPS[pop] is None else g >= POPS[pop]


def eta_on(C, sel, m):
    """audit_direct.eta on the users in m."""
    rows = np.nonzero(m)[0]
    G = C[rows].max(1).mean()
    return [float(np.mean([C[rows, sel[i, j][rows]].mean() for i in range(sel.shape[0])]) / G)
            for j in range(sel.shape[1])]


def cells(pop, R, fz, cal, bounds, labels):
    """Per cell, the direct calculation on the population and the screen's record against it."""
    eps, H = fz['eps'], fz['H_us']
    dS = {int(k): v for k, v in fz['reference']['d_S_us'].items()}
    out, pops = {}, {}
    for city in AD.CITIES:
        C, sel = R[f'{city}.C'], R[f'{city}.sel'][:len(AD.PRIMARY)]
        m = members(C, pop)
        ep_all = eta_on(C, sel, m)
        pops[city] = {'users': int(m.sum()), 'users_share': float(m.mean()),
                      'genie_share': float(C.max(1)[m].sum() / C.max(1).sum()),
                      'eta_pop': dict(zip(map(str, AD.SAMPLES), ep_all))}
        for j, ns in enumerate(AD.SAMPLES):
            ep, ds = ep_all[j], dS[ns]
            for b in AD.BUILDS:
                D, v = AD.mixture(cal[b])
                key = f"{city}|{ns}|{b}"
                a_h = AD.advantage(D, v, H, ep, ds)
                lab = 'dominated' if a_h < -eps else ('opportunity' if a_h > eps else 'tie')
                by_k = {}
                for k in map(str, fz['budgets']):
                    dl = labels[key][k]
                    lbs = bounds[f"{city}|{ns}|{k}"]
                    over = [max(x - ep, 0.0) for x in lbs if x is not None]
                    n_ex = sum(x == 'excluded' for x in dl)
                    by_k[k] = {'draws': len(dl), 'excluded': n_ex,
                               'unresolved': sum(x == 'unresolved' for x in dl),
                               'not_excluded': sum(x == 'not excluded' for x in dl),
                               'contradicted': sum(x == 'excluded' and a_h > eps for x in dl),
                               'missed': sum(x == 'not excluded' and a_h < -eps for x in dl),
                               'coverage': float(np.mean([x is not None and x <= ep
                                                          for x in lbs])),
                               'failed_bounds': sum(x is None for x in lbs),
                               'max_overshoot': max(over) if over else 0.0,
                               'max_overshoot_slack_us': (H - ds) * (max(over) if over else 0.0)}
                out[key] = {'city': city, 'samples': ns, 'build': b, 'd_S_us': ds, 'eta_pop': ep,
                            'A_H': a_h, 'direct_label': lab, 'screen': by_k}
    return out, pops


def main():
    gate_only = '--gate' in sys.argv
    A = AD.frozen()
    fz = A['frozen']
    R = np.load(AD.RATES)
    R0 = json.load(open(AD.RATES[:-4] + '.json'))
    assert sha(AD.RATES) == R0['_npz_sha256'], "audit_rates.npz differs from its record"
    cal_s, _, _ = AD.records(fz)
    cal = {b: cal_s[b] for b in AD.BUILDS}
    # D1: all users, with the stored screen, reproduce audit_direct.json
    DIR = json.load(open(os.path.join(HERE, 'audit_direct.json')))
    BND = json.load(open(os.path.join(HERE, 'audit_bounds.json')))['bounds']
    DEC = json.load(open(os.path.join(HERE, 'audit_decisions.json')))['cells']
    b_all = {key: [x['eta_lb'] for x in v] for key, v in BND.items()}
    l_all = {key: {k: r['labels'] for k, r in by.items()} for key, by in DEC.items()}
    mine, pops = cells('all', R, fz, cal, b_all, l_all)
    bad = []
    for city in AD.CITIES:
        if pops[city]['eta_pop'] != DIR['populations'][city]['eta_pop']:
            bad.append(f"eta_pop|{city}")
    for key, c in mine.items():
        st = DIR['cells'][key]
        for f in ('eta_pop', 'direct_label', 'd_S_us'):
            if c[f] != st[f]:
                bad.append(f"{key}|{f}")
        if abs(c['A_H'] - st['A_H']) > A_TOL:
            bad.append(f"{key}|A_H")
        for k, r in c['screen'].items():
            for f in ('draws', 'excluded', 'unresolved', 'not_excluded', 'contradicted', 'missed',
                      'coverage', 'failed_bounds'):
                if r[f] != st['screen'][k][f]:
                    bad.append(f"{key}|{k}|{f}")
    print(f"D1: {'holds' if not bad else 'FAILS ' + str(bad[:10])}", flush=True)
    if bad:
        raise SystemExit("gate D1 failed: nothing written")
    if gate_only:
        print("gate D1 holds; no population's statistic computed, nothing written")
        return
    P = json.load(open(os.path.join(HERE, 'protocol.json')))
    an = [a for a in P['analyses'] if a['key'] == 'covpop']
    assert len(an) == 1 and an[0]['scripts_sha256']['covpop_direct.py'] == sha(
        os.path.abspath(__file__)), "covpop_direct.py is not the declared script"
    assert not os.path.exists(OUT), f"{OUT} exists: written once"
    CP = json.load(open(os.path.join(HERE, 'covpop.json')))
    out = {'rule': 'R37', 'side': 'auditor', 'script_sha256': sha(os.path.abspath(__file__)),
           'eps': fz['eps'], 'H_us': fz['H_us'], 'gate': {'D1_reproduced': True},
           'covpop_json_sha256': sha(os.path.join(HERE, 'covpop.json')), 'populations': {},
           'cells': {}}
    for pop in ('cqi1', 'snr_m6'):
        b_p = {key: [x['eta_lb'] for x in v] for key, v in CP['audit']['bounds'][pop].items()}
        l_p = {key: {k: r['labels'] for k, r in by.items()}
               for key, by in CP['audit']['decisions'][pop].items()}
        out['cells'][pop], out['populations'][pop] = cells(pop, R, fz, cal, b_p, l_p)
        # the population agrees with the screen's (C3 on the screen side)
        for city in AD.CITIES:
            assert out['populations'][pop][city]['users_share'] == screen_share(CP, pop, city), \
                (pop, city)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")
    for pop, cs in out['cells'].items():
        for key, c in cs.items():
            if c['samples'] == 1000:
                r = c['screen']['8']
                print(f"  {pop} {key}: {c['direct_label']} excluded {r['excluded']}/200, "
                      f"coverage {r['coverage']:.3f}, overshoot {r['max_overshoot_slack_us']:.2f} us")


def screen_share(CP, pop, city):
    return CP['shares'][city][pop]['users_share']


if __name__ == '__main__':
    main()
