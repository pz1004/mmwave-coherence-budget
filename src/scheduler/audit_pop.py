#!/usr/bin/env python3
"""Rule R26: the decision audit's reference populations (protocol.json analysis 'decision_audit').

  python3 audit_pop.py --pilot   Los Angeles, the validation scenario: the coupled measurement at
                                 the three pilots on pilot-tagged realizations and its resources
                                 -> audit_pilot.json (before the freeze)
  python3 audit_pop.py           Houston and Phoenix, after the freeze: the sweep's selections at
                                 the three pilots for realizations 1-20 -> audit_pop.npz and
                                 audit_pop.json, written once
  python3 audit_pop.py --check-only   recomputes and compares, writing nothing

Reference procedures.  The per-probe 64-beam sweep of Section III with a pilot of N complex samples
over 100 MHz, N = 1000, 2000 or 3000 (10, 20 or 30 us), the 10 us feedback and the transmit power
unchanged.  Each probe returns Section III's band-averaged measurement with N in place of 1000
(csirs.coherent's law):
    y = (s2 / 2K_F) ((z + sqrt(lam))^2 + q) - s2,  clipped at 1e-3 s2,
    s2 = sigma_d^2 K_F / N,  lam = 2 K_F P_TX pbar / s2,
with z standard normal and q chi-square with 2K_F - 1 degrees of freedom.  The sweep selects the
argmax of its 64 measurements in the frozen spread order (methods.topk at k = 64).

Noise realizations.  Realization r of city c: numpy default_rng(stable_seed('decision_audit', c,
r)) draws z = standard_normal((n, 64)), then q = chisquare(2K_F - 1, (n, 64)), one pair per user
and narrow beam, shared by the three pilots (common random numbers, so the pilots are paired).
r = 1-10 are the audit's frozen population; r = 11-20 the separately reported fresh-noise sample.
The pilot uses stable_seed('decision_audit', 'pilot', c, r), r = 1-10.

This script writes only the selections and the users' blocks.  Population efficiencies, coverage
and every full-population value are the auditor's (audit_direct.py); the screen
(audit_screen.py) reads the selections of the blocks it draws, with the stored per-beam rates.

The law is csirs.coherent itself, on the GPU as csirs.py ran it (its gate (i) reproduced
sim.measure bit for bit at N = 1000), and the selection is methods.topk at k = 64.  A numpy
re-implementation of the law is not bit-for-bit equal: numpy's and torch's square roots differ in
the last place for some entries (found by the pilot), so the auditor's own selections are compared
with these, not required to equal them.
"""
import argparse, hashlib, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT_NPZ = os.path.join(HERE, 'audit_pop.npz')
OUT = os.path.join(HERE, 'audit_pop.json')
PILOT = os.path.join(HERE, 'audit_pilot.json')
CITIES = ('city_3_houston_28', 'city_4_phoenix_28')
PILOT_CITY = 'city_1_losangeles_28'
SAMPLES = (1000, 2000, 3000)             # pilot dimensions: 10, 20 and 30 us over 100 MHz
PRIMARY = tuple(range(1, 11))            # the audit's frozen realizations
FRESH = tuple(range(11, 21))             # the fresh-noise sample, reported separately


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def draws(city, r, pilot=False):
    """Realization r's (z, q), each (n, 64), as R26 defines them."""
    import evaluate, obs
    n = np.load(os.path.join(HERE, 'data', f'{city}.npz'))['pbar'].shape[0]
    tag = ('decision_audit', 'pilot', city, r) if pilot else ('decision_audit', city, r)
    rng = np.random.default_rng(evaluate.stable_seed(*tag))
    z = rng.standard_normal((n, 64))
    q = rng.chisquare(2 * obs.K_F - 1, (n, 64))
    return z, q


def city_selections(city, reals, pilot=False):
    """(selections (len(reals), 3, n) int8, blocks): csirs.coherent's measurement of the 64 narrow
    beams on each realization's draws, at each pilot, and methods.topk's selection at k = 64."""
    import torch, csirs, evaluate, methods, obs, sim
    site = sim.Site(city)
    L6 = obs.LAYER6_OFFSET
    sel = np.empty((len(reals), len(SAMPLES), site.n), dtype=np.int8)
    zt = torch.zeros_like(site.pbar)
    qt = torch.zeros_like(site.pbar)
    for i, r in enumerate(reals):
        z, q = draws(city, r, pilot)
        zt[:, L6:] = torch.from_numpy(z).to(sim.DEV)
        qt[:, L6:] = torch.from_numpy(q).to(sim.DEV)
        for j, ns in enumerate(SAMPLES):
            _, c = methods.topk(site, csirs.coherent(site.pbar, zt, qt, ns), 64, 0.0)
            sel[i, j] = (c - L6).cpu().numpy()
    return sel, evaluate.block_ids(site.rx)


def pilot():
    """Los Angeles: the coupled measurement, the screen side's efficiency per pilot
    (the stored rates; audit_direct.py --pilot computes its own for the tolerance check), the
    pairing across pilots and the resources."""
    import obs
    assert not os.path.exists(PILOT), f"{PILOT} exists: written once"
    t0 = time.time()
    sel, blocks = city_selections(PILOT_CITY, PRIMARY, pilot=True)
    secs = time.time() - t0
    se = np.load(os.path.join(HERE, 'data', f'{PILOT_CITY}.npz'))['se'][
        :, obs.LAYER6_OFFSET:].astype(np.float64)
    gse = se.max(1)
    rows = np.arange(se.shape[0])
    eta = {str(ns): float(np.mean([se[rows, sel[i, j]].mean() for i in range(sel.shape[0])])
                          / gse.mean()) for j, ns in enumerate(SAMPLES)}
    same = {f"{SAMPLES[a]}-{SAMPLES[b]}": float((sel[:, a] == sel[:, b]).mean())
            for a in range(3) for b in range(a + 1, 3)}
    out = {'rule': 'R26', 'city': PILOT_CITY, 'realizations': list(PRIMARY),
           'seed_tag': "stable_seed('decision_audit', 'pilot', city, r)",
           'samples': list(SAMPLES), 'users': int(sel.shape[2]), 'blocks': int(blocks.max()) + 1,
           'eta_screen_side': eta, 'same_selection_share': same,
           'seconds_selections': round(secs, 1),
           'script_sha256': sha(os.path.abspath(__file__))}
    json.dump(out, open(PILOT, 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('eta_screen_side', 'same_selection_share',
                                          'seconds_selections')}, indent=1))


def compute():
    reals = PRIMARY + FRESH
    arrays, rec = {}, {'rule': 'R26', 'cities': {}, 'samples': list(SAMPLES),
                       'realizations_primary': list(PRIMARY), 'realizations_fresh': list(FRESH),
                       'seed_tag': "stable_seed('decision_audit', city, r)"}
    for city in CITIES:
        t0 = time.time()
        sel, blocks = city_selections(city, reals)
        arrays[f'{city}.sel'] = sel
        arrays[f'{city}.blocks'] = blocks.astype(np.int32)
        rec['cities'][city] = {'users': int(sel.shape[2]), 'blocks': int(blocks.max()) + 1,
                               'users_per_block': np.bincount(blocks).tolist()}
        print(f"  {city}: {sel.shape[2]} users, {blocks.max() + 1} blocks, "
              f"{time.time() - t0:.1f} s", flush=True)
    rec['script_sha256'] = sha(os.path.abspath(__file__))
    rec['inputs_sha256'] = {f"data/{c}.npz": sha(os.path.join(HERE, 'data', f'{c}.npz'))
                            for c in CITIES}
    return arrays, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.pilot:
        return pilot()
    import protocol
    A = protocol.analysis('decision_audit')
    assert A.get('frozen'), "the study is not frozen: run audit_screen.py --freeze first"
    assert A['frozen']['scripts_sha256']['audit_pop.py'] == sha(os.path.abspath(__file__)), \
        "audit_pop.py differs from the frozen script"
    arrays, rec = compute()
    if a.check_only:
        z = np.load(OUT_NPZ)
        old = json.load(open(OUT))
        old.pop('npz_sha256')
        same = all(np.array_equal(z[k], v) for k, v in arrays.items()) and \
            json.loads(json.dumps(rec)) == old
        print(f"audit_pop {'reproduces' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    for p in (OUT, OUT_NPZ):
        assert not os.path.exists(p), f"{p} exists: written once (use --check-only)"
    np.savez_compressed(OUT_NPZ, **arrays)
    rec['npz_sha256'] = sha(OUT_NPZ)
    json.dump(rec, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT} and {OUT_NPZ}")


if __name__ == '__main__':
    main()
