#!/usr/bin/env python3
"""Rule R49: the coverage of the decision audit's efficiency bound against the number of
calibration blocks (protocol.json analysis 'covcurve', declared before it ran; R49 is post hoc).

The bound.  audit_screen.one's estimator (csirs.lin_se and effbound.lower_bound at 0.05/7 with the
scan over [0, eta_hat], on the bootstrap stream of tag 'decision_audit|city|k|draw'), applied by
covpop._audit_one to the users of a draw's blocks: every eligible user ('all') or rule R37's covered
population ('cqi1'); realizations 1-10; the three pilots.

The draws.  Per city and number of blocks k:
  k = 4, 8, 12   the audit's 200 stored draws (rule R26);
  k = 16         200 draws by the audit's own rule, numpy default_rng(evaluate.stable_seed(
                 'decision_audit', 'draws', city, 16)).choice(N, 16, replace=False);
  k = N - 1      the N leave-one-block-out draws, draw j leaving out block j (N = 22 in Houston, 21
                 in Phoenix).

The truth.  The auditor's population efficiency eta_pop per city, pilot and population, computed
by code that shares none of the screen's (audit_direct.json for all users, covpop_direct.json for
the covered population).

Coverage.  As covpop_direct.py defines it: the share of draws whose bound is at most eta_pop, a
failed bound counting as not covering; with the largest overshoot of a bound above eta_pop and the
slack it hides at H = 10 ms, (H - d_S) times the overshoot.  The bound's nominal one-sided level is
1 - 0.05/7; per cell, the smallest k of the five whose coverage reaches it.

Gates, before anything is written:
  V1  for k = 4, 8 and 12 the coverage computed here from the stored bounds (audit_bounds.json for
      all users, covpop.json for the covered population) equals audit_direct.json's and
      covpop_direct.json's exactly;
  V2  covpop._audit_one recomputes the first ten stored draws of every (population, city, k)
      cell with k = 4, 8, 12 bit for bit.

usage:  python3 covcurve.py [--gate]      -> covcurve.json (refuses to overwrite it)
        --gate runs V1-V2, computes no new draw and writes nothing
"""
import hashlib, json, multiprocessing, os, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'covcurve.json')
POPS = ('all', 'cqi1')
STORED_K = (4, 8, 12)
NEW_K = 16
DRAWS = 200
WORKERS = 24
FIELDS = ('eta_hat', 'eta_lb', 'eta_lb_raw', 'eta_lb_boot', 'eta_lb_t', 'scan_transitions',
          'failed')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def J(name):
    return json.load(open(os.path.join(HERE, name)))


def arrays():
    """covpop.audit's arrays: the sweep's per-pilot selected rates, the genie rate and the blocks."""
    import audit_screen as AS
    z = np.load(os.path.join(HERE, 'audit_pop.npz'))
    arr, nb = {}, {}
    for c in AS.CITIES:
        se = np.load(os.path.join(HERE, 'data', f'{c}.npz'))['se'][:, 62:].astype(np.float64)
        sel = z[c + '.sel'][:10].astype(np.int64)
        rows = np.arange(se.shape[0])
        arr[c + '.s'] = np.stack([np.stack([se[rows, sel[r, j]] for r in range(10)])
                                  for j in range(len(AS.SAMPLES))])
        arr[c + '.g'] = se.max(1)
        arr[c + '.blocks'] = z[c + '.blocks'].astype(np.int64)
        nb[c] = int(arr[c + '.blocks'].max()) + 1
    return arr, nb


def stored_bounds():
    """{(pop, city, ns, k): [row per draw]} from audit_bounds.json (all) and covpop.json (cqi1)."""
    import audit_screen as AS
    AB, CP = J('audit_bounds.json')['bounds'], J('covpop.json')['audit']['bounds']
    out = {}
    for c in AS.CITIES:
        for ns in AS.SAMPLES:
            for k in STORED_K:
                key = f"{c}|{ns}|{k}"
                out[('all', c, ns, k)] = AB[key]
                out[('cqi1', c, ns, k)] = CP['cqi1'][key]
    return out


def truth():
    """eta_pop per (pop, city, ns): the auditor's."""
    import audit_screen as AS
    AD, CD = J('audit_direct.json'), J('covpop_direct.json')
    t = {}
    for c in AS.CITIES:
        for ns in AS.SAMPLES:
            t[('all', c, ns)] = AD['populations'][c]['eta_pop'][str(ns)]
            t[('cqi1', c, ns)] = CD['populations']['cqi1'][c]['eta_pop'][str(ns)]
    return t


def coverage(rows, ep, d_s, h):
    lbs = [r['eta_lb'] for r in rows]
    over = [lb - ep for lb in lbs if lb is not None and lb > ep]
    return {'draws': len(lbs),
            'coverage': float(np.mean([lb is not None and lb <= ep for lb in lbs])),
            'failed_bounds': int(sum(lb is None for lb in lbs)),
            'max_overshoot': max(over) if over else 0.0,
            'max_overshoot_slack_us': (h - d_s) * (max(over) if over else 0.0)}


def stored_coverage(pop, c, ns, k):
    """The coverage the auditor stored for this cell (identical over the three builds)."""
    import audit_screen as AS
    if pop == 'all':
        cells = J('audit_direct.json')['cells']
    else:
        cells = J('covpop_direct.json')['cells'][pop]
    vals = {cells[f"{c}|{ns}|{b}"]['screen'][str(k)]['coverage'] for b in AS.BUILDS}
    assert len(vals) == 1, (pop, c, ns, k, vals)
    return vals.pop()


def run(tasks, arr):
    import covpop
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'pop.npz')
        np.savez(path, **arr)
        with multiprocessing.get_context('spawn').Pool(
                WORKERS, initializer=covpop._audit_load, initargs=(path,)) as pool:
            return pool.map(covpop._audit_one, tasks, chunksize=2)


def gate(arr, nb):
    import evaluate, audit_screen as AS
    fz = AS.frozen()[2]['frozen']
    H = fz['H_us']
    dS = {int(k): v for k, v in fz['reference']['d_S_us'].items()}
    SB, T = stored_bounds(), truth()
    g, bad = {}, []
    # V1: the coverage from the stored bounds is the auditor's
    for (pop, c, ns, k), rows in SB.items():
        got = coverage(rows, T[(pop, c, ns)], dS[ns], H)['coverage']
        if got != stored_coverage(pop, c, ns, k):
            bad.append(('V1', pop, c, ns, k, got))
    g['V1_stored_coverage_reproduced'] = not [b for b in bad if b[0] == 'V1']
    # V2: the first ten stored draws of every cell, recomputed bit for bit
    tasks = []
    for c in AS.CITIES:
        for k in STORED_K:
            rng = np.random.default_rng(evaluate.stable_seed(AS.KEY, 'draws', c, k))
            for d in range(fz['draws']):
                drawn = rng.choice(nb[c], size=k, replace=False)
                if d < 10:
                    tasks += [(pop, c, k, d, drawn) for pop in POPS]
    for pop, c, k, d, nbk, users, rows in run(tasks, arr):
        for ns, r in zip(AS.SAMPLES, rows):
            s = SB[(pop, c, ns, k)][d]
            assert s['draw'] == d
            if any(s.get(f) != r.get(f) for f in FIELDS) or s['users'] != users:
                bad.append(('V2', pop, c, ns, k, d))
    g['V2_draws_recomputed'] = not [b for b in bad if b[0] == 'V2']
    g['holds'] = not bad
    return g, bad


def main():
    import evaluate, audit_screen as AS
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    arr, nb = arrays()
    t0 = time.time()
    g, bad = gate(arr, nb)
    print(f"gate: {g}")
    if bad:
        sys.exit(f"covcurve: gate failed: {bad[:10]}")
    if '--gate' in sys.argv:
        return
    assert not os.path.exists(OUT), f"{OUT} exists: written once"
    fz = AS.frozen()[2]['frozen']
    H = fz['H_us']
    dS = {int(k): v for k, v in fz['reference']['d_S_us'].items()}
    SB, T = stored_bounds(), truth()
    tasks = []
    for c in AS.CITIES:
        rng = np.random.default_rng(evaluate.stable_seed(AS.KEY, 'draws', c, NEW_K))
        for d in range(DRAWS):
            drawn = rng.choice(nb[c], size=NEW_K, replace=False)
            tasks += [(pop, c, NEW_K, d, drawn) for pop in POPS]
        for j in range(nb[c]):
            drawn = np.delete(np.arange(nb[c]), j)
            tasks += [(pop, c, nb[c] - 1, j, drawn) for pop in POPS]
    rows_new = {}
    for pop, c, k, d, nbk, users, rows in run(tasks, arr):
        kk = 'loo' if k == nb[c] - 1 else k
        for ns, r in zip(AS.SAMPLES, rows):
            rows_new.setdefault((pop, c, ns, kk), []).append({'draw': d, 'users': users,
                                                               'blocks': nbk, **r})
    for v in rows_new.values():
        v.sort(key=lambda x: x['draw'])
    nominal = 1 - AS.ALPHA
    cells = {}
    for pop in POPS:
        for c in AS.CITIES:
            for ns in AS.SAMPLES:
                ep = T[(pop, c, ns)]
                row = {}
                for k in STORED_K:
                    row[str(k)] = coverage(SB[(pop, c, ns, k)], ep, dS[ns], H)
                row[str(NEW_K)] = coverage(rows_new[(pop, c, ns, NEW_K)], ep, dS[ns], H)
                row['loo'] = {**coverage(rows_new[(pop, c, ns, 'loo')], ep, dS[ns], H),
                              'blocks': nb[c] - 1}
                order = [str(k) for k in STORED_K] + [str(NEW_K), 'loo']
                reach = next((k for k in order if row[k]['coverage'] >= nominal), None)
                cells[f"{pop}|{c}|{ns}"] = {'eta_pop': ep, 'd_S_us': dS[ns], 'by_k': row,
                                            'first_k_at_nominal': reach}
    out = {'rule': 'R49', 'script_sha256': sha(os.path.abspath(__file__)),
           'alpha': AS.ALPHA, 'nominal': nominal, 'H_us': H, 'blocks_in_city': nb,
           'k': list(STORED_K) + [NEW_K, 'loo'], 'draws_new': DRAWS, 'gate': g, 'cells': cells,
           'new_bounds': {f"{p}|{c}|{ns}|{k}": v for (p, c, ns, k), v in rows_new.items()},
           'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in (
               'audit_bounds.json', 'covpop.json', 'audit_direct.json', 'covpop_direct.json',
               'audit_pop.npz')},
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for key, cell in cells.items():
        print(key, {k: round(v['coverage'], 3) for k, v in cell['by_k'].items()},
              'first at nominal:', cell['first_k_at_nominal'])
    print(f"wrote {OUT} in {out['seconds']} s")


if __name__ == '__main__':
    main()
