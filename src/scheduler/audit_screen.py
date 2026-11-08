#!/usr/bin/env python3
"""Rule R26: the decision audit's screen (protocol.json analysis 'decision_audit').

  python3 audit_screen.py --freeze   writes the analysis's frozen record into protocol.json, once,
                                     before any population, bound or session: the session order,
                                     the scripts, the software, the weights, the reference inputs,
                                     the tolerance eps and the budgets
  python3 audit_screen.py --bounds   after audit_pop.py: the efficiency bound of every calibration
                                     draw -> audit_bounds.json
  python3 audit_screen.py --decide   after rounds 1-5 and before any later round: the screening
                                     decisions -> audit_decisions.json, whose sha256 it records in
                                     protocol.json; rounds 6-15 refuse to start without it
  --check-only with --bounds or --decide recomputes and compares, writing nothing

What the screen sees.  For a city, budget k and draw: the sweep's selections (audit_pop.npz) and the
stored per-beam rates (data/<city>.npz) of the users in the drawn blocks only, over realizations
1-10; the declared reference completion time d_S of each pilot; and the calibration timing record,
rounds 1-5 of each build.  It never reads the auditor's rates, population efficiencies or later
sessions.

Calibration draws.  Per city and budget k = 8 (primary), 4 and 12: 200 draws of k of the city's
50 m blocks, without replacement within a draw, from numpy default_rng(stable_seed(
'decision_audit', 'draws', city, k)); each draw serves the three pilots and the three builds.

Bound.  R25's estimator (csirs.lin_se and effbound.lower_bound) at its level 0.05/7 with the scan
over [0, eta_hat], users weighted equally in the ratio of means, the ten realizations as its seeds,
on the bootstrap stream of tag 'decision_audit|city|k|draw' shared by the three pilots; clipped to
[0, 1].  A draw on which the procedure fails has no bound, and the screen excludes nothing on it.
Six bounds at 0.05/7 each (two cities, three pilots) have a family-wise nominal error of at most
6 x 0.05/7 = 0.043 by Bonferroni; each bound's level is nominal, and R25 found coverage short of it
with four blocks.

Decision, per city, pilot, build, k and draw, at H = 10 ms (primary):
    phi(H) = sum_r v_r (1 - D_r/H)^+ - eta_lb (1 - d_S/H)^+,
v the calibration record's weights (each of the five sessions 1/5, its calls equally): excluded if
phi(H) < -eps, unresolved if |phi(H)| <= eps, not excluded otherwise.  Secondary: the same labels
at each period of the protocol's grid, and 'excluded at every period up to H' when excluded at H
and the record's fastest call is at least d_S (Proposition 1(d)).
"""
import argparse, hashlib, json, multiprocessing, os, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROT = os.path.join(HERE, 'protocol.json')
LAT = os.path.join(HERE, '..', 'latency')
OUTD = os.path.join(LAT, 'profile_audit')
BOUNDS = os.path.join(HERE, 'audit_bounds.json')
DECISIONS = os.path.join(HERE, 'audit_decisions.json')
KEY = 'decision_audit'
CITIES = ('city_3_houston_28', 'city_4_phoenix_28')
SAMPLES = (1000, 2000, 3000)
BUILDS = ('resnet18_ts', 'vit_fp16_graph', 'vit_fp32')
BUDGETS = (8, 4, 12)                      # the first is primary
DRAWS = 200
ALPHA = 0.05 / 7
WINDOW = 1.0
N_ROUNDS, CAL_ROUNDS = 15, 5
PHASES_S = {'idle': 60, 'warmup': 60, 'window': 120}
SPACING_MS = 10
H_US = 10_000.0
FEEDBACK_US, PROCESSING_US = 10.0, 0.966  # the processing: the largest per-probe time measured
SCRIPTS = ('../latency/profile_audit.py', '../latency/run_profile_audit.sh', 'audit_pop.py',
           'audit_screen.py', 'audit_direct.py')
SOFTWARE = ('torch', 'torchvision', 'triton', 'cuda', 'cudnn', 'driver', 'python', 'numpy')
WORKERS = 24
_DATA = {}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def protocol():
    P = json.load(open(PROT))
    i = [k for k, a in enumerate(P['analyses']) if a['key'] == KEY]
    assert len(i) == 1, f"protocol.json declares '{KEY}' exactly once"
    return P, i[0]


def write_protocol(P):
    s = json.dumps(P, indent=1, sort_keys=True) + "\n"
    s.encode('ascii')
    open(PROT, 'w').write(s)


def d_s_us(n_samp):
    """The declared reference completion time: 64 probes of pilot, feedback and processing."""
    return 64 * (n_samp / 100.0 + FEEDBACK_US + PROCESSING_US)


def order():
    from evaluate import stable_seed
    out = []
    for rnd in range(1, N_ROUNDS + 1):
        for j in np.random.default_rng(stable_seed(KEY, 'round', rnd)).permutation(len(BUILDS)):
            out.append({'slot': len(out) + 1, 'round': rnd, 'build': BUILDS[int(j)]})
    return out


def freeze(smoke_json):
    P, i = protocol()
    A = P['analyses'][i]
    assert 'frozen' not in A, "the frozen record is written once"
    assert not os.path.exists(OUTD) or not os.listdir(OUTD), "a session ran before the freeze"
    for p in ('audit_pop.json', 'audit_bounds.json', 'audit_rates.npz', 'audit_decisions.json'):
        assert not os.path.exists(os.path.join(HERE, p)), f"{p} exists before the freeze"
    CAD = next(a for a in P['analyses'] if a['key'] == 'cadence')['frozen']
    sm = json.load(open(smoke_json))
    wts = {w['build']: w['weights_sha256'] for w in sm['runs']}
    assert wts['vit_fp16_graph'] == CAD['weights_sha256']['fp16_graph']
    assert wts['vit_fp32'] == CAD['weights_sha256']['fp32']
    assert {k: sm['env'][k] for k in SOFTWARE} == CAD['software']
    DP = json.load(open(os.path.join(HERE, 'audit_direct_pilot.json')))
    AP = json.load(open(os.path.join(HERE, 'audit_pilot.json')))
    assert DP['eligibility']['digest_equal'] and AP['city'] == DP['city']
    A['frozen'] = {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'order': order(),
        'order_rule': "15 rounds; round j runs the three builds in the order ('resnet18_ts', "
                      "'vit_fp16_graph', 'vit_fp32') permuted by numpy.random.default_rng("
                      "evaluate.stable_seed('decision_audit', 'round', j)).permutation(3)",
        'builds': list(BUILDS), 'spacing_ms': SPACING_MS, 'phases_s': PHASES_S,
        'rounds': N_ROUNDS, 'calibration_rounds': CAL_ROUNDS,
        'software': CAD['software'], 'software_source': "the cadence study's frozen record",
        'weights_sha256': {b: wts[b] for b in BUILDS},
        'weights_source': "the smoke test (" + os.path.basename(smoke_json) + ") for "
                          "resnet18_ts; the ViT builds equal the cadence study's",
        'reference': {'samples': list(SAMPLES), 'pilot_us': [n / 100.0 for n in SAMPLES],
                      'feedback_us': FEEDBACK_US, 'processing_us': PROCESSING_US,
                      'd_S_us': {str(n): d_s_us(n) for n in SAMPLES}},
        'cities': list(CITIES), 'realizations_primary': list(range(1, 11)),
        'realizations_fresh': list(range(11, 21)),
        'budgets': list(BUDGETS), 'primary_budget': BUDGETS[0], 'draws': DRAWS,
        'alpha': ALPHA, 'window': WINDOW, 'H_us': H_US,
        'grid_us': [float(t) for t in P['coherence_grid_us']],
        'eps': DP['eps'], 'eps_source': 'audit_direct_pilot.json',
        'pilot_sha256': {'audit_pilot.json': sha(os.path.join(HERE, 'audit_pilot.json')),
                         'audit_direct_pilot.json': sha(os.path.join(HERE,
                                                                     'audit_direct_pilot.json'))},
        'scripts_sha256': {s: sha(os.path.join(HERE, s)) for s in SCRIPTS}}
    write_protocol(P)
    print("frozen: eps", DP['eps'], "; order " + ", ".join(
        f"{o['slot']}:{o['build']}" for o in A['frozen']['order']))


def frozen():
    P, i = protocol()
    A = P['analyses'][i]
    assert A.get('frozen'), "not frozen: run --freeze"
    assert A['frozen']['scripts_sha256']['audit_screen.py'] == sha(os.path.abspath(__file__)), \
        "audit_screen.py differs from the frozen script"
    return P, i, A


# ------------------------------------------------------------------- bounds ---
def _load(path):
    z = np.load(path)
    for c in CITIES:
        _DATA[c] = (z[c + '.s'], z[c + '.g'], z[c + '.blocks'])


def one(task):
    """The three pilots' bounds on one draw."""
    import csirs, effbound
    city, k, d, drawn = task
    s_all, g_all, blocks = _DATA[city]
    m = np.isin(blocks, drawn)
    b = np.searchsorted(np.sort(drawn), blocks[m])
    tag = f"{KEY}|{city}|{k}|{d}"
    rows = []
    for j, ns in enumerate(SAMPLES):
        s, g = s_all[j][:, m], g_all[m]
        eta = float(s.mean(1).mean() / g.mean())
        try:
            se_lin = csirs.lin_se(s, g, b, k, eta, tag + "|lin")
            lb, e_b, e_t, flips = effbound.lower_bound(s, g, b, k, eta, se_lin, tag + "|lb",
                                                       ALPHA, WINDOW)
            rows.append({'eta_hat': eta, 'eta_lb': float(min(max(lb, 0.0), 1.0)),
                         'eta_lb_raw': lb, 'eta_lb_boot': e_b, 'eta_lb_t': e_t,
                         'scan_transitions': flips, 'failed': None})
        except AssertionError as x:
            rows.append({'eta_hat': eta, 'eta_lb': None, 'failed': f"AssertionError: {x}"[:300]})
    return city, k, d, [int(x) for x in np.sort(drawn)], int(m.sum()), rows


def bounds(check):
    import evaluate
    P, i, A = frozen()
    fz = A['frozen']
    PZ = json.load(open(os.path.join(HERE, 'audit_pop.json')))
    assert sha(os.path.join(HERE, 'audit_pop.npz')) == PZ['npz_sha256']
    z = np.load(os.path.join(HERE, 'audit_pop.npz'))
    arrays, nb = {}, {}
    for c in CITIES:
        se = np.load(os.path.join(HERE, 'data', f'{c}.npz'))['se'][:, 62:].astype(np.float64)
        sel = z[c + '.sel'][:10].astype(np.int64)              # realizations 1-10, (10, 3, n)
        rows = np.arange(se.shape[0])
        arrays[c + '.s'] = np.stack([np.stack([se[rows, sel[r, j]] for r in range(10)])
                                     for j in range(len(SAMPLES))])      # (3, 10, n)
        arrays[c + '.g'] = se.max(1)
        arrays[c + '.blocks'] = z[c + '.blocks'].astype(np.int64)
        nb[c] = int(arrays[c + '.blocks'].max()) + 1
    tasks = []
    for c in CITIES:
        for k in fz['budgets']:
            rng = np.random.default_rng(evaluate.stable_seed(KEY, 'draws', c, k))
            for d in range(fz['draws']):
                tasks.append((c, k, d, rng.choice(nb[c], size=k, replace=False)))
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'pop.npz')
        np.savez(path, **arrays)
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(WORKERS, initializer=_load, initargs=(path,)) as pool:
            res = pool.map(one, tasks, chunksize=2)
    secs = time.time() - t0
    out = {'rule': 'R26', 'alpha': ALPHA, 'window': WINDOW, 'draws': fz['draws'],
           'budgets': fz['budgets'], 'bounds': {}, 'draw_blocks': {}, 'burden': {},
           'script_sha256': sha(os.path.abspath(__file__)),
           'inputs_sha256': {'audit_pop.npz': PZ['npz_sha256'],
                             **{f"data/{c}.npz": sha(os.path.join(HERE, 'data', f'{c}.npz'))
                                for c in CITIES}}}
    for c, k, d, blocks, users, rows in res:
        out['draw_blocks'].setdefault(f"{c}|{k}", []).append({'draw': d, 'blocks': blocks,
                                                              'users': users})
        for ns, r in zip(SAMPLES, rows):
            out['bounds'].setdefault(f"{c}|{ns}|{k}", []).append({'draw': d, 'users': users, **r})
    for key in out['bounds']:
        out['bounds'][key].sort(key=lambda x: x['draw'])
    for key in out['draw_blocks']:
        out['draw_blocks'][key].sort(key=lambda x: x['draw'])
    for k in fz['budgets']:
        per = {}
        for c in CITIES:
            us = [x['users'] for x in out['draw_blocks'][f"{c}|{k}"]]
            per[c] = {'blocks': k, 'blocks_in_city': nb[c], 'users_median': float(np.median(us)),
                      'users_min': int(min(us)), 'users_max': int(max(us)),
                      'sweep_measurements_median': float(np.median(us)) * 64 * 10,
                      'genie_values_median': float(np.median(us)) * 64 * 16}
        out['burden'][str(k)] = per
    if check:
        old = json.load(open(BOUNDS))
        old.pop('seconds', None)
        same = json.loads(json.dumps(out)) == old
        print(f"audit_bounds.json {'reproduces' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    out['seconds'] = round(secs, 1)
    assert not os.path.exists(BOUNDS), f"{BOUNDS} exists: written once"
    json.dump(out, open(BOUNDS, 'w'), indent=1)
    n_fail = sum(r['eta_lb'] is None for v in out['bounds'].values() for r in v)
    print(f"wrote {BOUNDS} in {secs:.0f} s; {n_fail} failed bounds")


# ---------------------------------------------------------------- decisions ---
def calibration_record(fz):
    """The calibration sessions (rounds 1-5) of each build: completion times and weights."""
    rec = {}
    for b in BUILDS:
        Ds, stems = [], []
        for o in fz['order']:
            if o['build'] != b or o['round'] > fz['calibration_rounds']:
                continue
            stem = None
            for st in (f"a{o['slot']:02d}_a2", f"a{o['slot']:02d}"):
                p = os.path.join(OUTD, st + '.json')
                if os.path.exists(p) and not json.load(open(p))['failed']:
                    stem = st
                    break
            assert stem, f"slot {o['slot']} has no record that did not fail"
            r = json.load(open(os.path.join(OUTD, stem + '.json')))
            assert r['frozen'] == fz and r['runs'][0]['build'] == b
            z = np.load(os.path.join(OUTD, stem + '.npz'))
            s, e = z['start_ns'], z['end_ns']
            win = s >= int(z['window_from_ns'])
            Ds.append((e[win] - s[win]) / 1e3)
            stems.append(stem)
        assert len(Ds) == fz['calibration_rounds'], (b, stems)
        D = np.concatenate(Ds)
        v = np.concatenate([np.full(d.size, 1.0 / (len(Ds) * d.size)) for d in Ds])
        rec[b] = (D, v, stems)
    return rec


def phi(D, v, T, eta_lb, d_s):
    return float(np.dot(v, np.maximum(1.0 - D / T, 0.0)) - eta_lb * max(1.0 - d_s / T, 0.0))


def label(x, eps):
    return 'excluded' if x < -eps else ('unresolved' if abs(x) <= eps else 'not excluded')


def decide(check):
    P, i, A = frozen()
    fz = A['frozen']
    if not check:
        assert 'decisions_sha256' not in A, "the decisions are recorded once"
        for o in fz['order']:
            if o['round'] > fz['calibration_rounds']:
                for st in (f"a{o['slot']:02d}", f"a{o['slot']:02d}_a2"):
                    assert not os.path.exists(os.path.join(OUTD, st + '.json')), \
                        f"slot {o['slot']} (round {o['round']}) ran before the decisions"
    BND = json.load(open(BOUNDS))
    eps, H, grid = fz['eps'], fz['H_us'], fz['grid_us']
    dS = {int(k): v for k, v in fz['reference']['d_S_us'].items()}
    cal = calibration_record(fz)
    out = {'rule': 'R26', 'eps': eps, 'H_us': H, 'cells': {},
           'calibration_sessions': {b: cal[b][2] for b in BUILDS},
           'record': {b: {'calls': int(cal[b][0].size),
                          'expected_share_left_at_H': float(np.dot(cal[b][1], np.maximum(
                              1.0 - cal[b][0] / H, 0.0))),
                          'min_us': float(cal[b][0].min())} for b in BUILDS},
           'script_sha256': sha(os.path.abspath(__file__)),
           'bounds_sha256': sha(BOUNDS)}
    for c in CITIES:
        for ns in SAMPLES:
            for b in BUILDS:
                D, v, _ = cal[b]
                cell = {}
                for k in map(str, fz['budgets']):
                    lbs = [x['eta_lb'] for x in BND['bounds'][f"{c}|{ns}|{k}"]]
                    ph = [phi(D, v, H, (lb if lb is not None else 0.0), dS[ns]) for lb in lbs]
                    labs = [label(x, eps) if lb is not None else 'not excluded'
                            for x, lb in zip(ph, lbs)]
                    every = [lab == 'excluded' and float(D.min()) >= dS[ns] for lab in labs]
                    grid_ex = [sum(label(phi(D, v, T, (lb if lb is not None else 0.0), dS[ns]),
                                         eps) == 'excluded' for lb in lbs) for T in grid]
                    cell[k] = {'phi_H': ph, 'labels': labs, 'excluded_every_period': every,
                               'grid_excluded_draws': grid_ex,
                               'excluded_share': float(np.mean([x == 'excluded' for x in labs]))}
                out['cells'][f"{c}|{ns}|{b}"] = cell
    out = json.loads(json.dumps(out))
    if check:
        same = out == json.load(open(DECISIONS))
        print(f"audit_decisions.json {'reproduces' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    assert not os.path.exists(DECISIONS), f"{DECISIONS} exists: written once"
    json.dump(out, open(DECISIONS, 'w'), indent=1)
    P, i = protocol()
    P['analyses'][i]['decisions_sha256'] = sha(DECISIONS)
    P['analyses'][i]['decided_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_protocol(P)
    k0 = str(fz['primary_budget'])
    for key, cell in out['cells'].items():
        print(f"  {key}: excluded in {cell[k0]['excluded_share']:.3f} of draws at k = {k0}")
    print(f"wrote {DECISIONS}; its sha256 is recorded in protocol.json")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze', metavar='SMOKE_JSON')
    g.add_argument('--bounds', action='store_true')
    g.add_argument('--decide', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.freeze:
        return freeze(a.freeze)
    if a.bounds:
        return bounds(a.check_only)
    return decide(a.check_only)


if __name__ == '__main__':
    main()
