#!/usr/bin/env python3
"""Rule R25: the calibration audit of the sweep-efficiency bound (protocol.json analysis
'calib_audit').  Declared before it ran.

  python3 calib_audit.py               -> calib_audit.json
  python3 calib_audit.py --check-only  recomputes and compares, writing nothing

The paper bounds the exhaustive sweep's efficiency eta_S = mean(r_S) / mean(G) from below,
scenario by scenario, with csirs.py's procedure (min of a crossed seed-by-block bootstrap bound
and a delta-method t bound, one-sided at 0.05/7), and screens implementations against the
smallest of those bounds.  This audit asks how that procedure behaves when the calibration sees
only part of a scenario, by repeated pilot sampling from fixed ray-traced populations:

  population   per evaluation scenario, the probe model's sweep selections (csirs.py's
               'coherent|1000', sim.measure's law) over the protocol's 10 seeds and the
               genie-aided rates of every eligible user, re-simulated by csirs.scenario, whose
               gates must hold; eta_pop, the ratio of means over all of them
  sample       k of the scenario's 50 m blocks (evaluate.block_ids), k = 4, 8, 12, drawn without
               replacement, uniformly, 200 draws per scenario and k from numpy's
               default_rng(evaluate.stable_seed('calib_audit', scenario, k)); every user of the
               drawn blocks and all 10 seeds, users weighted equally
  bound        effbound.lower_bound at csirs.ALPHA_LB = 0.05/7, its scan over [0, eta_hat] (so a
               small sample cannot leave the scan), bisection to 1e-6, with csirs.lin_se for the
               t part; a draw whose procedure fails is recorded as failed, never redrawn
  reported     coverage, the share of draws with eta_lb <= eta_pop, with a Clopper-Pearson 95%
               interval for its Monte Carlo error; the bound's quantiles; for each of four
               implementations, the share of draws whose bound reaches its threshold
               (exclusion) and the share that reaches it while eta_pop does not (false
               exclusion); the users and measurements a draw represents

Thresholds, (4) solved for the efficiency at adverse inputs, T_max = 10 ms and the sweep's upper
completion time d_S_hi of margins.json: eta* = (T_max - dL_lo) / (T_max - d_S_hi), dL_lo the
latency's lower value plus four probes' airtime, as the paper computes them (reconcile.json's
margin rows); for the image-only FP16 build, which probes nothing and is read in expected rate,
the largest over the matched study's sessions of the lowest efficiency that keeps its exclusion
on the grid (matched_post.json).

Gates: csirs.scenario's own (bit for bit with sim.measure, the per-period means and eta against
the stored results); csirs.lower_bound on each full population reproduces csirs.json's
per-scenario bound exactly, and the audit's settings reproduce it to 1e-6.
"""
import argparse, hashlib, json, multiprocessing, os, sys, tempfile, time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'calib_audit.json')
SCEN = ['city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28', 'city_5_philadelphia_28',
        'city_6_miami_28', 'city_7_sandiego_28']
KEY = 'coherent|1000'
SIZES = (4, 8, 12)
DRAWS = 200
WINDOW = 1.0                # the scan starts at max(0, eta_hat - 1) = 0
IMPLEMENTATIONS = (('resnet18_vision', 'cpu-8t-pcore'), ('vit_b16_vision', 'gpu-resident'),
                   ('resnet18_vision', 'gpu-resident, repeated runs'),
                   ('vit_b16_image_only', 'gpu-resident, fp16_graph'))
WORKERS = 24
_DATA = {}                  # scenario -> (sel, gse, blocks), loaded by each worker


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def thresholds():
    """eta* for the four implementations, from the stored outputs the paper uses."""
    MG = json.load(open(os.path.join(HERE, 'margins.json')))
    RC = json.load(open(os.path.join(HERE, 'reconcile.json')))
    MP = json.load(open(os.path.join(HERE, 'matched_post.json')))
    rows = {(r['component'], r['profile']): r for r in RC['margin_rows']}
    t_max, ds = MG['t_max_us'], MG['dS_hi_us']
    out = {}
    for c, p in IMPLEMENTATIONS[:2]:
        dl = rows[(c, p)]['dL_lo_us']
        out[f"{c}|{p}"] = {'dL_lo_us': dl, 'eta_star': (t_max - dl) / (t_max - ds),
                           'source': "reconcile.json margin_rows"}
    rr = RC['repeated']['C']
    assert (rr['component'], rr['profile']) == ('resnet18_vision', 'gpu-resident')
    low = min(v['lower_us'] for v in rr['runtimes'].values())
    dl = MG['K'] * MG['t_air_us'] + low
    out['resnet18_vision|gpu-resident, repeated runs'] = {
        'dL_lo_us': dl, 'eta_star': (t_max - dl) / (t_max - ds),
        'source': "reconcile.json repeated C, the smaller lower value of its two runtimes"}
    need = [MP['sessions'][k]['fp16_graph']['eta_needed'] for k in MP['sessions']]
    out['vit_b16_image_only|gpu-resident, fp16_graph'] = {
        'eta_star': max(need), 'eta_needed_by_session': need,
        'source': "matched_post.json, the largest per-session eta_needed"}
    return out


def clopper(x, n):
    lo = 0.0 if x == 0 else float(stats.beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(stats.beta.ppf(0.975, x + 1, n - x))
    return [lo, hi]


def _load(path):
    z = np.load(path)
    for name in SCEN:
        _DATA[name] = (z[name + '.sel'], z[name + '.gse'], z[name + '.blocks'])


def one(task):
    """The bound on one draw: (scenario, k, r, blocks drawn) -> its record."""
    import csirs, effbound
    name, k, r, drawn = task
    sel, gse, blocks = _DATA[name]
    m = np.isin(blocks, drawn)
    b = np.searchsorted(np.sort(drawn), blocks[m])
    s, g = sel[:, m], gse[m]
    eta = float(s.mean(1).mean() / g.mean())
    tag = f"calib_audit|{name}|{k}|{r}"
    try:
        se_lin = csirs.lin_se(s, g, b, k, eta, tag + "|lin")
        lb, e_b, e_t, flips = effbound.lower_bound(s, g, b, k, eta, se_lin, tag + "|lb",
                                                   csirs.ALPHA_LB, WINDOW)
        res = {'eta_lb': lb, 'eta_lb_boot': e_b, 'eta_lb_t': e_t, 'scan_transitions': flips,
               'failed': None}
    except AssertionError as x:
        res = {'eta_lb': None, 'failed': f"AssertionError: {x}"[:500]}
    return {'scenario': name, 'k': k, 'draw': r, 'blocks': [int(x) for x in np.sort(drawn)],
            'users': int(m.sum()), 'eta_hat': eta, **res}


def compute():
    import csirs, effbound, evaluate, protocol
    seeds = protocol.seeds()
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))
    assert CS['alpha_lb'] == csirs.ALPHA_LB
    pop, gate = {}, {}
    for name in SCEN:
        sel, gse, blocks, nb, g, *_ = csirs.scenario(name, seeds, t_probe, grid)
        assert g['holds'], (name, g)
        s = sel[KEY]
        eta = float(s.mean(1).mean() / gse.mean())
        st = CS['configs'][KEY]['per_scenario'][name]
        se_lin = csirs.lin_se(s, gse, blocks, nb, eta, f"{name}|{KEY}|lin")
        lb = csirs.lower_bound(s, gse, blocks, nb, eta, se_lin, f"{name}|{KEY}|lb")[0]
        lbw = effbound.lower_bound(s, gse, blocks, nb, eta, se_lin, f"{name}|{KEY}|lb",
                                   csirs.ALPHA_LB, WINDOW)[0]
        gate[name] = {'scenario': g, 'eta_equal': eta == st['eta'],
                      'se_lin_equal': se_lin == st['se_lin'], 'lb_equal': lb == st['eta_lb'],
                      'lb_audit_window_abs_diff': abs(lbw - st['eta_lb']),
                      'holds': bool(eta == st['eta'] and se_lin == st['se_lin']
                                    and lb == st['eta_lb'] and abs(lbw - st['eta_lb']) <= 1e-6)}
        assert gate[name]['holds'], (name, gate[name])
        _DATA[name] = (s, gse, blocks)
        users = np.bincount(blocks, minlength=nb)
        pop[name] = {'eta_pop': eta, 'eta_lb_full': st['eta_lb'], 'blocks': int(nb),
                     'users': int(blocks.size), 'users_per_block': [int(u) for u in users]}

    tasks = []
    for name in SCEN:
        nb = pop[name]['blocks']
        for k in SIZES:
            rng = np.random.default_rng(evaluate.stable_seed('calib_audit', name, k))
            for r in range(DRAWS):
                tasks.append((name, k, r, rng.choice(nb, size=k, replace=False)))
    # numpy-only workers, started fresh (no CUDA or thread pool inherited), one thread each
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'populations.npz')
        np.savez(path, **{f"{n}.{f}": a for n, (s, g, b) in _DATA.items()
                          for f, a in (('sel', s), ('gse', g), ('blocks', b))})
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(WORKERS, initializer=_load, initargs=(path,)) as pool:
            draws = pool.map(one, tasks, chunksize=4)

    thr = thresholds()
    S = len(seeds)
    cells = {}
    for name in SCEN:
        for k in SIZES:
            d = [x for x in draws if x['scenario'] == name and x['k'] == k]
            ok = [x for x in d if x['failed'] is None]
            lbs = np.array([x['eta_lb'] for x in ok])
            ep = pop[name]['eta_pop']
            cov = int((lbs <= ep).sum())
            users = np.array([x['users'] for x in d])
            impl = {}
            for key, t in thr.items():
                ex = int((lbs >= t['eta_star']).sum())
                possible = ep < t['eta_star']
                fx = int(((lbs >= t['eta_star']) & possible).sum())
                impl[key] = {'exclusion_share': ex / len(ok),
                             'exclusion_ci95': clopper(ex, len(ok)),
                             'false_exclusion_possible': bool(possible),
                             'false_exclusion_share': fx / len(ok)}
            cells[f"{name}|{k}"] = {
                'scenario': name, 'k': k, 'draws': len(d), 'failed': len(d) - len(ok),
                'covered': cov, 'coverage': cov / len(ok), 'coverage_ci95': clopper(cov, len(ok)),
                'nominal': 1 - csirs.ALPHA_LB,
                'eta_lb_quantiles': {q: float(np.quantile(lbs, float(q)))
                                     for q in ('0', '0.05', '0.5', '0.95', '1')},
                't_part_binds_share': float(np.mean([x['eta_lb_t'] < x['eta_lb_boot']
                                                     for x in ok])),
                'users': {'min': int(users.min()), 'median': float(np.median(users)),
                          'max': int(users.max())},
                'sweep_probes_median': float(np.median(users)) * 64 * S,
                'implementations': impl}
    pooled = {}
    for k in SIZES:
        cs = [cells[f"{n}|{k}"] for n in SCEN]
        n_ok = sum(c['draws'] - c['failed'] for c in cs)
        cov = sum(c['covered'] for c in cs)
        pooled[str(k)] = {
            'coverage': cov / n_ok, 'coverage_ci95': clopper(cov, n_ok),
            'coverage_by_scenario': [min(c['coverage'] for c in cs),
                                     max(c['coverage'] for c in cs)],
            'failed': sum(c['failed'] for c in cs),
            'exclusion_share_by_scenario': {
                key: [min(c['implementations'][key]['exclusion_share'] for c in cs),
                      max(c['implementations'][key]['exclusion_share'] for c in cs)]
                for key in thr},
            'users_median_by_scenario': [min(c['users']['median'] for c in cs),
                                         max(c['users']['median'] for c in cs)]}
    return {'rule': 'R25', 'script_sha256': sha(os.path.abspath(__file__)),
            'scenarios': SCEN, 'config': KEY, 'sizes': list(SIZES), 'draws': DRAWS,
            'alpha': csirs.ALPHA_LB, 'window': WINDOW, 'seeds': list(seeds),
            'gate': gate, 'population': pop, 'thresholds': thr, 'cells': cells,
            'pooled': pooled, 'draw_records': draws,
            'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in
                              ('csirs.json', 'margins.json', 'reconcile.json',
                               'matched_post.json', 'csirs.py', 'effbound.py', 'evaluate.py')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    out = compute()
    out = json.loads(json.dumps(out))
    if a.check_only:
        same = out == json.load(open(OUT))
        print(f"calib_audit.json {'reproduces' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    assert not os.path.exists(OUT), f"{OUT} exists: written once (use --check-only)"
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")
    for k, v in out['pooled'].items():
        print(f"  k = {k}: coverage {v['coverage']:.3f} {v['coverage_ci95']}, by scenario "
              f"{v['coverage_by_scenario']}, failed {v['failed']}")


if __name__ == '__main__':
    main()
