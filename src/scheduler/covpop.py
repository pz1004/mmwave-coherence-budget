#!/usr/bin/env python3
"""Rule R37: the paper's efficiency bounds and its decision audit's screen on the operating
population (protocol.json analysis 'covpop', declared before it ran; R37 is post hoc).

The population.  A user belongs to its scenario's operating population when its genie-aided rate,
the best narrow beam's frequency-averaged rate C* (site.oracle_se; data/<city>.npz se, the largest
of the 64 narrow beams), reaches CQI 1 of TS 38.214 V19.5.0's 4-bit CQI Table 2 (Table 5.2.2.1-3):
QPSK at code rate 78/1024, 2 x 78/1024 = 0.15234375 bit/s/Hz (listed as 0.1523), the table's lowest
entry, below which a UE reports CQI index 0, 'out of range'; equivalently an effective SNR 2^C* - 1
of at least -9.53 dB.  The rule reads each user's channel alone, never a method's selection.  The
same table's highest entry, 7.4063 bit/s/Hz, is the rate cap of rule R40.  Sensitivity: an
effective SNR of at least -6 dB, C* >= log2(1 + 10^-0.6).

What is recomputed.  Each pipeline's own selections (its scenario(): the same measurements, draws
and choices), statistics, bootstrap streams, tags, levels and windows, on the population's users;
blocks without such a user drop out and the rest are renumbered:
  csirs.py      every configuration, coherent|N and per_re|N, the exact choice (csirs_cfg.py's
                'exact', equal to csirs.py's by its gate G3), level 0.05/7, window 0.25
  csirs_cfg.py  the two configured variants, every configuration, level 0.05/7, window 1
  refbounds.py  Top-8, Top-16, Top-32 and Hier-6, level 0.05/28, window 1
  ssbref.py     both layers, seven laws, level 0.05/14, window 1
  the audit     audit_screen.one's draws, streams, estimator and level on the population's users of
                the drawn blocks (a draw with fewer than two such blocks has no bound, and the screen
                excludes nothing on it); then audit_screen.phi and label on the frozen calibration
                record, eps and H.  The auditor's side is covpop_direct.py's.
Descriptive: per scenario, the population's share of users and of G; the per-probe sweep's
(coherent|1000) efficiency in each prespecified stratum of the best beam's SNR (edges 0, 10, 20 dB)
on each population; per block of Phoenix and Philadelphia, whose bounds the paper reports as its
smallest, its users, its share of G and its per-probe efficiency on each population.

Gates, before anything is written:
  C1  on all users the recomputation reproduces every stored efficiency, delta-method standard
      error and bound with both halves exactly (csirs.json, csirs_cfg.json, refbounds.json,
      ssbref.json, audit_bounds.json) and the audit's phi and labels (audit_decisions.json), and the
      pipelines' own gates hold;
  C2  every population bound lies at or below its efficiency; a pipeline's efficiency is below
      one, a drawn calibration's at most one (to 1e-12): one where every user of its blocks selects
      its genie-aided beam in every realization (amended 2026-11-08, after the first run stopped
      here);
  C3  the population from site.oracle_se equals the one from data/<city>.npz in every scenario.

usage:  python3 covpop.py [--gate]      -> covpop.json
        --gate runs C1 and C3, computes no population's statistic and writes nothing
"""
import hashlib, json, math, multiprocessing, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KEY = 'covpop'
OUT = os.path.join(HERE, 'covpop.json')
CQI1 = 2 * 78 / 1024
SNR_M6 = math.log2(1 + 10 ** -0.6)
POPS = {'all': None, 'cqi1': CQI1, 'snr_m6': SNR_M6}
WORKERS = 24
EDGES = (0.0, 10.0, 20.0)
CONC = ('city_4_phoenix_28', 'city_5_philadelphia_28')
ALL_CITIES = ('city_0_newyork_28', 'city_1_losangeles_28', 'city_2_chicago_28',
              'city_3_houston_28', 'city_4_phoenix_28', 'city_5_philadelphia_28',
              'city_6_miami_28', 'city_7_sandiego_28')
SAME = ('eta', 'se_lin', 'eta_lb', 'eta_lb_boot', 'eta_lb_t')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def keep(gse, pop):
    return np.ones(gse.size, bool) if POPS[pop] is None else gse >= POPS[pop]


def stored_genie(city):
    return np.load(os.path.join(HERE, 'data', f'{city}.npz'))['se'][:, 62:].astype(np.float64).max(1)


def _bound(task):
    """One bound: (key, record).  All users keep the pipeline's own block numbering."""
    import csirs, effbound
    key, selk, gse, blocks, nb, alpha, window, tag = task
    eta = float(selk.mean(1).mean() / gse.mean())
    rec = {'eta': eta, 'users': int(gse.size), 'blocks': int(nb)}
    if nb < 2:
        return key, {**rec, 'failed': 'fewer than two blocks'}
    try:
        se_lin = csirs.lin_se(selk, gse, blocks, nb, eta, tag + '|lin')
        lb, e_b, e_t, flips = effbound.lower_bound(selk, gse, blocks, nb, eta, se_lin, tag + '|lb',
                                                   alpha, window)
    except AssertionError as x:
        return key, {**rec, 'failed': f"AssertionError: {x}"[:300]}
    return key, {**rec, 'se_lin': se_lin, 'eta_lb': lb, 'eta_lb_boot': e_b, 'eta_lb_t': e_t,
                 'scan_transitions': flips, 'failed': None}


def tasks_for(prefix, sel, gse, blocks, nb, alpha, window, tag, pops):
    """The bound tasks of one selection array on each population."""
    out = []
    for pop in pops:
        if pop == 'all':
            out.append(((pop,) + prefix, sel, gse, blocks, nb, alpha, window, tag))
            continue
        m = keep(gse, pop)
        present = np.unique(blocks[m])
        out.append(((pop,) + prefix, sel[:, m], gse[m], np.searchsorted(present, blocks[m]),
                    int(present.size), alpha, window, tag))
    return out


# ------------------------------------------------------------- the pipelines ---
def pipelines(pool, pops, seeds, t_probe, grid):
    """Simulate every scenario under each pipeline and queue its bounds; returns the pending
    results, the pipelines' gates and the descriptive records."""
    import csirs, csirs_cfg as cc, refbounds, ssbref, obs
    pending, gates, desc = [], {}, {}
    for n in csirs.SCEN:
        sel, gse, blocks, nb, g, _, _ = cc.scenario(n, seeds, t_probe, grid)
        gates[f'csirs_cfg|{n}'] = bool(g['csirs_gates'] and g['sets_and_exact_choice']
                                       and g['worst_never_better'])
        gates[f'C3|{n}'] = bool(np.array_equal(gse >= CQI1, stored_genie(n) >= CQI1)
                                and np.array_equal(gse >= SNR_M6, stored_genie(n) >= SNR_M6))
        for v, by in sel.items():
            win = 0.25 if v == 'exact' else cc.WINDOW
            for key, sk in by.items():
                pending.append(pool.map_async(_bound, tasks_for(
                    ('csirs_cfg', v, key, n), sk, gse, blocks, nb, csirs.ALPHA_LB, win,
                    f"{n}|{key}", pops), chunksize=1))
        if 'all' not in pops or len(pops) > 1:
            desc[n] = describe(n, sel['exact']['coherent|1000'], gse, blocks)
        del sel
        sel, gse, blocks, nb, h1, _ = refbounds.scenario(n, seeds, t_probe, grid)
        gates[f'refbounds|{n}'] = bool(h1['stored_reproduced'])
        for m in refbounds.REFS:
            pending.append(pool.map_async(_bound, tasks_for(
                ('refbounds', m, n), sel[m], gse, blocks, nb, refbounds.ALPHA_REF,
                refbounds.WINDOW, f"{n}|ref|{m}", pops), chunksize=1))
        del sel
        sel, gse, blocks, nb, g, _, _ = ssbref.scenario(n, seeds, t_probe)
        gates[f'ssbref|{n}'] = bool(g['S1_bit_for_bit'] and g['S2_sweep_reproduces_csirs_json'])
        for (l, k), sk in sel.items():
            pending.append(pool.map_async(_bound, tasks_for(
                ('ssbref', f'guided{l}', k, n), sk, gse, blocks, nb, ssbref.ALPHA, ssbref.WINDOW,
                f"{n}|guided{l}|{k}", pops), chunksize=1))
        del sel
        print(f"  {n}: simulated, bounds queued", flush=True)
    return pending, gates, desc


def describe(n, sel, gse, blocks):
    """Shares, the per-probe sweep's efficiency per stratum and, in CONC, per block."""
    import obs
    pbar = np.load(os.path.join(HERE, 'data', f'{n}.npz'))['pbar'][:, 62:].astype(np.float64)
    snr = 10 * np.log10(obs.P_TX * pbar.max(1) / obs.SIGMA_D2)
    eta = lambda m: float(sel[:, m].mean(1).mean() / gse[m].mean()) if m.any() else None
    out = {}
    for pop in POPS:
        k = keep(gse, pop)
        st = []
        for lo_, hi_ in zip((-np.inf,) + EDGES, EDGES + (np.inf,)):
            m = k & (snr >= lo_) & (snr < hi_)
            st.append({'edges_dB': [lo_ if np.isfinite(lo_) else None,
                                    hi_ if np.isfinite(hi_) else None],
                       'users_share': float(m.sum() / k.sum()),
                       'genie_share': float(gse[m].sum() / gse[k].sum()), 'eta': eta(m)})
        out[pop] = {'users': int(k.sum()), 'users_share': float(k.mean()),
                    'genie_share': float(gse[k].sum() / gse.sum()), 'eta': eta(k), 'strata': st}
    if n in CONC:
        rows = []
        for b in np.unique(blocks):
            m = blocks == b
            mc = m & keep(gse, 'cqi1')
            rows.append({'block': int(b), 'users': int(m.sum()), 'users_cqi1': int(mc.sum()),
                         'genie_share': float(gse[m].sum() / gse.sum()), 'eta': eta(m),
                         'eta_cqi1': eta(mc)})
        out['blocks'] = sorted(rows, key=lambda r: -r['genie_share'])
    return out


# ------------------------------------------------------------------- audit ---
_A = {}


def _audit_load(path):
    z = np.load(path)
    for k in z.files:
        _A[k] = z[k]


def _audit_one(task):
    """audit_screen.one on one population: the three pilots' bounds on one draw."""
    import csirs, effbound, audit_screen as AS
    pop, city, k, d, drawn = task
    s_all, g_all, blocks = _A[city + '.s'], _A[city + '.g'], _A[city + '.blocks']
    m = np.isin(blocks, drawn) & keep(g_all, pop)
    present = np.unique(blocks[m])
    if pop == 'all':
        assert np.array_equal(present, np.sort(drawn)), "every drawn block has users"
    b = np.searchsorted(present, blocks[m])
    nb = int(present.size)
    tag = f"{AS.KEY}|{city}|{k}|{d}"
    rows = []
    for j, ns in enumerate(AS.SAMPLES):
        s, g = s_all[j][:, m], g_all[m]
        eta = float(s.mean(1).mean() / g.mean()) if m.any() else None
        if nb < 2:
            rows.append({'eta_hat': eta, 'eta_lb': None, 'failed': 'fewer than two blocks'})
            continue
        try:
            se_lin = csirs.lin_se(s, g, b, nb, eta, tag + "|lin")
            lb, e_b, e_t, flips = effbound.lower_bound(s, g, b, nb, eta, se_lin, tag + "|lb",
                                                       AS.ALPHA, AS.WINDOW)
            rows.append({'eta_hat': eta, 'eta_lb': float(min(max(lb, 0.0), 1.0)),
                         'eta_lb_raw': lb, 'eta_lb_boot': e_b, 'eta_lb_t': e_t,
                         'scan_transitions': flips, 'failed': None})
        except AssertionError as x:
            rows.append({'eta_hat': eta, 'eta_lb': None, 'failed': f"AssertionError: {x}"[:300]})
    return pop, city, k, d, nb, int(m.sum()), rows


def audit(pops):
    import evaluate, tempfile, audit_screen as AS
    P, i, A = AS.frozen()
    fz = A['frozen']
    z = np.load(os.path.join(HERE, 'audit_pop.npz'))
    arrays, nb = {}, {}
    for c in AS.CITIES:
        se = np.load(os.path.join(HERE, 'data', f'{c}.npz'))['se'][:, 62:].astype(np.float64)
        sel = z[c + '.sel'][:10].astype(np.int64)
        rows = np.arange(se.shape[0])
        arrays[c + '.s'] = np.stack([np.stack([se[rows, sel[r, j]] for r in range(10)])
                                     for j in range(len(AS.SAMPLES))])
        arrays[c + '.g'] = se.max(1)
        arrays[c + '.blocks'] = z[c + '.blocks'].astype(np.int64)
        nb[c] = int(arrays[c + '.blocks'].max()) + 1
    tasks = []
    for c in AS.CITIES:
        for k in fz['budgets']:
            rng = np.random.default_rng(evaluate.stable_seed(AS.KEY, 'draws', c, k))
            for d in range(fz['draws']):
                drawn = rng.choice(nb[c], size=k, replace=False)
                tasks += [(pop, c, k, d, drawn) for pop in pops]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'pop.npz')
        np.savez(path, **arrays)
        with multiprocessing.get_context('spawn').Pool(WORKERS, initializer=_audit_load,
                                                        initargs=(path,)) as pool:
            res = pool.map(_audit_one, tasks, chunksize=2)
    bounds = {}
    for pop, c, k, d, nbk, users, rows in res:
        for ns, r in zip(AS.SAMPLES, rows):
            bounds.setdefault(pop, {}).setdefault(f"{c}|{ns}|{k}", []).append(
                {'draw': d, 'users': users, 'blocks': nbk, **r})
    for by in bounds.values():
        for v in by.values():
            v.sort(key=lambda x: x['draw'])
    # the screen's decisions, audit_screen.decide's arithmetic on each population's bounds
    cal = AS.calibration_record(fz)
    eps, H = fz['eps'], fz['H_us']
    dS = {int(k): v for k, v in fz['reference']['d_S_us'].items()}
    dec = {}
    for pop, by in bounds.items():
        for c in AS.CITIES:
            for ns in AS.SAMPLES:
                for b in AS.BUILDS:
                    D, v, _ = cal[b]
                    for k in map(str, fz['budgets']):
                        lbs = [x['eta_lb'] for x in by[f"{c}|{ns}|{k}"]]
                        ph = [AS.phi(D, v, H, (lb if lb is not None else 0.0), dS[ns]) for lb in lbs]
                        labs = [AS.label(x, eps) if lb is not None else 'not excluded'
                                for x, lb in zip(ph, lbs)]
                        dec.setdefault(pop, {}).setdefault(f"{c}|{ns}|{b}", {})[k] = {
                            'phi_H': ph, 'labels': labs,
                            'excluded_share': float(np.mean([x == 'excluded' for x in labs]))}
    return bounds, dec


def c1(results, bounds, dec):
    """Gate C1: on all users every stored statistic is reproduced exactly."""
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))['configs']
    CG = json.load(open(os.path.join(HERE, 'csirs_cfg.json')))['variants']
    RB = json.load(open(os.path.join(HERE, 'refbounds.json')))['references']
    SB = json.load(open(os.path.join(HERE, 'ssbref.json')))['references']
    BD = json.load(open(os.path.join(HERE, 'audit_bounds.json')))['bounds']
    DC = json.load(open(os.path.join(HERE, 'audit_decisions.json')))['cells']
    bad = []
    for key, r in results.items():
        if key[0] != 'all':
            continue
        _, pipe, *rest = key
        if pipe == 'csirs_cfg':
            v, k, n = rest
            st = (CS[k] if v == 'exact' else CG[v]['configs'][k])['per_scenario'][n]
        elif pipe == 'refbounds':
            m, n = rest
            st = RB[m]['per_scenario'][n]
        else:
            ref, k, n = rest
            st = SB[ref][k]['per_scenario'][n]
        if r['failed'] or any(r[f] != st[f] for f in SAME):
            bad.append('|'.join(key))
    for cell, rows in BD.items():
        mine = bounds['all'][cell]
        for x, y in zip(rows, mine):
            for f in ('eta_hat', 'eta_lb', 'eta_lb_raw', 'eta_lb_boot', 'eta_lb_t'):
                if x.get(f) != y.get(f):
                    bad.append(f"audit|{cell}|{x['draw']}|{f}")
        if len(rows) != len(mine):
            bad.append(f"audit|{cell}|count")
    for cell, by in DC.items():
        for k, r in by.items():
            m = dec['all'][cell][k]
            if r['phi_H'] != m['phi_H'] or r['labels'] != m['labels']:
                bad.append(f"decision|{cell}|{k}")
    return bad


def summarize(results):
    """The stored files' shapes, per population, with each family's least bound and where."""
    out = {}
    for key, r in results.items():
        pop, pipe, *rest = key
        if pipe == 'csirs_cfg':
            v, k, n = rest
            out.setdefault(pop, {}).setdefault('csirs_cfg', {}).setdefault(v, {}).setdefault(
                k, {'per_scenario': {}})['per_scenario'][n] = r
        elif pipe == 'refbounds':
            m, n = rest
            out.setdefault(pop, {}).setdefault('refbounds', {}).setdefault(
                m, {'per_scenario': {}})['per_scenario'][n] = r
        else:
            ref, k, n = rest
            out.setdefault(pop, {}).setdefault('ssbref', {}).setdefault(ref, {}).setdefault(
                k, {'per_scenario': {}})['per_scenario'][n] = r
    def close(rec):
        ps = {n: r for n, r in rec['per_scenario'].items() if not r['failed']}
        lo_at = min(ps, key=lambda n: ps[n]['eta_lb'])
        rec.update(eta_min=min(r['eta'] for r in ps.values()),
                   eta_lb_min=ps[lo_at]['eta_lb'], eta_lb_min_at=lo_at,
                   eta_lb_boot_at_min=ps[lo_at]['eta_lb_boot'],
                   eta_lb_t_at_min=ps[lo_at]['eta_lb_t'],
                   eta_lb_t_min=min(r['eta_lb_t'] for r in ps.values()),
                   failed=sorted(n for n, r in rec['per_scenario'].items() if r['failed']))
    for pop in out.values():
        for v in pop['csirs_cfg'].values():
            for rec in v.values():
                close(rec)
        for rec in pop['refbounds'].values():
            close(rec)
        for ref in pop['ssbref'].values():
            for rec in ref.values():
                close(rec)
    return out


def main():
    import evaluate, protocol
    gate_only = '--gate' in sys.argv
    pops = ['all'] if gate_only else list(POPS)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[v] = '1'
    if not gate_only:
        P = json.load(open(os.path.join(HERE, 'protocol.json')))
        A = [a for a in P['analyses'] if a['key'] == KEY]
        want = A[0].get('amended', {}).get('scripts_sha256', {}).get(
            'covpop.py', A[0]['scripts_sha256']['covpop.py'])
        assert len(A) == 1 and want == sha(os.path.abspath(__file__)), \
            "covpop.py is not the declared script"
        assert not os.path.exists(OUT), f"{OUT} exists: written once"
    seeds = protocol.seeds()
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    t0 = time.time()
    with multiprocessing.get_context('spawn').Pool(WORKERS) as pool:
        pending, gates, desc = pipelines(pool, pops, seeds, t_probe, grid)
        results = {}
        for p in pending:
            for key, r in p.get():
                results[key] = r
    t_pipe = time.time() - t0
    print(f"pipelines: {len(results)} bounds in {t_pipe:.0f} s", flush=True)
    bounds, dec = audit(pops)
    print(f"audit: done in {time.time() - t0 - t_pipe:.0f} s", flush=True)
    bad = c1(results, bounds, dec)
    g = {'C1_reproduced': not bad, 'C1_mismatches': bad[:50],
         'pipelines': all(v for k, v in gates.items() if not k.startswith('C3')),
         'C3_population': all(v for k, v in gates.items() if k.startswith('C3')),
         'detail': gates}
    print(json.dumps({k: v for k, v in g.items() if k != 'detail'}, indent=1), flush=True)
    if bad or not g['pipelines'] or not g['C3_population']:
        raise SystemExit("gate failed: nothing written")
    if gate_only:
        print(f"gates C1 and C3 hold; no population's statistic computed, nothing written "
              f"({time.time() - t0:.0f} s)")
        return
    c2 = [ '|'.join(k) for k, r in results.items()
           if not r['failed'] and not (r['eta_lb'] <= r['eta'] < 1)]
    c2 += [f"audit|{pop}|{cell}|{x['draw']}" for pop, by in bounds.items() for cell, v in by.items()
           for x in v if x['eta_lb'] is not None
           and not (x['eta_lb'] <= x['eta_hat'] + 1e-12 and x['eta_hat'] <= 1 + 1e-12)]
    g['C2_bound_below_eta_below_one'] = not c2
    assert not c2, c2[:20]
    shares = {}
    for c in ALL_CITIES:
        gs = stored_genie(c)
        shares[c] = {pop: {'users_share': float(keep(gs, pop).mean()),
                           'genie_share': float(gs[keep(gs, pop)].sum() / gs.sum())}
                     for pop in POPS}
    out = {'rule': 'R37', 'script_sha256': sha(os.path.abspath(__file__)),
           'populations': {'all': 'every eligible user',
                           'cqi1': f"C* >= 2 x 78/1024 = {CQI1} bit/s/Hz (CQI 1, TS 38.214 "
                                   "V19.5.0 Table 5.2.2.1-3)",
                           'snr_m6': f"C* >= log2(1 + 10^-0.6) = {SNR_M6} bit/s/Hz"},
           'thresholds': {'cqi1': CQI1, 'snr_m6': SNR_M6,
                          'cqi1_effective_snr_db': 10 * math.log10(2 ** CQI1 - 1)},
           'seeds': list(seeds), 'gate': g, 'shares': shares, 'describe': desc,
           'bounds': summarize({k: r for k, r in results.items() if k[0] != 'all'}),
           'audit': {'bounds': {p: v for p, v in bounds.items() if p != 'all'},
                     'decisions': {p: v for p, v in dec.items() if p != 'all'}},
           'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in (
               'csirs.json', 'csirs_cfg.json', 'refbounds.json', 'ssbref.json',
               'audit_bounds.json', 'audit_decisions.json', 'audit_pop.npz')},
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT} in {out['seconds']} s")


if __name__ == '__main__':
    main()
