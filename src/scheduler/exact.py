#!/usr/bin/env python3
"""The controller re-solved with its ranking time counted exactly, and both adaptive
baselines with a deadline guard.  Declared in protocol.json (analysis 'exact') before it ran.

WHY.  rerank.py's re-solve (the `_rr` files) charges the masked argmax that picks each next
codeword after an inference call (c_rank) on every probe of a sequence that has called the
predictor, the probes taken before the first call included, which keeps the elapsed time a
function of (i, k).  The probes before the first call follow the fixed order and rank nothing.
The exact count is k - kappa, kappa the probe count at the first call, so the planner's state
carries kappa for i >= 1 (sched.solve_exact,
solve_osla_exact, solve_blinkered_exact), the executor tracks it (methods.scheduled with
rank_exact), and the deployed table becomes kappa-major: 65 blocks of the original shape,
whose consult is timed on that table (--time-lookup) and charged to the variants that use it.
And neither form of the adaptive baseline stops a descent at the end of the period, which
the controller does; methods.zhang_guarded and zhang_np2_guarded add that guard and keep the
forms' own rules.

Every output carries the tag `_rx` (the pilot `_rxpilot`, moved out of the tree); no recorded
file is written, nothing is retrained, and the only timing is the augmented table's consult.

usage:  python3 exact.py --selftest            unit checks; writes nothing
        python3 exact.py --check-only DIR      the gates; their outputs go to DIR
        python3 exact.py --pilot DIR           the tables the lookup is timed on (seeds 17/29/43)
        OMP_NUM_THREADS=1 taskset -c 2 python3 exact.py --time-lookup DIR
                                               the consult on them -> lat_lookup_aug.json
        python3 exact.py --run DIR             the declared sequence; logs in DIR
"""
import glob, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
PY = sys.executable
TAG, PILOT = '_rx', '_rxpilot'
VAL = 'city_1_losangeles_28'
HELD = ['city_4_phoenix_28', 'city_5_philadelphia_28', 'city_6_miami_28',
        'city_7_sandiego_28']
P3 = ['--seeds', '17', '29', '43']
EXACT = ['--rank-exact']
ZH = ['--zhang-np2', '--zhang-guard']
PILOT_SEEDS = (17, 29, 43)
PILOT_T = 1000                            # the headline table retime.py replays
TOL = 1e-12


def outputs():
    """Every file the declared sequence writes into the tree."""
    return sorted(set(glob.glob(os.path.join(HERE, f'*{TAG}*')))
                  | {p for p in (os.path.join(HERE, 'exact.json'),
                                 os.path.join(HERE, 'lat_lookup_aug.json'),
                                 os.path.join(HERE, 'selector.json')) if os.path.exists(p)})


def run(args, log, env=None):
    print('  $', ' '.join(args), flush=True)
    with open(log, 'a') as f:
        f.write('$ ' + ' '.join(args) + '\n')
        f.flush()
        subprocess.run([PY, *args], cwd=HERE, check=True, stdout=f, stderr=subprocess.STDOUT,
                       env=env)


def charges():
    import obs, sim
    lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
    tc = lat['tau_restart_us']
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    return dict(lat=lat, TAU=np.array([0.0, tc['1'], tc['2'], tc['3']]),
                T_PROBE=obs.T_P_US + ch['probe_update'], C_LOOK=sim.lookup_charge(lat),
                CR=ch['sweep_select'], C_DEC=ch['np_test'])


def seed_primitives(seed=17):
    """Seed `seed`'s off-policy model on the fit city, as sched.py estimates it."""
    import sim, sched
    q = charges()
    site = sim.Site(sched.FIT)
    model = sim.load_predictor(seed)
    off = sched.explore(site, model, seed, q['TAU'], reorder=True, spec=None)
    return sched.primitives(off)[:3]


# ------------------------------------------------- naive reference solvers -----
# One kappa at a time: the i >= 1 rows are the original solvers with the elapsed time of that
# kappa (sim.elapsed_us patched, so the original code and its matrix-vector products run),
# and the i = 0 rows a separate pass whose compute action lands in (1, kappa = k, k).
def _patched(kappa):
    import sim

    def el(k, i, tau, t_probe, c_lookup=0.0, c_rank=0.0):
        return sim.elapsed_exact_us(k, i, tau, t_probe, c_lookup, c_rank, kappa)
    return el


def naive(kind, pr, tau, T, tp, cl, cr):
    import torch, sim, sched
    from sim import K_MAX, N_EXIT, NB, NC, COMMIT_M, COMMIT_P, PROBE, COMPUTE
    g_m, g_p, P = pr
    fn = {'dp': sched.solve, 'osla': sched.solve_osla, 'blnk': sched.solve_blinkered}[kind]
    keep = sim.elapsed_us
    rows = []
    try:
        for kap in range(K_MAX + 1):
            sim.elapsed_us = _patched(kap)
            V, A = fn(g_m, g_p, P, tau, T, tp, cl, c_rank=cr)
            rows.append((V, A))
    finally:
        sim.elapsed_us = keep
    K1 = K_MAX + 1
    V1 = torch.stack([v for v, _ in rows], 1)            # [i, kappa, k, b, c, f]
    A1 = torch.stack([a for _, a in rows], 1)
    el0 = lambda k: sim.elapsed_exact_us(k, 0, tau, tp, cl, cr, 0)
    first = lambda k: sim.elapsed_exact_us(k, 1, tau, tp, cl, cr, k)
    V0 = torch.zeros((K1, NB, NC, 2), dtype=torch.float64, device=sim.DEV)
    A0 = torch.zeros((K1, NB, NC, 2), dtype=torch.int8, device=sim.DEV)
    Wp0 = torch.zeros((K1, NB, NC), dtype=torch.float64, device=sim.DEV)
    for k in range(K_MAX, -1, -1):
        u = max(0.0, 1.0 - el0(k) / T)
        vm = u * g_m[0, k]
        okp = k < K_MAX and el0(k + 1) < T
        okc = first(k) < T
        vc = V1[1, k, k, :, :, 1]
        if kind == 'dp':
            vpr = sched._probe_value(P[0, k], V0[k + 1, :, :, 0]) if okp else None
            for fresh in (1, 0):
                v = vm.clone(); a = torch.full_like(v, COMMIT_M, dtype=torch.int8)
                if vpr is not None:
                    t = vpr > v; v = torch.where(t, vpr, v); a = torch.where(t, torch.full_like(a, PROBE), a)
                if okc:
                    t = vc > v; v = torch.where(t, vc, v); a = torch.where(t, torch.full_like(a, COMPUTE), a)
                V0[k, :, :, fresh], A0[k, :, :, fresh] = v, a
            continue
        if kind == 'osla':
            d2 = el0(k + 1) if k < K_MAX else float('inf')
            u2 = max(0.0, 1 - d2 / T)
            u3 = max(0.0, 1 - first(k) / T)
            cand = [(vm, COMMIT_M)]
            if d2 < T:
                cand.append((u2 * sched._probe_value(P[0, k], g_m[0, k + 1]), PROBE))
            if okc:
                cand.append((u3 * torch.maximum(g_m[1, k], g_p[1, k]), COMPUTE))
            true_p = sched._probe_value(P[0, k], V0[k + 1, :, :, 0]) if d2 < T else None
        else:                                            # blinkered
            chain_p = sched._probe_value(P[0, k], Wp0[k + 1]) if okp else None
            Wp0[k] = torch.maximum(vm, chain_p) if okp else vm
            # level 1's pure compute chain at kappa = k, as solve_blinkered builds it
            wc = None
            for i in range(N_EXIT, 0, -1):
                ui = max(0.0, 1.0 - sim.elapsed_exact_us(k, i, tau, tp, cl, cr, k) / T)
                here = ui * torch.maximum(g_m[i, k], g_p[i, k])
                ok_i = i < N_EXIT and sim.elapsed_exact_us(k, i + 1, tau, tp, cl, cr, k) < T
                wc = torch.maximum(here, wc) if ok_i else here
            cand = [(vm, COMMIT_M)]
            if okp:
                cand.append((chain_p, PROBE))
            if okc:
                cand.append((wc, COMPUTE))
            true_p = sched._probe_value(P[0, k], V0[k + 1, :, :, 0]) if okp else None
        for fresh in (1, 0):
            pick = torch.stack([c[0] for c in cand]).argmax(0)
            lut = torch.tensor([c[1] for c in cand], dtype=torch.int8, device=sim.DEV)
            a = lut[pick]
            v = vm.clone()
            if true_p is not None:
                v = torch.where(a == PROBE, true_p, v)
            if okc:
                v = torch.where(a == COMPUTE, vc, v)
            V0[k, :, :, fresh], A0[k, :, :, fresh] = v, a
    V1[0] = V0[None].expand(K1, *V0.shape)
    A1[0] = A0[None].expand(K1, *A0.shape)
    unreach = (torch.arange(K1, device=sim.DEV)[:, None] > torch.arange(K1, device=sim.DEV)[None, :])
    A1[1:, unreach] = COMMIT_M
    return V1, A1.permute(1, 0, 2, 3, 4, 5).contiguous()


def valid_mask(A):
    """Entries of a kappa-major table some sequence can reach: every block at i = 0, and
    blocks kappa <= k at i >= 1 (the others hold COMMIT_M by construction)."""
    import torch
    K1 = A.shape[0]
    ok = (torch.arange(K1, device=A.device)[:, None] <= torch.arange(K1, device=A.device)[None, :])
    m = torch.ones(A.shape[:3], dtype=torch.bool, device=A.device)
    m[:, 1:] = ok[:, None, :]
    return m[..., None, None, None].expand_as(A)


def compare(Va, Aa, Vb, Ab, reach=None, valid=None):
    """Actions identical (on `valid` entries), values within TOL."""
    import torch
    dv = float((Va - Vb).abs().max())
    ne = (Aa != Ab) if valid is None else ((Aa != Ab) & valid)
    out = {"max_abs_dv": dv, "actions_differing": int(ne.sum()), "entries": int(ne.numel())}
    if reach is not None:
        out["reachable_differing"] = int((ne & reach).sum())
    out["holds"] = bool(dv <= TOL and out["actions_differing"] == 0)
    return out


# ------------------------------------------------------------------ selftest -----
def selftest():
    import torch, sim, sched, methods, evaluate as E, obs
    q = charges()
    tau, tp, cl, cr = q['TAU'], q['T_PROBE'], q['C_LOOK'], q['CR']
    # 1. the exact elapsed time
    kap = torch.arange(65, dtype=torch.float64, device=sim.DEV)
    for k in (0, 3, 64):
        for i in (0, 1, 3):
            a = sim.elapsed_exact_us(k, i, tau, tp, cl, cr, 0)
            b = sim.elapsed_us(k, i, tau, tp, cl, cr)
            assert a == b, (k, i, a, b)                   # kappa = 0 is elapsed_us, bit for bit
            v = sim.elapsed_exact_us(k, i, tau, tp, cl, cr, kap)
            assert float(v[0]) == b
            want = k * tp + tau[i] + (k + i + 1) * cl + (cr * (k - kap) if i >= 1 else 0 * kap)
            assert torch.allclose(v, want), (k, i)
            if i == 0:
                assert bool((v == v[0]).all())             # nothing ranked before a call
    # 2. vectorized against naive solvers on random models, three periods each
    torch.manual_seed(5)
    from sim import K_MAX, N_EXIT, NB, NC
    g_m = torch.rand((N_EXIT + 1, K_MAX + 1, NB, NC), dtype=torch.float64, device=sim.DEV) * 6
    g_p = torch.rand_like(g_m) * 6
    P = torch.rand((N_EXIT + 1, K_MAX + 1, NB, NC, NB, NC), dtype=torch.float32, device=sim.DEV)
    P = P / P.sum((-2, -1), keepdim=True)
    vec = {'dp': sched.solve_exact, 'osla': sched.solve_osla_exact,
           'blnk': sched.solve_blinkered_exact}
    for kind, fn in vec.items():
        for T in (250.0, 1000.0, 10000.0):
            Vv, Av = fn(g_m, g_p, P, tau, T, tp, cl, cr)
            Vn, An = naive(kind, (g_m, g_p, P), tau, T, tp, cl, cr)
            r = compare(Vv, Av, Vn, An)
            assert r['holds'], (kind, T, r)
    # 3. executor and planner agree on the elapsed time of every user
    pr = seed_primitives(17)
    _, A = sched.solve_exact(*pr, tau, 1000.0, tp, cl, cr)
    site = sim.Site(VAL)
    gen = torch.Generator(device=sim.DEV).manual_seed(E.stable_seed(VAL, 17))
    y = sim.measure(site.pbar, gen)
    ee = sim.load_predictor(17)
    ko = []
    d, c, k_, lv, up = methods.scheduled(site, y, ee, A.cpu().numpy(), tau, tp, cl, c_rank=cr,
                                         rank_exact=True, kappa_out=ko)
    kp = ko[0].clamp(min=0).double()
    want = sim.elapsed_exact_us(k_.double(), lv, torch.tensor(tau, device=sim.DEV), tp, cl, cr, kp)
    # the executor forms k t_p in float32, as every recorded run did: agreement to 1e-3 us
    assert torch.allclose(d, want, rtol=0, atol=1e-3), float((d - want).abs().max())
    called = ko[0] >= 0
    assert bool((ko[0][called] <= k_[called]).all())
    # 4. the guards: no overrun, and with a free check and no deadline pressure, the forms
    for nm in ('city_2_chicago_28', 'city_3_houston_28'):
        site = sim.Site(nm)
        gen = torch.Generator(device=sim.DEV).manual_seed(E.stable_seed(nm, 17))
        y = sim.measure(site.pbar, gen)
        cdec = q['C_DEC']
        for T in (1e9, 10000.0, 250.0):
            g2 = torch.Generator(device=sim.DEV).manual_seed(E.stable_seed(nm, 17, 'r'))
            snap = g2.get_state()
            d0, c0, s0, l0 = methods.zhang(site, y, T, tp, cdec, True, True, gen=g2)
            g3 = torch.Generator(device=sim.DEV); g3.set_state(snap)
            d1, c1, s1, l1, info = methods.zhang_guarded(site, y, T, tp, cdec, g3, 0.0)
            if T == 1e9:
                assert torch.equal(c0, c1) and torch.equal(s0, s1) and torch.equal(l0, l1), nm
                assert torch.equal(d0, d1), nm
            g4 = torch.Generator(device=sim.DEV); g4.set_state(snap)
            d2, c2, s2, l2, info2 = methods.zhang_guarded(site, y, T, tp, cdec, g4, cdec)
            assert bool(((d2 < T) | (l2 == 0)).all()), (nm, T)
        X1 = sim.measure_chi2(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            E.stable_seed(nm, 17)))
        zr = np.random.default_rng(E.stable_seed(nm, 17, 'zf'))
        Zr = zr.standard_normal((site.n, site.pbar.shape[1], 4))
        Qr = zr.chisquare(2 * obs.K_F - 1, (site.n, site.pbar.shape[1], 4))
        zd = methods.zhang_np2_descent(site, X1, Zr, Qr, tp, cdec)
        for T in (1e9, 10000.0, 250.0):
            d0, c0, l0, _ = methods.zhang_np2(zd, T)
            d1, c1, l1, _ = methods.zhang_np2_guarded(site, X1, Zr, Qr, T, tp, cdec, 0.0, zd)
            if T == 1e9:
                assert torch.equal(c0, c1) and np.array_equal(l0, l1), nm
                assert torch.equal(d0, d1), nm
            d2, c2, l2, dg = methods.zhang_np2_guarded(site, X1, Zr, Qr, T, tp, cdec, cdec, zd)
            assert bool(((d2 < T) | torch.tensor(l2 == 0, device=sim.DEV)).all()), (nm, T)
    print("selftest: elapsed time (kappa = 0 is the first re-solve's, bit for bit), the three vectorized "
          "solvers against naive per-kappa solvers on random models, executor = planner on "
          "every user, and both guarded forms (exact with a free check and no deadline "
          "pressure; never overrunning) all pass")


# ---------------------------------------------------------------- the gates -----
def check_only(d):
    import torch, rerank, sim, sched
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, 'gates.log')
    gates = {}
    # (a) the evaluator in rerank.py's mode (--rank-charge --zhang-np2), on the _rr tables,
    #     guards off
    out = os.path.join(d, 'validation_gate_rr.json')
    if not os.path.exists(out):
        run(['evaluate.py', '--rank-charge', '--zhang-np2', '--tables-suffix', '_rr',
             '--sites', VAL, '--out', out], log)
    gates['a_rr_validation'] = rerank.compare_json(
        json.load(open(out)), json.load(open(os.path.join(HERE, 'validation_rr.json'))))
    gates['a_rr_validation']['holds'] = gates['a_rr_validation']['differing'] == 0
    # (b), (c), (d) on seed 17's off-policy model
    q = charges()
    tau, tp, cl, cr = q['TAU'], q['T_PROBE'], q['C_LOOK'], q['CR']
    pr = seed_primitives(17)
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    rec_rr = np.load(os.path.join(HERE, 'tables_17_rr.npz'))
    reach = torch.tensor(np.load(os.path.join(HERE, 'coverage_17_rr.npz'))['cnt_off'] > 0,
                         device=sim.DEV)
    old = {'dp': lambda T, c: sched.solve(*pr, tau, T, tp, cl, c_rank=c),
           'osla': lambda T, c: sched.solve_osla(*pr, tau, T, tp, cl, c_rank=c),
           'blnk': lambda T, c: sched.solve_blinkered(*pr, tau, T, tp, cl, c_rank=c)}
    new = {'dp': lambda T, c: sched.solve_exact(*pr, tau, T, tp, cl, c),
           'osla': lambda T, c: sched.solve_osla_exact(*pr, tau, T, tp, cl, c),
           'blnk': lambda T, c: sched.solve_blinkered_exact(*pr, tau, T, tp, cl, c)}
    b_, c_ = {}, {}
    for kind in old:
        bz, cz = [], []
        for T in grid:
            # (b) zero charge: every kappa block is the old[kind] table
            Vo, Ao = old[kind](float(T), 0.0)
            Vn, An = new[kind](float(T), 0.0)
            K1 = An.shape[0]
            bz.append(compare(Vn, An, Vo[:, None].expand_as(Vn),
                              Ao[None].expand_as(An), valid=valid_mask(An)))
            # (c) with c_rank: block kappa = 0 at i >= 1 is the every-probe solve, and its _rr table
            Vo, Ao = old[kind](float(T), cr)
            Vn, An = new[kind](float(T), cr)
            r = compare(Vn[1:, 0], An[0, 1:], Vo[1:], Ao[1:],
                        reach=reach[1:][..., None].expand_as(Ao[1:]))
            recA = torch.tensor(rec_rr[f'{kind}off_{T}'], device=sim.DEV)
            r["vs_recorded_tables_actions_differing"] = int((An[0, 1:] != recA[1:]).sum())
            r["holds"] = r["holds"] and r["vs_recorded_tables_actions_differing"] == 0
            cz.append(r)
        b_[kind] = {"holds": all(x['holds'] for x in bz),
                    "max_abs_dv": max(x['max_abs_dv'] for x in bz),
                    "actions_differing": sum(x['actions_differing'] for x in bz)}
        c_[kind] = {"holds": all(x['holds'] for x in cz),
                    "max_abs_dv": max(x['max_abs_dv'] for x in cz),
                    "actions_differing": sum(x['actions_differing'] for x in cz),
                    "vs_recorded_tables_actions_differing":
                        sum(x['vs_recorded_tables_actions_differing'] for x in cz)}
    gates['b_zero_charge_every_kappa'] = b_
    gates['c_kappa0_is_rr'] = c_
    # (d) naive per-kappa solvers on the real model
    d_ = {}
    for kind in new:
        rs = []
        for T in (250.0, 1000.0, 10000.0):
            Vv, Av = new[kind](T, cr)
            Vn, An = naive(kind, pr, tau, T, tp, cl, cr)
            rs.append(compare(Vv, Av, Vn, An))
        d_[kind] = {"holds": all(x['holds'] for x in rs),
                    "max_abs_dv": max(x['max_abs_dv'] for x in rs),
                    "actions_differing": sum(x['actions_differing'] for x in rs)}
    gates['d_naive_vs_vectorized'] = d_
    # (e) the guards against the unguarded forms at T = 10 ms, Chicago, seed 17
    import methods, evaluate as E, obs
    nm = 'city_2_chicago_28'
    site = sim.Site(nm)
    y = sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(E.stable_seed(nm, 17)))
    e_ = {}
    for cg, lab in ((0.0, 'free_check'), (q['C_DEC'], 'charged_check')):
        g2 = torch.Generator(device=sim.DEV).manual_seed(E.stable_seed(nm, 17, 'r'))
        snap = g2.get_state()
        d0, c0, s0, l0 = methods.zhang(site, y, 10000.0, tp, q['C_DEC'], True, True, gen=g2)
        g3 = torch.Generator(device=sim.DEV); g3.set_state(snap)
        d1, c1, s1, l1, _ = methods.zhang_guarded(site, y, 10000.0, tp, q['C_DEC'], g3, cg)
        X1 = sim.measure_chi2(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            E.stable_seed(nm, 17)))
        zr = np.random.default_rng(E.stable_seed(nm, 17, 'zf'))
        Zr = zr.standard_normal((site.n, site.pbar.shape[1], 4))
        Qr = zr.chisquare(2 * obs.K_F - 1, (site.n, site.pbar.shape[1], 4))
        zd = methods.zhang_np2_descent(site, X1, Zr, Qr, tp, q['C_DEC'])
        e0, f0, m0, _ = methods.zhang_np2(zd, 10000.0)
        e1, f1, m1, _ = methods.zhang_np2_guarded(site, X1, Zr, Qr, 10000.0, tp, q['C_DEC'], cg, zd)
        e_[lab] = {"simplified_beam_differs": float((c0 != c1).float().mean()),
                   "published_beam_differs": float((f0 != f1).float().mean())}
    e_['holds'] = (e_['free_check']['simplified_beam_differs'] == 0.0
                   and e_['free_check']['published_beam_differs'] == 0.0)
    gates['e_guarded_vs_unguarded_10ms'] = e_
    gates['holds'] = bool(gates['a_rr_validation']['holds']
                          and all(v['holds'] for v in b_.values())
                          and all(v['holds'] for v in c_.values())
                          and all(v['holds'] for v in d_.values()) and e_['holds'])
    json.dump(gates, open(os.path.join(d, 'gates.json'), 'w'), indent=1)
    print(json.dumps(gates, indent=1))


# ------------------------------------------------------------------- the pilot -----
def pilot(d):
    """dp's tables for seeds 17/29/43 with the real table's lookup charge (sim.lookup_charge):
    the tables --time-lookup replays.  Written under the pilot tag and moved to DIR, out of the
    tree."""
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, 'pilot.log')
    run(['sched.py', *EXACT, '--c-lookup-aug', 'original', '--variants', 'dp', '--seeds',
         *map(str, PILOT_SEEDS), '--out-suffix', PILOT], log)
    sizes = {}
    for f in [f'tables_{s}{PILOT}.npz' for s in PILOT_SEEDS] + \
             [f'coverage_{s}{PILOT}.npz' for s in PILOT_SEEDS] + [f'scheduler{PILOT}.json']:
        p = os.path.join(HERE, f)
        sizes[f] = os.path.getsize(p)
        shutil.move(p, os.path.join(d, f))
    json.dump(sizes, open(os.path.join(d, 'pilot_sizes.json'), 'w'), indent=1)
    print(json.dumps(sizes, indent=1))


# ------------------------------------------------------- the lookup, timed -----
def _replay_aug(d, seed):
    """States the pilot's headline controller consults its kappa-major table in, in
    decision order on the validation city, as retime.replay_states records them."""
    import torch, sim, methods, evaluate as E
    q = charges()
    site = sim.Site(VAL)
    tab = np.load(os.path.join(d, f'tables_{seed}{PILOT}.npz'))[f'dppi_{PILOT_T}']
    gen = torch.Generator(device=sim.DEV).manual_seed(E.stable_seed(site.name, seed))
    y = sim.measure(site.pbar, gen)
    ee = sim.load_predictor(seed)
    tr = []
    methods.scheduled(site, y, ee, tab, q['TAU'], q['T_PROBE'], q['C_LOOK'], trace=tr,
                      c_rank=q['CR'], rank_exact=True)
    live = np.stack([m for m, _ in tr])
    S = np.stack([s for _, s in tr])                     # (steps, n, 6)
    flat = np.concatenate([S[live[:, r], r] for r in range(min(2000, live.shape[1]))])
    return tab, flat


def time_lookup(d):
    assert os.sched_getaffinity(0) == {2}, "run as: OMP_NUM_THREADS=1 taskset -c 2 ..."
    assert os.environ.get('OMP_NUM_THREADS') == '1', "run with OMP_NUM_THREADS=1"
    import retime, sim
    from calibrate import nearest_rank_p95, timed
    out_path = os.path.join(HERE, 'lat_lookup_aug.json')
    assert not os.path.exists(out_path), "lat_lookup_aug.json exists: nothing is re-timed"
    B_LO, B_W, NB, C_LO, C_W, NC = sim.B_LO, sim.B_W, sim.NB, sim.C_LO, sim.C_W, sim.NC
    W, N, CHUNK = retime.WARMUP, retime.TIMED, retime.CHUNK

    def per_call(tab, st, aug):
        lv = st[:, 0].astype(int).tolist(); kk = st[:, 1].astype(int).tolist()
        fr = st[:, 2].astype(int).tolist(); best = st[:, 3].tolist(); mean = st[:, 4].tolist()
        kp = st[:, 5].astype(int).tolist() if aug else None
        n = len(lv)
        pos = [0]

        def consult(i):
            # what the executor computes per decision: bucket, class, and the index (6-D on
            # the augmented table: the first-call block, then the original table's five)
            k = kk[i]
            if k == 0:
                b = c = 0
            else:
                b = int((best[i] - B_LO) // B_W) + 1
                b = 1 if b < 1 else (NB - 1 if b > NB - 1 else b)
                c = int((mean[i] - C_LO) // C_W) + 1
                c = 1 if c < 1 else (NC - 1 if c > NC - 1 else c)
            if aug:
                return tab[kp[i], lv[i], k, b, c, fr[i]]
            return tab[lv[i], k, b, c, fr[i]]

        def chunk():
            p = pos[0]
            for j in range(CHUNK):
                consult((p + j) % n)
            pos[0] = (p + CHUNK) % n

        def one():
            p = pos[0]
            consult(p)
            pos[0] = (p + 1) % n
        a = nearest_rank_p95(timed(chunk, W, N)) / 1000.0 / CHUNK
        b = nearest_rank_p95(timed(chunk, W // 4, N // 4)) / 1000.0 / CHUNK
        pc = nearest_rank_p95(timed(one, W, N)) / 1000.0
        pc2 = nearest_rank_p95(timed(one, W // 4, N // 4)) / 1000.0
        return {"charge_us": a, "recheck_us": b, "drift": max(abs(b - a) / a, abs(pc2 - pc) / pc),
                "percall_p95_us": pc, "percall_recheck_us": pc2, "n_states": n,
                "table_shape": list(tab.shape), "table_bytes": int(tab.nbytes)}
    # every replay first (the GPU), then every timing (the pinned core alone)
    reps = {s: (retime.replay_states(s), _replay_aug(d, s)) for s in PILOT_SEEDS}
    rows, ref = [], []
    for s in PILOT_SEEDS:
        (t0, s0), (t1, s1) = reps[s]
        r0 = per_call(t0, s0, False); r0.update(seed=s, component="lookup_real (the original table, re-timed)")
        r1 = per_call(t1, s1, True); r1.update(seed=s, component="lookup_aug")
        ref.append(r0); rows.append(r1)
        print(f"  seed {s}: original table {r0['percall_p95_us']:.3f} us per call, augmented "
              f"{r1['percall_p95_us']:.3f} us ({r1['n_states']} states)", flush=True)
    con = retime.contention(rows + ref)
    recorded = json.load(open(os.path.join(HERE, 'lat_lookup_real.json')))['charge_us']
    ref_pc = max(r['percall_p95_us'] for r in ref)
    out = {"env": retime.env('cpu-1t-pcore'),
           "method": ("as for the original table: the pilot's kappa-major (65, 4, 65, 40, 10, 2) dp tables of seeds "
                      "17/29/43 at T = 1 ms, consults replayed in order from the deployed "
                      "controller's trajectories on the validation city (2,000 users), each "
                      "consult computing the bucket and class and indexing the first-call "
                      "block; nearest-rank p95 per seed, largest of three seeds, timed per call "
                      "(charged) and batched; each timed twice.  The original tables re-timed in "
                      "the same session by the same code, as a check"),
           "rows": rows, "reference_rows": ref, "contention": con,
           "original_table_us": recorded, "reference_percall_us": round(ref_pc, 3),
           "reference_within_10pct": bool(abs(ref_pc / recorded - 1) <= 0.10),
           "percall_us": round(max(r['percall_p95_us'] for r in rows), 3),
           "batched_us": round(max(r['charge_us'] for r in rows), 3),
           "charged_rule": "per call, as the original table's consult is charged",
           "table_shape": rows[0]['table_shape'], "table_bytes": rows[0]['table_bytes']}
    out["charge_us"] = out["percall_us"]
    if con['contended'] or not out['reference_within_10pct']:
        json.dump(out, open(os.path.join(d, 'lat_lookup_aug_REJECTED.json'), 'w'), indent=1)
        sys.exit(f"the machine is not as the original table was timed on (contended={con['contended']}, reference "
                 f"{ref_pc:.3f} us against {recorded} us): nothing written to the tree")
    json.dump(out, open(out_path, 'w'), indent=1)
    print(f"lookup_aug charged {out['charge_us']} us per call (the original table re-timed at "
          f"{out['reference_percall_us']} us against {recorded}); contention {con}")


# -------------------------------------------------------------- solver cost -----
def solver_cost(reps=5):
    """The kappa-vectorized passes over the whole grid on seed 17's model, timed as
    complexity.py times the (i, k) solver: median of `reps` after one warm-up, twice."""
    import torch, sim, sched, complexity
    q = charges()
    tau, tp, cr = q['TAU'], q['T_PROBE'], q['CR']
    cla = sim.lookup_charge_aug(q['lat'])
    tau0 = np.zeros_like(tau)
    S = {'dp': lambda pr, T: sched.solve_exact(*pr, tau, float(T), tp, cla, cr),
         'blind': lambda pr, T: sched.solve_exact(*pr, tau0, float(T), tp, cla, cr),
         'myopic': lambda pr, T: sched.solve_osla_exact(*pr, tau, float(T), tp, cla, cr),
         'blinkered': lambda pr, T: sched.solve_blinkered_exact(*pr, tau, float(T), tp, cla, cr)}
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    pr = seed_primitives(17)
    a = complexity.bench(pr, grid, S, reps, sim.DEV)
    b = complexity.bench(pr, grid, S, reps, sim.DEV)
    return {"seed": 17, "reps": reps, "gpu_util_before": complexity.gpu_util(),
            "solve_gpu": a, "solve_gpu_recheck": b,
            "recheck_drift": max(abs(a[k]['total_median_s'] / b[k]['total_median_s'] - 1)
                                 for k in S)}


# ------------------------------------------------------ the declared sequence ---
def full_run(d):
    assert os.path.exists(os.path.join(HERE, 'lat_lookup_aug.json')), "time the lookup first"
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, 'run.log')
    steps = {}

    def step(name, out, fn):
        t0 = time.time()
        if os.path.exists(os.path.join(HERE, out)):
            print(f"== {name}: {out} exists, skipped", flush=True)
            return
        print(f"== {name}", flush=True)
        fn()
        assert os.path.exists(os.path.join(HERE, out)), out
        steps[name] = round(time.time() - t0, 1)

    step('solve', f'scheduler{TAG}.json', lambda: run(['sched.py', *EXACT, '--out-suffix', TAG], log))
    ev = ['evaluate.py', *EXACT, '--tables-suffix', TAG]
    step('validation', f'validation{TAG}.json',
         lambda: run([*ev, *ZH, '--sites', VAL, '--out', f'validation{TAG}.json'], log))
    step('results', f'results{TAG}.json',
         lambda: run([*ev, *ZH, '--validation', f'validation{TAG}.json',
                      '--out', f'results{TAG}.json'], log))
    step('heldout', f'heldout{TAG}.json',
         lambda: run([*ev, *ZH, '--validation', f'validation{TAG}.json', '--sites', *HELD,
                      '--out', f'heldout{TAG}.json'], log))
    s3 = f'{TAG}_3seed'
    step('validation_3seed', f'validation{s3}.json',
         lambda: run([*ev, *P3, '--sites', VAL, '--out', f'validation{s3}.json'], log))
    step('results_3seed', f'results{s3}.json',
         lambda: run([*ev, *P3, '--validation', f'validation{s3}.json',
                      '--out', f'results{s3}.json'], log))
    step('heldout_3seed', f'heldout{s3}.json',
         lambda: run([*ev, *P3, '--validation', f'validation{s3}.json', '--sites', *HELD,
                      '--out', f'heldout{s3}.json'], log))
    step('analyze', f'report{TAG}.json', lambda: run(['analyze.py', '--suffix', TAG], log))
    step('analyze_3seed', f'report{s3}.json', lambda: run(['analyze.py', '--suffix', s3], log))
    import sweeps
    sweeps.TAG, sweeps.EXTRA = TAG, list(EXACT)
    for cond in list(sweeps.CONDITIONS) + list(sweeps.PREDQ):
        step(f'sweep {cond}', f'sweep{TAG}_{cond}.json', lambda c=cond: sweeps.build(c))
    step('summary sweeps', f'sweeps{TAG}.json', lambda: sweeps.summarize(path=f'sweeps{TAG}.json'))
    step('summary predq', f'predq{TAG}.json',
         lambda: sweeps.summarize(sweeps.PREDQ, f'predq{TAG}.json',
                                  {"q": "oracle-mixed anytime predictor: the rate-optimal "
                                        "beam ranks first for a fraction q of receivers; "
                                        "charges unchanged"}))
    step('rotation', f'selection_stability{TAG}.json',
         lambda: run(['selection_stability.py', '--suffix', TAG], log))
    step('interleave', f'interleave{TAG}.json', lambda: run(['interleave.py', '--suffix', TAG], log))
    step('rules', f'rules{TAG}.json', lambda: run(['rules_suffix.py', TAG], log))
    p = os.path.join(HERE, 'exact.json')
    rec = json.load(open(p)) if os.path.exists(p) else {}
    if 'solver_cost' not in rec:
        rec['solver_cost'] = solver_cost()
    q = charges()
    lk = json.load(open(os.path.join(HERE, 'lat_lookup_aug.json')))
    gp = os.path.join(d, 'gates.json')
    sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
    rec.update({"tag": TAG, "c_rank_us": q['CR'], "c_lookup_us": q['C_LOOK'],
                "c_lookup_aug_us": lk['charge_us'], "c_guard_us": q['C_DEC'],
                "gates": json.load(open(gp)) if os.path.exists(gp) else rec.get('gates'),
                "tables_sha256": {os.path.basename(f): sha(f) for f in
                                  sorted(glob.glob(os.path.join(HERE, f'tables_*{TAG}*.npz')))},
                "step_seconds": {**rec.get('step_seconds', {}), **steps}})
    json.dump(rec, open(p, 'w'), indent=1)
    print(f"wrote exact.json; steps {steps}")


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--check-only' in sys.argv:
        check_only(sys.argv[sys.argv.index('--check-only') + 1])
    elif '--pilot' in sys.argv:
        pilot(sys.argv[sys.argv.index('--pilot') + 1])
    elif '--time-lookup' in sys.argv:
        time_lookup(sys.argv[sys.argv.index('--time-lookup') + 1])
    elif '--run' in sys.argv:
        full_run(sys.argv[sys.argv.index('--run') + 1])
    else:
        print(__doc__)
