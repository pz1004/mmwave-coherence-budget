#!/usr/bin/env python3
"""The controller re-solved and re-evaluated with the charge for choosing its next
codeword, and Zhang et al.'s descent as published beside the reimplementation.
Declared in protocol.json (analysis 'rerank') before it ran.

WHY.  After an inference call the controller picks each next codeword by a masked
argmax over the exit's 64 scores (methods.scheduled).  No other timed component
contains that work, so without c_rank it is charged nowhere.  The same operation is
timed as `sweep_select` (a masked argmax over 64 entries, 1.060 us on one P-core);
this script charges it on every probe of a decision sequence that invokes the
predictor (sim.elapsed_us), in the planner and in the executor, and re-solves every
controller result.  And the adaptive baseline methods.zhang is a simplified
reimplementation (a one-sided test on clipped reports, four looks, a look-ahead
layer rule); methods.zhang_np2 follows Zhang et al.'s Algorithms 1-2 and eq. (24),
and is evaluated beside it.

Every output carries the tag `_rr`; no recorded file is read for writing, and no
model is retrained or re-timed.

usage:  python3 rerank.py --selftest          unit checks; writes nothing
        python3 rerank.py --check-only DIR    the gates; their outputs go to DIR
        python3 rerank.py --run DIR           the declared sequence; logs in DIR
"""
import glob, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = sys.executable
TAG = '_rr'
VAL = 'city_1_losangeles_28'
HELD = ['city_4_phoenix_28', 'city_5_philadelphia_28', 'city_6_miami_28',
        'city_7_sandiego_28']
P3 = ['--seeds', '17', '29', '43']
RANK = ['--rank-charge']
NP2 = ['--zhang-np2']
FA_DRAWS, FA_SEED = 2_000_000, 23        # the reimplementation's realized false alarms


def outputs():
    """Every file the declared sequence writes."""
    return sorted(set(glob.glob(os.path.join(HERE, f'*{TAG}*')))
                  | {p for p in [os.path.join(HERE, 'rerank.json')] if os.path.exists(p)})


def run(args, log):
    print('  $', ' '.join(args), flush=True)
    with open(log, 'a') as f:
        f.write('$ ' + ' '.join(args) + '\n')
        f.flush()
        subprocess.run([PY, *args], cwd=HERE, check=True, stdout=f, stderr=subprocess.STDOUT)


# ------------------------------------------------------------------ selftest -----
def selftest():
    import torch
    from scipy import stats, integrate
    import sim, methods as M, obs
    # elapsed time: c_rank = 0 gives k t_p + tau_i + (k + i + 1) c_lookup;
    # otherwise + c_rank k 1[i >= 1]
    tau = np.array([0.0, 14.0, 36.0, 65.0])
    for k in (0, 3, 64):
        for i in (0, 1, 3):
            old = k * 20.339 + tau[i] + (k + i + 1) * 0.296
            assert sim.elapsed_us(k, i, tau, 20.339, 0.296) == old
            assert abs(sim.elapsed_us(k, i, tau, 20.339, 0.296, 1.06)
                       - (old + 1.06 * k * (i >= 1))) < 1e-12
    kt, it = torch.tensor([0, 5, 5]), torch.tensor([2, 0, 1])
    got = sim.elapsed_us(kt, it, torch.tensor(tau), 20.339, 0.296, 1.06)
    want = torch.tensor([sim.elapsed_us(int(a), int(b), tau, 20.339, 0.296, 1.06)
                         for a, b in zip(kt, it)], dtype=got.dtype)
    assert torch.allclose(got, want)
    # the likelihood ratio against a numerical mixture over the Rayleigh fade
    def lr_num(S, n, m):
        k = 2 * obs.K_F * n
        g = lambda lam: (stats.ncx2.pdf(S, k, n * lam) * np.exp(-lam / (2 * m)) / (2 * m))
        return np.log(integrate.quad(g, 0, np.inf, limit=400)[0] / stats.chi2.pdf(S, k))
    for S, n, m in ((30., 1, .5), (45., 1, 2.), (80., 2, 1.), (120., 3, .3), (25., 1, 20.),
                    (200., 2, 50.)):
        a = float(M.np2_lnlr(np.array([S]), n, np.array([m]))[0])
        assert abs(a - lr_num(S, n, m)) < 1e-5, (S, n, m, a)
    # the ln 1F1 table against the exact form
    for b in (16., 32., 48.):
        z = 10 ** np.random.default_rng(1).uniform(-9, 7, 4000)
        e, t = M._ln_1f1_exact(b, z), M._ln_1f1_one(b, z)
        assert float(np.max(np.abs(e - t) / np.maximum(1, np.abs(e)))) < 1e-6, b
    # t0 puts a noise-free first look exactly on the upper threshold
    m = np.array([1e-2, 1e-1, 1., 10., 1e3, 1e6])
    t0 = M._np2_t0(m)
    assert np.allclose(M.np2_lnlr(2 * m * t0, 1, m), M.NP2_LNA, atol=1e-6)
    # eq. (24)'s rate against its integral
    for S, t in ((1.0, 0.5), (100.0, 0.01), (0.01, 2.0), (1e4, 1e-3)):
        num = integrate.quad(lambda x: np.log2(1 + S * x) * np.exp(-x), t, np.inf)[0]
        got = float(M._np2_rate(np.array([S]), np.array([t]))[0])
        assert abs(got - num) < 1e-6 * max(1, num), (S, t, got, num)
    # gate (c): the raw first look reproduces y exactly
    import evaluate as E
    site = sim.Site(VAL)
    g = lambda: torch.Generator(device=sim.DEV).manual_seed(E.stable_seed(VAL, 17))
    assert torch.equal(sim.report_of(sim.measure_chi2(site.pbar, g())), sim.measure(site.pbar, g()))
    print("selftest: elapsed time, likelihood ratio, 1F1 table, t0, eq. (24) rate and the "
          "first-look gate all pass")
    print("false alarms of the reimplementation:", false_alarms())


def false_alarms():
    """The reimplementation's (methods.zhang) realized per-look false-alarm rate on
    noise-only looks: its own thresholds, applied to the running mean of clipped
    reports, and, for reference, of unclipped ones."""
    import methods as M, obs
    rng = np.random.default_rng(FA_SEED)
    R = 4
    raw = rng.chisquare(2 * obs.K_F, size=(FA_DRAWS, R)) / (2 * obs.K_F)  # E / sigma_p^2
    thr = np.array([M._chi2_ppf(0.8, 2 * obs.K_F * r) / (2 * obs.K_F * r) - 1
                    for r in range(1, R + 1)])
    out = {"draws": FA_DRAWS, "seed": FA_SEED, "share_clipped": float((raw - 1 < 1e-3).mean())}
    for lab, v in (("clipped", np.maximum(raw - 1, 1e-3)), ("unclipped", raw - 1)):
        mean = np.cumsum(v, 1) / np.arange(1, R + 1)
        per = (mean > thr).mean(0)
        hit = np.zeros(FA_DRAWS, bool)
        for r in range(R):
            hit |= mean[:, r] > thr[r]
        out[lab] = {"per_look": [round(float(x), 4) for x in per],
                    "within_four_looks": round(float(hit.mean()), 4)}
    return out


# ---------------------------------------------------------------- the gates -----
def _leaves(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, f'{path}.{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f'{path}[{i}]')
    else:
        yield path, x


def compare_json(a, b, skip=('seconds', 'code_commit', 'protocol_sha256')):
    # protocol_sha256 records the protocol file as it stood when a file was written, so it is
    # not part of the comparison
    A = {p: v for p, v in _leaves({k: v for k, v in a.items() if k not in skip})}
    B = {p: v for p, v in _leaves({k: v for k, v in b.items() if k not in skip})}
    assert set(A) == set(B), sorted(set(A) ^ set(B))[:10]
    worst, n_diff = 0.0, 0
    for p in A:
        x, y = A[p], B[p]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or x is None:
            assert x == y, (p, x, y)
        elif x != y:
            n_diff += 1
            worst = max(worst, abs(float(x) - float(y)))
    return {"leaves": len(A), "differing": n_diff, "max_abs_dev": worst}


def check_only(d):
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, 'gates.log')
    gates = {}
    # (a) the evaluator, with nothing switched on, on the recorded tables
    out = os.path.join(d, 'validation_gate.json')
    if not os.path.exists(out):
        run(['evaluate.py', '--sites', VAL, '--out', out], log)
    gates['a_validation'] = compare_json(json.load(open(out)),
                                         json.load(open(os.path.join(HERE, 'validation.json'))))
    # (b) a zero-charge re-solve of seed 17 against the recorded table
    run(['sched.py', '--seeds', '17', '--out-suffix', '_gate'], log)
    new, old = (np.load(os.path.join(HERE, f'tables_17{s}.npz')) for s in ('_gate', ''))
    cov = np.load(os.path.join(HERE, 'coverage_17.npz'))
    reach = cov['cnt_off'] > 0                         # (exit, k, b, c): visited cells
    tot = dif = rtot = rdif = 0
    for key in old.files:
        a, b = old[key], new[key]
        assert a.shape == b.shape, key
        if key in ('g_m', 'g_p'):
            continue
        ne = a != b
        tot += a.size; dif += int(ne.sum())
        r = np.broadcast_to(reach[..., None], a.shape)
        rtot += int(r.sum()); rdif += int((ne & r).sum())
    gates['b_resolve_seed17'] = {"entries": tot, "differing": dif,
                                 "reachable_entries": rtot, "reachable_differing": rdif,
                                 "share_reachable_differing": rdif / rtot,
                                 "g_m_max_abs_dev": float(np.max(np.abs(old['g_m'] - new['g_m']))),
                                 "g_p_max_abs_dev": float(np.max(np.abs(old['g_p'] - new['g_p'])))}
    for f in ('tables_17_gate.npz', 'coverage_17_gate.npz', 'scheduler_gate.json'):
        if os.path.exists(os.path.join(HERE, f)):
            shutil.move(os.path.join(HERE, f), os.path.join(d, f))
    json.dump(gates, open(os.path.join(d, 'gates.json'), 'w'), indent=1)
    print(json.dumps(gates, indent=1))


# ------------------------------------------------------ the declared sequence ---
def full_run(d):
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

    step('solve', f'scheduler{TAG}.json',
         lambda: run(['sched.py', *RANK, '--out-suffix', TAG], log))
    ev = ['evaluate.py', *RANK, '--tables-suffix', TAG]
    step('validation', f'validation{TAG}.json',
         lambda: run([*ev, *NP2, '--sites', VAL, '--out', f'validation{TAG}.json'], log))
    step('results', f'results{TAG}.json',
         lambda: run([*ev, *NP2, '--validation', f'validation{TAG}.json',
                      '--out', f'results{TAG}.json'], log))
    step('heldout', f'heldout{TAG}.json',
         lambda: run([*ev, *NP2, '--validation', f'validation{TAG}.json', '--sites', *HELD,
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
    # the sensitivity conditions: sweeps.py's own build() and summarize(), tagged
    import sweeps
    sweeps.TAG, sweeps.EXTRA = TAG, list(RANK)
    for cond in list(sweeps.CONDITIONS) + list(sweeps.PREDQ):
        step(f'sweep {cond}', f'sweep{TAG}_{cond}.json', lambda c=cond: sweeps.build(c))
    step('summary sweeps', f'sweeps{TAG}.json',
         lambda: sweeps.summarize(path=f'sweeps{TAG}.json'))
    step('summary predq', f'predq{TAG}.json',
         lambda: sweeps.summarize(sweeps.PREDQ, f'predq{TAG}.json',
                                  {"q": "oracle-mixed anytime predictor: the rate-optimal "
                                        "beam ranks first for a fraction q of receivers; "
                                        "charges unchanged"}))
    step('rotation', f'selection_stability{TAG}.json',
         lambda: run(['selection_stability.py', '--suffix', TAG], log))
    step('interleave', f'interleave{TAG}.json',
         lambda: run(['interleave.py', '--suffix', TAG], log))
    step('rules', f'rules{TAG}.json', lambda: run(['rules_suffix.py', TAG], log))
    lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    gp = os.path.join(d, 'gates.json')
    rec = {"c_rank_us": ch['sweep_select'], "tag": TAG,
           "gates": json.load(open(gp)) if os.path.exists(gp) else None,
           "false_alarms": false_alarms(), "step_seconds": steps}
    p = os.path.join(HERE, 'rerank.json')
    if os.path.exists(p):              # a resumed run keeps the first run's timings
        old = json.load(open(p))
        old['step_seconds'] = {**old.get('step_seconds', {}), **steps}
        rec = {**rec, 'step_seconds': old['step_seconds']}
    json.dump(rec, open(p, 'w'), indent=1)
    print(f"wrote rerank.json; steps {steps}")


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--check-only' in sys.argv:
        check_only(sys.argv[sys.argv.index('--check-only') + 1])
    elif '--run' in sys.argv:
        full_run(sys.argv[sys.argv.index('--run') + 1])
    else:
        print(__doc__)
