#!/usr/bin/env python3
"""The paper's final tests at the levels Holm's procedure uses, under a crossed seed-by-block
model with each scenario's own 50 m blocks.  Declared in protocol.json (analysis 'calib_holm')
before it ran; it imports calib.py and evaluate.py and edits neither.

The tests.  Section V's rule calls a difference established in a scenario when the seed-by-block
bootstrap interval and the t interval of evaluate.paired_bootstrap_hier both exclude zero on the same
side and the larger p-value max(p_boot, p_t) survives Holm across the six evaluation scenarios, so a
scenario's test runs at one of the levels 0.05/k, k = 1..6.  Per-period comparisons use p_boot alone,
Holm across the 12 periods (levels 0.05/k, k = 1..12).  Rule R12 takes the larger p
of the bootstrap and a delta-method t, Holm across seven scenarios; its binding p is the bootstrap's.

The model.  calib.py's: d_su = a_s + b_b(u) + c_s,b(u) + e_su with S = 10 seeds and mu = 0, at the
shares of Var(D) of its ten configurations and at two corners with block effects alone (normal and
t3).  Both tests are invariant to the scale of d, so the shares and the block law are the only
nuisance parameters.  Seven geometries: the six evaluation scenarios and Los Angeles, the validation
scenario, which R12 also tests.

The simulation.  Per geometry, 20,000 calibration and 10,000 evaluation replications, each a call of
evaluate.paired_bootstrap_hier (20,000 resamples) on one period.  Common random numbers across
configurations: a replication's standard normal and t3 draws and its bootstrap stream are the same
for every configuration, which differs only in the component scales.  The streams come from numpy
SeedSequence spawn keys (geometry, replication, stream), so no two coincide.

Outputs.  calib_holm_null.npz: per geometry and configuration, the calibration replications' null
distributions of max(p_boot, p_t) ('max'), of the both-intervals statistic ('joint': max(p_boot,
p_t) when both 95% intervals exclude zero on the same side, else infinity) and of p_boot ('boot'), as
counts on calib_cal.grid().  calib_holm.json: the sizes at Holm's levels with binomial Monte Carlo
standard errors; the strong family-wise bound Holm's proof gives the recorded six-scenario rule, the
largest over k of the sum of the k largest scenario sizes at 0.05/k, each scenario's size the largest
over the configurations; the same for the calibrated rule of R14 on the evaluation replications; the
per-period bound, the largest over k of k times the size at 0.05/k; R12's calibrated rule on the
evaluation replications; and the gates.

Gates, before anything is written.  (1) With calib.py's own streams this file's generator reproduces
calib.json's summaries exactly in every scenario and configuration.  (2) Each geometry's users and
blocks equal the stored runs'.  (3) Every R12 contrast's seed share of variance has a 99.9% upper
bound below 10%.  (4) The counts are nondecreasing, the 'max' and 'boot' counts reach N at the last
grid point, and the calibration replications' 95% coverages agree with calib.json's within four
standard errors.

usage:  python3 calib_holm.py [--check-only] [--workers N]  -> calib_holm.json, calib_holm_null.npz
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib, json, sys, time
from multiprocessing import Pool
import numpy as np
import scipy
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate, calib, calib_cal

ALPHA = 0.05
GEOMS = ['city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28', 'city_5_philadelphia_28',
         'city_6_miami_28', 'city_7_sandiego_28', 'city_1_losangeles_28']
EVAL6 = GEOMS[:6]
CONFIGS = dict(calib.CONFIGS)
CONFIGS['block_pure'] = ((0.0, 1.0, 0.0, 0.0), 'normal')
CONFIGS['block_pure_t3'] = ((0.0, 1.0, 0.0, 0.0), 't3')
CFG = list(CONFIGS)
LOW_SEED = [c for c in CFG if CONFIGS[c][0][0] <= 0.10]       # rule R12's configurations
R12_BETA = 0.001
N_CAL, N_EVAL = 20000, 10000
CHUNK = 100
KMAX = {'six': 6, 'r12': 7, 'period': 12}
ENTROPY = 0x67e447e4ebfa595fef61c11b61452a1d    # the fixed seed of every calibration stream
OUT_JSON, OUT_NPZ = calib_cal.SUMMARY, calib_cal.NULL
assert len(CFG) == 12 and LOW_SEED == ['block', 'block_only', 'inter_mixed', 'inter', 'inter_only',
                                      'block_t3', 'block_only_t3', 'block_pure', 'block_pure_t3']


def assemble(sig, law, blocks, za, zb_n, zb_t, zc, ze):
    """calib.replicate's data from standard draws, with its operations in its order."""
    a = sig[0] * za
    bb = sig[1] * zb_t / np.sqrt(3.0) if law == 't3' else sig[1] * zb_n
    c = sig[2] * zc
    return a[:, None] + bb[blocks][None, :] + c[:, blocks] + sig[3] * ze


def gate_work(task):
    """Gate 1: calib.py's replications, on its own streams, through assemble()."""
    name, cfg, lo, hi = task
    blocks, nb = calib.blocks_of(name)
    w = np.bincount(blocks, minlength=nb) / blocks.size
    shares, law = calib.CONFIGS[cfg]
    sig = calib.sigmas(shares, w, blocks.size)
    out = []
    for r in range(lo, hi):
        rng = np.random.default_rng(evaluate.stable_seed('calib', name, cfg, r))
        za = rng.standard_normal(calib.S)
        zb_n, zb_t = (None, rng.standard_t(3, nb)) if law == 't3' else (rng.standard_normal(nb), None)
        zc = rng.standard_normal((calib.S, nb))
        ze = rng.standard_normal((calib.S, blocks.size))
        d = assemble(sig, law, blocks, za, zb_n, zb_t, zc, ze)
        boot = np.random.default_rng(evaluate.stable_seed('calib', 'boot', name, cfg, r))
        _, agg, D, t = evaluate.paired_bootstrap_hier([d], blocks, nb, boot)
        out.append((D, agg[0], agg[1], t['ci95_t'][0], t['ci95_t'][1],
                    t['seed_sd'] ** 2 / calib.S + t['se_block'] ** 2,
                    ((agg[1] - agg[0]) / 3.92) ** 2))
    return name, cfg, lo, out


def main_work(task):
    """One chunk of replications in one geometry, every configuration on common random numbers.
    Per replication and configuration: p_boot, p_t, both (the two 95% intervals exclude zero on
    the same side), and whether each interval covers zero."""
    gi, lo, hi = task
    blocks, nb = calib.blocks_of(GEOMS[gi])
    n = blocks.size
    w = np.bincount(blocks, minlength=nb) / n
    sigs = [calib.sigmas(CONFIGS[c][0], w, n) for c in CFG]
    res = np.empty((hi - lo, len(CFG), 5))
    for j, r in enumerate(range(lo, hi)):
        rng = np.random.default_rng(np.random.SeedSequence(ENTROPY, spawn_key=(gi, r, 0)))
        za = rng.standard_normal(calib.S)
        zb_n = rng.standard_normal(nb)
        zb_t = rng.standard_t(3, nb)
        zc = rng.standard_normal((calib.S, nb))
        ze = rng.standard_normal((calib.S, n))
        for ci, c in enumerate(CFG):
            d = assemble(sigs[ci], CONFIGS[c][1], blocks, za, zb_n, zb_t, zc, ze)
            boot = np.random.default_rng(np.random.SeedSequence(ENTROPY, spawn_key=(gi, r, 1)))
            _, agg, D, t = evaluate.paired_bootstrap_hier([d], blocks, nb, boot)
            blo, bhi, pb = agg
            tlo, thi = t['ci95_t']
            cb, ct = (blo <= 0 <= bhi), (tlo <= 0 <= thi)
            both = (not cb) and (not ct) and (np.sign(blo) == np.sign(tlo))
            res[j, ci] = (pb, t['p_t'], float(both), float(cb), float(ct))
    return gi, lo, res


def gate1(workers):
    """calib.json reproduced exactly by assemble() on calib.py's streams."""
    CJ = json.load(open(os.path.join(HERE, 'calib.json')))
    tasks = [(nm, c, lo, min(lo + calib.CHUNK, calib.N_MC)) for nm in calib.SCEN
             for c in calib.CONFIGS for lo in range(0, calib.N_MC, calib.CHUNK)]
    out = {}
    with Pool(workers) as pool:
        for nm, c, lo, rows in pool.imap_unordered(gate_work, tasks):
            out.setdefault((nm, c), {})[lo] = rows
    bad = []
    for (nm, c), v in out.items():
        got = calib.summarize([r for lo in sorted(v) for r in v[lo]])
        if got != CJ['scenarios'][nm]['results'][c]:
            bad.append((nm, c))
    assert len(out) == len(calib.SCEN) * len(calib.CONFIGS) and not bad, \
        f"gate 1: calib.json not reproduced in {bad[:5]}"
    return CJ


def gate2():
    """Each geometry's users and blocks equal the stored runs'."""
    stored = {}
    for f in ('results_rx.json', 'heldout_rx.json', 'validation_rx.json'):
        for s in json.load(open(os.path.join(HERE, f)))['sites']:
            stored[s['site']] = (s['n'], s['n_blocks'])
    geo = {}
    for nm in GEOMS:
        blocks, nb = calib.blocks_of(nm)
        w = np.bincount(blocks, minlength=nb) / blocks.size
        assert (blocks.size, nb) == tuple(stored[nm]), (nm, blocks.size, nb, stored[nm])
        geo[nm] = {"n": int(blocks.size), "blocks": int(nb), "kish_blocks": float(1 / (w ** 2).sum()),
                   "role": "evaluation" if nm in EVAL6 else "validation (rule R12 only)"}
    return geo


def gate3():
    """Every R12 contrast's seed share of variance: a 99.9% upper bound on the seed variance of
    the per-seed contrast, which is the mean genie rate times (eta_s - threshold), over itself plus
    the bootstrap variance of the contrast."""
    C = json.load(open(os.path.join(HERE, 'csirs.json')))
    S = len(C['seeds'])
    chi = stats.chi2.ppf(0.001, S - 1)
    worst, n = 0.0, 0
    for cf in C['configs'].values():
        for ps in cf['per_scenario'].values():
            at = ps['at_threshold']
            gbar = abs(at['diff'] / at['eta_minus_thr'])
            sd = gbar * float(np.std(ps['eta_seeds'], ddof=1))
            v_seed = (S - 1) * sd ** 2 / chi / S
            v_boot = ((at['ci95'][1] - at['ci95'][0]) / 3.92) ** 2
            worst, n = max(worst, v_seed / (v_seed + v_boot)), n + 1
    assert n == len(C['configs']) * len(C['scenarios']) and worst < 0.10, (n, worst)
    return worst


def sizes(P, levels):
    """Share of replications at or below each level, and its binomial standard error."""
    F = np.array([(P <= a).mean() for a in levels])
    return F, np.sqrt(F * (1 - F) / P.size)


def holm_bound(F):
    """Holm's strong family-wise bound for tests of sizes F[j][k-1] at 0.05/k: the largest over k of
    the sum of the k largest F[j][k-1].  Returns (bound, k, the scenarios in the sum)."""
    F = np.asarray(F)
    best = (0.0, 0, [])
    for k in range(1, F.shape[0] + 1):
        top = np.argsort(-F[:, k - 1])[:k]
        b = float(F[top, k - 1].sum())
        if b > best[0]:
            best = (b, k, [int(j) for j in top])
    return best


def main():
    check = '--check-only' in sys.argv
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 24
    if not check:
        for f in (OUT_JSON, OUT_NPZ):
            assert not os.path.exists(os.path.join(HERE, f)), f"{f} exists: it is not rewritten"
    t0 = time.time()
    geo = gate2()
    r12_share = gate3()
    print(f"gates 2-3 hold: geometries as stored; R12 seed share at most {r12_share:.2e}")
    CJ = gate1(workers)
    print(f"gate 1 holds: calib.json reproduced exactly ({time.time() - t0:.0f} s)")
    if check:
        # smoke test of the main generator: two replications in Chicago, every configuration
        gi, lo, res = main_work((0, 0, 2))
        assert res.shape == (2, len(CFG), 5) and np.all((res[:, :, :2] > 0) & (res[:, :, :2] <= 1))
        print("check only: the main generator runs; nothing written")
        return
    N = N_CAL + N_EVAL
    tasks = [(gi, lo, lo + CHUNK) for lo in range(0, N, CHUNK) for gi in range(len(GEOMS))]
    R = np.full((len(GEOMS), N, len(CFG), 5), np.nan)
    done = 0
    with Pool(workers) as pool:
        for gi, lo, res in pool.imap_unordered(main_work, tasks):
            R[gi, lo:lo + res.shape[0]] = res
            done += 1
            if done % 70 == 0:
                print(f"  {done}/{len(tasks)} chunks, {time.time() - t0:.0f} s", flush=True)
    assert not np.isnan(R).any()
    g = calib_cal.grid()
    cal, ev = R[:, :N_CAL], R[:, N_CAL:]

    def stat(X, s):
        pb, pt, both = X[..., 0], X[..., 1], X[..., 2] > 0.5
        if s == 'boot':
            return pb
        P = np.maximum(pb, pt)
        return P if s == 'max' else np.where(both, P, np.inf)

    cnt = {s: np.stack([np.stack([calib_cal.counts(stat(cal[gi, :, ci], s), g)
                                  for ci in range(len(CFG))]) for gi in range(len(GEOMS))])
           for s in calib_cal.STATS}
    # gate 4: the counts, and the model against calib.json at the 95% level
    for s in calib_cal.STATS:
        assert np.all(np.diff(cnt[s], axis=-1) >= 0), s
    assert np.all(cnt['max'][..., -1] == N_CAL) and np.all(cnt['boot'][..., -1] == N_CAL)
    agree = []
    for gi, nm in enumerate(EVAL6):
        for c in calib.CONFIGS:
            ci = CFG.index(c)
            ref = CJ['scenarios'][nm]['results'][c]
            for key, col in (('coverage_boot', 3), ('coverage_t', 4)):
                p1, p0 = cal[gi, :, ci, col].mean(), ref[key]
                se = np.sqrt(p0 * (1 - p0) * (1 / N_CAL + 1 / ref['n_mc']))
                agree.append(abs(p1 - p0) / max(se, 1e-9))
    assert max(agree) < 4, f"gate 4: coverage disagrees with calib.json by {max(agree):.1f} SE"

    L6 = [ALPHA / k for k in range(1, KMAX['six'] + 1)]
    L7 = [ALPHA / k for k in range(1, KMAX['r12'] + 1)]
    L12 = [ALPHA / k for k in range(1, KMAX['period'] + 1)]
    cells = {}
    for gi, nm in enumerate(GEOMS):
        cells[nm] = {}
        for ci, c in enumerate(CFG):
            X = cal[gi, :, ci]
            Fj, Sj = sizes(stat(X, 'joint'), L7)
            Fb, Sb = sizes(stat(X, 'boot'), L12)
            cells[nm][c] = {
                "joint_size": Fj.tolist(), "joint_se": Sj.tolist(),
                "boot_size": Fb.tolist(), "boot_se": Sb.tolist(),
                "coverage_boot": float(X[:, 3].mean()), "coverage_t": float(X[:, 4].mean()),
                "error_both": float((X[:, 2] > 0.5).mean()),
                "period_bound": float(max(k * Fb[k - 1] for k in range(1, 13)))}
    # the recorded six-scenario rule: per configuration, and with each scenario's worst
    # configuration inside the sum
    rec = {}
    for c in CFG:
        F = [cells[nm][c]['joint_size'][:6] for nm in EVAL6]
        b, k, top = holm_bound(F)
        se = float(np.sqrt(sum(F[j][k - 1] * (1 - F[j][k - 1]) for j in top) / N_CAL)) if k else 0.0
        rec[c] = {"bound": b, "k": k, "se": se}
    Fw = [[max(cells[nm][c]['joint_size'][k] for c in CFG) for k in range(6)] for nm in EVAL6]
    bw, kw, topw = holm_bound(Fw)
    # R14's calibrated rule on the evaluation replications
    cal_rule, r12_rule = {}, {}
    Fc_all, Fr_all = {}, {}
    for gi, nm in enumerate(GEOMS):
        Fc_all[nm], Fr_all[nm] = {}, {}
        cm = [cnt['max'][gi, ci] for ci in range(len(CFG))]
        cb = [cnt['boot'][gi, CFG.index(c)] for c in LOW_SEED]
        for ci, c in enumerate(CFG):
            X = ev[gi, :, ci]
            both = X[:, 2] > 0.5
            pt = calib_cal.calibrated_from(cm, g, N_CAL, stat(X, 'max'))
            Fc_all[nm][c] = [float(((pt <= a) & both).mean()) for a in L6]
            if c in LOW_SEED:
                pr = calib_cal.calibrated_from(cb, g, N_CAL, stat(X, 'boot'), beta=R12_BETA)
                Fr_all[nm][c] = [float((pr <= a).mean()) for a in L7]
    Fc = [[max(Fc_all[nm][c][k] for c in CFG) for k in range(6)] for nm in EVAL6]
    bc, kc, topc = holm_bound(Fc)
    cal_rule = {"bound": bc, "k": kc,
                "se": float(np.sqrt(sum(Fc[j][kc - 1] * (1 - Fc[j][kc - 1]) for j in topc) / N_EVAL))
                if kc else 0.0,
                "max_size_over_level": float(max(Fc_all[nm][c][k] / L6[k] for nm in EVAL6
                                                 for c in CFG for k in range(6)))}
    Fr = [[max(Fr_all[nm][c][k] for c in LOW_SEED) for k in range(7)] for nm in GEOMS]
    br, kr, topr = holm_bound(Fr)
    Fr_rec = [[max(cells[nm][c]['boot_size'][k] for c in LOW_SEED) for k in range(7)] for nm in GEOMS]
    brr, krr, _ = holm_bound(Fr_rec)
    r12_rule = {"recorded_bound": brr, "recorded_k": krr, "calibrated_bound": br, "calibrated_k": kr,
                "beta": R12_BETA, "configs": LOW_SEED}
    np.savez_compressed(os.path.join(HERE, OUT_NPZ), grid=g, n_cal=N_CAL,
                        geoms=np.array(GEOMS), configs=np.array(CFG),
                        counts_max=cnt['max'], counts_joint=cnt['joint'], counts_boot=cnt['boot'])
    sha = lambda f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
    out = {
        "script_sha256": sha('calib_holm.py'), "cal_sha256": sha('calib_cal.py'),
        "null_npz_sha256": sha(OUT_NPZ), "numpy": np.__version__, "scipy": scipy.__version__,
        "alpha": ALPHA, "n_seeds": calib.S, "n_boot": evaluate.N_BOOT, "n_cal": N_CAL,
        "n_eval": N_EVAL, "levels": {"six": L6, "r12": L7, "period": L12},
        "geometries": geo,
        "configs": {c: {"shares": list(CONFIGS[c][0]), "block_law": CONFIGS[c][1]} for c in CFG},
        "low_seed_configs": LOW_SEED, "r12_beta": R12_BETA,
        "cells": cells,
        "recorded_rule": {"by_config": rec, "worst_inside": {"bound": bw, "k": kw,
                                                            "scenarios": [EVAL6[j] for j in topw]}},
        "calibrated_rule_eval": cal_rule, "r12_rule_eval": r12_rule,
        "gates": {"calib_json_reproduced": True, "geometries_as_stored": True,
                  "r12_seed_share_max_upper": r12_share, "coverage_agreement_max_se": float(max(agree))},
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(out, open(os.path.join(HERE, OUT_JSON), 'w'), indent=1)
    print(f"wrote {OUT_JSON} and {OUT_NPZ}: recorded rule's bound {bw:.4f} (k = {kw}); "
          f"calibrated {bc:.4f}; R12 recorded {brr:.4f}, calibrated {br:.4f}; "
          f"{out['seconds']:.0f} s")


if __name__ == '__main__':
    main()
