#!/usr/bin/env python3
"""Coverage of the paper's two intervals, and the error rate of the rule that needs both, under a
crossed seed-by-block model.  Declared in protocol.json (analysis 'csirs_margins_calib') before it
ran; it touches no recorded output and changes no verdict.

Model.  For seed s and user u in block b(u) of a scenario's 50 m blocks (evaluate.block_ids on that
scenario's users), the paired difference is
    d_su = mu + a_s + b_b(u) + c_s,b(u) + e_su,
a seed effect, a block effect, a seed-by-block interaction and a user term, independent with mean
zero.  With w_b the share of the N users in block b and S = 10 seeds, the estimator D, the mean over
seeds and users, has
    Var(D) = s_a^2/S + s_b^2 sum w^2 + s_c^2 sum w^2 / S + s_e^2/(N S).
Each configuration fixes the share of Var(D) each term carries (normal effects; two configurations
draw the block effects from a t distribution with three degrees of freedom, scaled to the same
variance).  To first order, with k_B = sum w^2 - 2 sum w^3 + (sum w^2)^2, the t interval's squared
standard error s_seed^2/S + SE_block^2 has expectation
    s_a^2/S + s_b^2 k_B + s_c^2 (sum w^2 + k_B)/S + s_e^2 (1/N + k_e)/S,
so it counts the interaction about twice and shrinks the block term by k_B / sum w^2 (1 - 1/B for
equal blocks); the script records that factor per scenario and the simulated mean of SE_t^2/Var(D).

Procedure.  evaluate.paired_bootstrap_hier itself (20,000 replicates), on one period, mu = 0.  Both
intervals are equivariant in mu, so the coverage of mu is the share of replications whose interval
contains 0, and the error rate of the rule that calls a difference established when both 95%
intervals exclude zero is the share in which both do.  2,000 replications per scenario and
configuration (a Monte Carlo standard error of about 0.5 point at 95%), each with its own data and
bootstrap streams (evaluate.stable_seed), so the result does not depend on how work is split.

Reading rule R13: the paper may call the t interval conservative under this model
only if its coverage is at least 94% in every configuration and scenario; otherwise it describes
it as an additional sensitivity criterion and names where coverage falls short.  The rule's error
rate is reported as simulated.  No recorded verdict changes either way.

Self-test (--check-only): with seed effects alone the t interval is exact (coverage 95%), and the
simulated variance of D matches its target; nothing is written.

usage:  python3 calib.py [--check-only] [--workers N]       -> calib.json
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib, json, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate

SCEN = ['city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28',
        'city_5_philadelphia_28', 'city_6_miami_28', 'city_7_sandiego_28']
# shares of Var(D): seed, block, seed-by-block interaction, user; and the block effects' law
CONFIGS = {
    'seed':          ((0.98, 0.01, 0.005, 0.005), 'normal'),
    'seed_block':    ((0.50, 0.49, 0.005, 0.005), 'normal'),
    'block':         ((0.10, 0.88, 0.01, 0.01), 'normal'),
    'block_only':    ((0.01, 0.98, 0.005, 0.005), 'normal'),
    'inter_mixed':   ((0.10, 0.50, 0.39, 0.01), 'normal'),
    'inter':         ((0.05, 0.20, 0.74, 0.01), 'normal'),
    'inter_only':    ((0.005, 0.005, 0.98, 0.01), 'normal'),
    'balanced':      ((0.33, 0.33, 0.33, 0.01), 'normal'),
    'block_t3':      ((0.10, 0.88, 0.01, 0.01), 't3'),
    'block_only_t3': ((0.01, 0.98, 0.005, 0.005), 't3'),
}
N_MC = 2000
CHUNK = 100
S = 10
LEVEL = 0.94          # R13
for _sh, _ in CONFIGS.values():
    assert abs(sum(_sh) - 1) < 1e-12


def blocks_of(name):
    z = np.load(os.path.join(HERE, 'data', f'{name}.npz'))
    b = evaluate.block_ids(z['rx_pos'])
    return b, int(b.max()) + 1


def sigmas(shares, w, n):
    """Component standard deviations giving Var(D) = 1 with the given shares."""
    sa, sb, sc, se = shares
    w2 = float((w ** 2).sum())
    return np.sqrt([sa * S, sb / w2, sc * S / w2, se * n * S])


def replicate(name, cfg, rep, blocks, nb, sig, law):
    rng = np.random.default_rng(evaluate.stable_seed('calib', name, cfg, rep))
    a = sig[0] * rng.standard_normal(S)
    if law == 't3':
        bb = sig[1] * rng.standard_t(3, nb) / np.sqrt(3.0)
    else:
        bb = sig[1] * rng.standard_normal(nb)
    c = sig[2] * rng.standard_normal((S, nb))
    d = (a[:, None] + bb[blocks][None, :] + c[:, blocks]
         + sig[3] * rng.standard_normal((S, blocks.size)))
    boot = np.random.default_rng(evaluate.stable_seed('calib', 'boot', name, cfg, rep))
    _, agg, D, t = evaluate.paired_bootstrap_hier([d], blocks, nb, boot)
    return (D, agg[0], agg[1], t['ci95_t'][0], t['ci95_t'][1],
            t['seed_sd'] ** 2 / S + t['se_block'] ** 2, ((agg[1] - agg[0]) / 3.92) ** 2)


def work(task):
    name, cfg, lo, hi = task
    blocks, nb = blocks_of(name)
    w = np.bincount(blocks, minlength=nb) / blocks.size
    shares, law = CONFIGS[cfg] if cfg in CONFIGS else (((1.0, 0.0, 0.0, 0.0), 'normal'))
    sig = sigmas(shares, w, blocks.size)
    return name, cfg, lo, [replicate(name, cfg, r, blocks, nb, sig, law) for r in range(lo, hi)]


def summarize(rows):
    a = np.array(rows)
    D, blo, bhi, tlo, thi, se2t, se2b = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    cb = (blo <= 0) & (bhi >= 0)
    ct = (tlo <= 0) & (thi >= 0)
    both = (~cb) & (~ct) & (np.sign(blo) == np.sign(tlo))
    vD = float(D.var(ddof=1))
    return {"n_mc": int(a.shape[0]), "coverage_boot": float(cb.mean()),
            "coverage_t": float(ct.mean()), "error_both": float(both.mean()),
            "var_D_mc": vD, "mean_se2_t_over_var": float(se2t.mean() / vD),
            "mean_se2_boot_over_var": float(se2b.mean() / vD)}


def analytic(w):
    w2, w3 = float((w ** 2).sum()), float((w ** 3).sum())
    kb = w2 - 2 * w3 + w2 ** 2
    return {"B": int(w.size), "sum_w2": w2, "kish_B": 1 / w2, "block_factor": kb / w2,
            "interaction_factor": 1 + kb / w2}


def run(tasks, workers):
    out = {}
    with Pool(workers) as pool:
        for name, cfg, lo, rows in pool.imap_unordered(work, tasks):
            out.setdefault((name, cfg), {})[lo] = rows
    return {k: [r for lo in sorted(v) for r in v[lo]] for k, v in out.items()}


def main():
    check = '--check-only' in sys.argv
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 24
    t0 = time.time()
    if check:
        # seed effects alone: D_s = a_s, SE_block = 0, the t interval is exact
        res = run([('city_2_chicago_28', 'selftest', lo, lo + 50) for lo in range(0, 400, 50)],
                  workers)
        s = summarize(res[('city_2_chicago_28', 'selftest')])
        print(json.dumps(s, indent=1))
        ok = 0.92 <= s['coverage_t'] <= 0.98 and 0.85 <= s['var_D_mc'] <= 1.15
        print("self-test", "holds" if ok else "FAILS", "; nothing written")
        raise SystemExit(0 if ok else 1)
    tasks = [(n, c, lo, lo + CHUNK) for n in SCEN for c in CONFIGS for lo in range(0, N_MC, CHUNK)]
    res = run(tasks, workers)
    out = {"script_sha256": hashlib.sha256(
               open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "n_seeds": S, "n_mc": N_MC, "n_boot": evaluate.N_BOOT, "level_r13": LEVEL,
           "configs": {c: {"shares": dict(zip(('seed', 'block', 'interaction', 'user'), sh)),
                           "block_law": law} for c, (sh, law) in CONFIGS.items()},
           "scenarios": {}}
    for n in SCEN:
        blocks, nb = blocks_of(n)
        w = np.bincount(blocks, minlength=nb) / blocks.size
        out["scenarios"][n] = {"n": int(blocks.size), **analytic(w),
                               "results": {c: summarize(res[(n, c)]) for c in CONFIGS}}
    allr = [r for s_ in out["scenarios"].values() for r in s_["results"].values()]
    out["coverage_t_min"] = min(r['coverage_t'] for r in allr)
    out["coverage_boot_min"] = min(r['coverage_boot'] for r in allr)
    out["error_both_max"] = max(r['error_both'] for r in allr)
    out["t_conservative"] = bool(out["coverage_t_min"] >= LEVEL)
    out["t_short"] = sorted({(n, c) for n, s_ in out["scenarios"].items()
                             for c, r in s_["results"].items() if r['coverage_t'] < LEVEL})
    out["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, 'calib.json')
    assert not os.path.exists(p), "calib.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"t coverage {out['coverage_t_min']:.3f}+, bootstrap {out['coverage_boot_min']:.3f}+, "
          f"both-exclude error at most {out['error_both_max']:.3f}; wrote calib.json in "
          f"{out['seconds']}s")


if __name__ == '__main__':
    main()
