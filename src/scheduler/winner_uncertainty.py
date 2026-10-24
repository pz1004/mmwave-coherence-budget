#!/usr/bin/env python3
"""Selection-aware uncertainty for the changes of the best method.

Table I lists the (scenario, T) pairs at which the best of the ten methods
of the published comparison differs between the published accounting (1),
probe airtime only, and the accounting with inference latency included (2).
Its winners and its rate losses are point estimates:
the mean efficiencies are treated as known, and each winner is selected from
the same estimates that rank it.  This script gives them the uncertainty the
controller's comparisons carry, and decides which changes the paper may state
as established by a rule fixed before it ran (R8, recorded in protocol.json).

DATA.  The per-receiver, per-seed rate of each method in accounting.PUBS
(copied: importing accounting.py writes accounting.json), under (1) and (2), at
every T, in the six evaluation scenarios.  Nothing is trained or timed: the
seed loop of evaluate.py is replicated on the stored checkpoints with its
common random numbers -- stable_seed(site, seed) for the measurements, and for
the published comparator a fresh stable_seed(site, seed, 'r') stream per seed
and per accounting, consumed as evaluate.py consumes it (T in grid order;
AHIBS, AHTBS, AHTBS+AHIBS).  Its decision time c_dec enters its stopping rule,
so under (1) it can stop at another layer, not merely pay less.

REPRODUCTION GATE.  Nothing is resampled unless the recomputed means reproduce
every stored one: (2) against results.json and heldout.json, to their 5
decimals; (1) against results_freecompute.json in the test scenarios, and
against aging.json at 0 km/h (the nine fixed methods) in the held-out ones;
and the nine-method winners against aging.json, the ten-method winners
against accounting.json in the test scenarios.  The published comparator under
(1) in the held-out scenarios has never been computed; its code path is the
one the test scenarios validate.

RESAMPLING.  The paper's seed-by-block bootstrap: 20,000 replicates, each
resampling the 10 seeds once -- shared by the six scenarios, since a seed's
models serve all of them -- and each scenario's 50 m blocks once, shared by
every method, accounting and T.

FAMILY.  Every pairwise difference in mean rate among the ten methods, under
both accountings, at every T, in every scenario: 6,480 contrasts, less those
that are identically zero (both methods infeasible).  Simultaneous 95%
intervals in the two versions Section V of the paper uses:
  (i)  bootstrap: q is the 95% quantile over replicates of
       max_c |D*_c - D_c| / SE*_c; interval D_c +/- q SE*_c;
  (ii) t: SE_t = (s_seed^2/S + SE_block^2)^(1/2), as
       evaluate.paired_bootstrap_hier computes it, and q_t the t_(S-1)
       quantile at the per-comparison level 2(1 - Phi(q)) that q implies;
       interval D_c +/- q_t SE_t.
A difference is established when both intervals exclude zero.

R8.  Under each accounting at (scenario, T), a method is a candidate for best
unless another method's difference over it is established.  A change of the
best method is established where no method is a candidate under both
accountings.  Because the intervals hold simultaneously for every contrast,
they hold for the contrasts the argmax selected (multiple comparisons with the
best).  Replicate frequencies are descriptive and decide nothing.

usage: python3 winner_uncertainty.py [--out winner_uncertainty.json]
       python3 winner_uncertainty.py --check-only    the reproduction gate alone
       python3 winner_uncertainty.py --selftest      the statistics on synthetic data
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
import sim, methods, evaluate, protocol

# accounting.PUBS, in its order (ties go to the first, as accounting.py and
# aging.py break them); the nine fixed methods are aging.PUBS
PUBS = ['sweep64', 'hier6', 'topk8', 'topk16', 'topk32', 'fc16', 'fc32', 'cnn16',
        'cnn32', 'zhang-AHTBS+AHIBS']
ZHANG = 'zhang-AHTBS+AHIBS'
N_FIXED = 9
EVAL = ['city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28',
        'city_5_philadelphia_28', 'city_6_miami_28', 'city_7_sandiego_28']
TEST = EVAL[:2]
ACCTS = ('free', 'joint')          # index 0 = accounting (1), index 1 = accounting (2)
N_BOOT = evaluate.N_BOOT           # 20,000
ALPHA = 0.05
TOL_STORED = 5e-6 + 1e-12          # the stored means are rounded to 5 decimals
TOL_AGING = 1e-5                   # aging.py's own v = 0 regression tolerance


def grid_us():
    return json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']


# ------------------------------------------------------------------ data ----
def scenario_rates(name, seeds, lat, grid):
    """Per-block sums Z[a, m, t, s, b] and per-seed receiver means mu[a, m, t, s] of
    every method's rate, replicating evaluate.py's seed loop line for line."""
    site = sim.Site(name)
    blocks = evaluate.block_ids(site.rx)
    nb = int(blocks.max()) + 1
    A, M, NT, S = len(ACCTS), len(PUBS), len(grid), len(seeds)
    Z = np.zeros((A, M, NT, S, nb))
    mu = np.zeros((A, M, NT, S))
    q = {a: evaluate.charges(lat, free_compute=(a == 'free')) for a in ACCTS}
    for si, seed in enumerate(seeds):
        ee, nets = evaluate.load_nets(seed)
        gen = torch.Generator(device=sim.DEV).manual_seed(evaluate.stable_seed(name, seed))
        y = sim.measure(site.pbar, gen)                  # evaluate.py:341
        c_ref = None
        for ai, a in enumerate(ACCTS):
            ch, tp, cdec = q[a]['ch'], q[a]['T_PROBE'], q[a]['C_DEC']
            runs = {}
            for k in (1, 2, 4, 8, 16, 32, 64):           # evaluate.py:348-357
                runs['sweep64' if k == 64 else f'topk{k}'] = methods.topk(site, y, k, tp)
            for L in (3, 4, 5, 6):
                runs[f'hier{L}'] = methods.hierarchical(site, y, L, tp)
            for k in (4, 8, 16, 32, 64):
                runs[f'fc{k}'] = methods.fixed_net(site, y, k, 'fc', nets[f'fc{k}'],
                                                   ch[f'fc_k{k}'], tp)
                runs[f'cnn{k}'] = methods.fixed_net(site, y, k, 'cnn', nets[f'cnn{k}'],
                                                    ch['cnn'], tp)
            # the fixed methods choose the same beam under both accountings
            cs = {m: runs[m][1] for m in PUBS[:N_FIXED]}
            if c_ref is None:
                c_ref = cs
            else:
                assert all(torch.equal(cs[m], c_ref[m]) for m in cs), (name, seed)
            gen2 = torch.Generator(device=sim.DEV).manual_seed(
                evaluate.stable_seed(name, seed, 'r'))   # one stream per accounting
            for T in grid:                               # evaluate.py:359-367
                for nm, (al, asl) in (('zhang-AHIBS', (True, False)),
                                      ('zhang-AHTBS', (False, True)),
                                      (ZHANG, (True, True))):
                    d, c, _, _ = methods.zhang(site, y, float(T), tp, cdec, al, asl, gen=gen2)
                    runs[(nm, T)] = (d, c)
            for mi, m in enumerate(PUBS):
                for ti, T in enumerate(grid):
                    d, c = runs[(m, T)] if m == ZHANG else runs[m]
                    se = site.se.gather(1, c[:, None])[:, 0].double()   # evaluate.py:399
                    r = (torch.clamp(1 - d / T, min=0.0) * se).cpu().numpy()
                    mu[ai, mi, ti, si] = r.mean()
                    Z[ai, mi, ti, si] = np.bincount(blocks, weights=r, minlength=nb)
        print(f"  [{name} seed {seed}] done", flush=True)
    return {"Z": Z, "mu": mu, "counts": np.bincount(blocks, minlength=nb).astype(float),
            "n": site.n, "n_blocks": nb}


def point_winners(Rhat, n_methods=None):
    """Rhat (A, M, NT) -> winners (A, NT), first maximum in PUBS order."""
    R = Rhat if n_methods is None else Rhat[:, :n_methods]
    return R.argmax(1)


def gate(arrs, grid):
    """The reproduction gate.  Returns (report, holds)."""
    ref = {s['site']: s for f in ('results.json', 'heldout.json')
           for s in json.load(open(os.path.join(HERE, f)))['sites']}
    rf = {s['site']: s for s in
          json.load(open(os.path.join(HERE, 'results_freecompute.json')))['sites']}
    ag = {r['site']: r for r in json.load(open(os.path.join(HERE, 'aging.json')))['rows']
          if r['speed_kmh'] == 0.0}
    acc = {s['site']: s for s in json.load(open(os.path.join(HERE, 'accounting.json')))['sites']}
    rep, ok = [], True
    for name, x in arrs.items():
        Rhat = x['mu'].mean(-1)                          # (A, M, NT): seeds, then users
        jj = max(abs(Rhat[1, mi, ti] - ref[name]['methods'][m]['per_T'][str(T)]['mean'])
                 for mi, m in enumerate(PUBS) for ti, T in enumerate(grid))
        aj = max(abs(Rhat[1, mi, ti] - ag[name]['methods'][m]['joint'][ti])
                 for mi, m in enumerate(PUBS[:N_FIXED]) for ti in range(len(grid)))
        af = max(abs(Rhat[0, mi, ti] - ag[name]['methods'][m]['free'][ti])
                 for mi, m in enumerate(PUBS[:N_FIXED]) for ti in range(len(grid)))
        ff = (max(abs(Rhat[0, mi, ti] - rf[name]['methods'][m]['per_T'][str(T)]['mean'])
                  for mi, m in enumerate(PUBS) for ti, T in enumerate(grid))
              if name in rf else None)
        w9 = point_winners(Rhat, N_FIXED)
        ch9 = [{"T_us": T, "free": PUBS[w9[0, ti]], "joint": PUBS[w9[1, ti]]}
               for ti, T in enumerate(grid) if w9[0, ti] != w9[1, ti]]
        win9 = ch9 == ag[name]['accounting']['winner_changes']
        win10 = None
        if name in acc:
            w10 = point_winners(Rhat)
            ch10 = [{"T_ms": T / 1000, "free": PUBS[w10[0, ti]], "joint": PUBS[w10[1, ti]]}
                    for ti, T in enumerate(grid) if w10[0, ti] != w10[1, ti]]
            win10 = ch10 == acc[name]['winner_changes']
        holds = (jj <= TOL_STORED and aj <= TOL_AGING and af <= TOL_AGING and win9
                 and (ff is None or ff <= TOL_STORED) and win10 is not False)
        ok &= holds
        rep.append({"site": name, "max_abs_joint_vs_stored": jj,
                    "max_abs_free_vs_results_freecompute": ff,
                    "max_abs_joint_vs_aging": aj, "max_abs_free_vs_aging": af,
                    "nine_method_winners_equal_aging": win9,
                    "ten_method_winners_equal_accounting": win10, "holds": bool(holds)})
    return rep, bool(ok)


# ------------------------------------------------------------ statistics ----
def analyze(arrs, n_boot=N_BOOT, tag='winner', alpha=ALPHA):
    """R8 on per-scenario arrays Z (A, M, NT, S, B), mu (A, M, NT, S), counts (B,).

    Returns the family's critical values and, per scenario, the point
    estimates, the established flags, the candidate sets, the changes and the
    descriptive replicate shares.  Generic in M, NT and S, so --selftest runs
    exactly this code on synthetic data."""
    names = list(arrs)
    A, M, NT, S, _ = arrs[names[0]]['Z'].shape
    I, J = np.triu_indices(M, 1)
    rows = np.arange(n_boot)[:, None]
    sidx = np.random.default_rng(evaluate.stable_seed(tag, 'seeds')).integers(
        0, S, size=(n_boot, S))
    mseed = np.zeros((n_boot, S)); np.add.at(mseed, (rows, sidx), 1.0)
    tq = stats.t.ppf(1 - alpha / 2, S - 1)
    sc, tmax = {}, []
    for nm in names:
        Z, mu, c = arrs[nm]['Z'], arrs[nm]['mu'], arrs[nm]['counts']
        B = Z.shape[-1]
        X = A * M * NT
        pick = np.random.default_rng(evaluate.stable_seed(tag, 'blocks', nm)).integers(
            0, B, size=(n_boot, B))
        w = np.zeros((n_boot, B)); np.add.at(w, (rows, pick), 1.0)
        den = w @ c                                              # receivers drawn
        om = (mseed[:, :, None] * w[:, None, :]).reshape(n_boot, S * B)
        R = ((om @ Z.reshape(X, S * B).T) / (S * den[:, None])).reshape(n_boot, A, M, NT)
        Rb = ((w @ Z.sum(3).reshape(X, B).T) / (S * den[:, None])).reshape(n_boot, A, M, NT)
        Rhat = mu.mean(-1)                                       # (A, M, NT)
        Dhat = Rhat[:, I] - Rhat[:, J]                           # (A, P, NT)
        Dst = R[:, :, I] - R[:, :, J]                            # (N, A, P, NT)
        Dbl = Rb[:, :, I] - Rb[:, :, J]
        se = Dst.std(0, ddof=1)
        seb = Dbl.std(0, ddof=1)
        s_seed = (mu[:, I] - mu[:, J]).std(-1, ddof=1)           # (A, P, NT)
        se_t = np.sqrt(s_seed ** 2 / S + seb ** 2)
        zero = (se == 0)
        assert np.all(Dhat[zero] == 0) and np.all(Dst[:, zero] == 0), \
            f"{nm}: a contrast with no variation is not identically zero"
        dev = np.abs(Dst - Dhat[None]) / np.where(zero, np.inf, se)[None]
        tmax.append(dev.reshape(n_boot, -1).max(1))
        wins = R.argmax(2)                                       # (N, A, NT)
        sc[nm] = dict(R=R, Rhat=Rhat, Dhat=Dhat, Dst=Dst, se=se, se_t=se_t,
                      s_seed=s_seed, seb=seb, zero=zero, wins=wins, n_blocks=B)
    T_ = np.max(np.stack(tmax), 0)
    q = float(np.quantile(T_, 1 - alpha))
    lvl = float(2 * (1 - stats.norm.cdf(q)))
    q_t = float(stats.t.ppf(1 - lvl / 2, S - 1))
    out = {"n_boot": n_boot, "alpha": alpha, "n_seeds": S, "q": q, "q_t": q_t,
           "per_comparison_level": lvl, "t_quantile_marginal": float(tq),
           "n_contrasts": int(sum(x['zero'].size for x in sc.values())),
           "n_identically_zero": int(sum(x['zero'].sum() for x in sc.values())),
           "scenarios": {}}
    n_change = np.zeros(n_boot, dtype=int)
    for nm, x in sc.items():
        Dhat, se, se_t, zero = x['Dhat'], x['se'], x['se_t'], x['zero']
        pos = ~zero & (Dhat - q * se > 0) & (Dhat - q_t * se_t > 0)
        neg = ~zero & (Dhat + q * se < 0) & (Dhat + q_t * se_t < 0)
        cand = np.ones((A, M, NT), dtype=bool)
        for p, (i, j) in enumerate(zip(I, J)):
            cand[:, j][pos[:, p]] = False                        # i beats j
            cand[:, i][neg[:, p]] = False                        # j beats i
        W = point_winners(x['Rhat'])                             # (A, NT)
        wins = x['wins']
        rep_change = wins[:, 0] != wins[:, 1]                    # (N, NT)
        n_change += rep_change.sum(1)
        pidx = {(i, j): p for p, (i, j) in enumerate(zip(I, J))}

        def contrast(a, hi, lo, t):
            """D = R_a(hi) - R_a(lo) at t, with its intervals, oriented hi minus lo."""
            p = pidx[(min(hi, lo), max(hi, lo))]
            sg = 1.0 if hi < lo else -1.0
            d, s1, s2 = sg * Dhat[a, p, t], se[a, p, t], se_t[a, p, t]
            reps = sg * x['Dst'][:, a, p, t]
            return {"diff": float(d), "se_boot": float(s1), "se_t": float(s2),
                    "s_seed": float(x['s_seed'][a, p, t]), "se_block": float(x['seb'][a, p, t]),
                    "ci95_boot": [float(v) for v in np.percentile(reps, [2.5, 97.5])],
                    "ci95_t": [float(d - tq * s2), float(d + tq * s2)],
                    "sim_boot": [float(d - q * s1), float(d + q * s1)],
                    "sim_t": [float(d - q_t * s2), float(d + q_t * s2)],
                    "established": bool(pos[a, p, t] if sg > 0 else neg[a, p, t]),
                    "_reps": reps}

        per_T, changes = [], []
        for t in range(NT):
            c1, c2 = cand[0, :, t], cand[1, :, t]
            est = not bool((c1 & c2).any())
            rec = {"t": t, "winner_1": int(W[0, t]), "winner_2": int(W[1, t]),
                   "change": bool(W[0, t] != W[1, t]),
                   "candidates_1": [int(m) for m in np.flatnonzero(c1)],
                   "candidates_2": [int(m) for m in np.flatnonzero(c2)],
                   "established": est,
                   "share_any_change": float(rep_change[:, t].mean())}
            assert not est or rec["change"], "an established change must be a point-estimate change"
            if rec["change"]:
                w1, w2 = int(W[0, t]), int(W[1, t])
                k2 = contrast(1, w2, w1, t)          # what keeping (1)'s choice costs under (2)
                k1 = contrast(0, w1, w2, t)          # (1)'s preference for its choice
                r2 = float(x['Rhat'][1, w2, t])
                lost = k2['diff'] / r2
                rec["share_same_change"] = float(((wins[:, 0, t] == w1)
                                                  & (wins[:, 1, t] == w2)).mean())
                changes.append({
                    "t": t, "winner_1": w1, "winner_2": w2, "established": est,
                    "rate_lost": lost,
                    "rate_lost_ci95_boot": [float(v) for v in
                                            np.percentile(k2['_reps'] / r2, [2.5, 97.5])],
                    "rate_lost_ci95_t": [k2['ci95_t'][0] / r2, k2['ci95_t'][1] / r2],
                    "rate_lost_sim_lower": min(k2['sim_boot'][0], k2['sim_t'][0]) / r2,
                    "best_rate_2": r2,
                    "contrast_2": {k: v for k, v in k2.items() if k != '_reps'},
                    "contrast_1": {k: v for k, v in k1.items() if k != '_reps'},
                    "both_contrasts_established": bool(k2['established'] and k1['established']),
                    "candidates_1": rec["candidates_1"], "candidates_2": rec["candidates_2"],
                    "share_same_change": rec["share_same_change"],
                    "share_any_change": rec["share_any_change"]})
            per_T.append(rec)
        out["scenarios"][nm] = {
            "n_blocks": x['n_blocks'], "per_T": per_T, "changes": changes,
            "n_established_contrasts": int(pos.sum() + neg.sum()),
            "n_contrasts": int((~zero).sum())}
    out["n_changes_replicates"] = {
        "mean": float(n_change.mean()), "median": float(np.median(n_change)),
        "p2_5": float(np.percentile(n_change, 2.5)),
        "p97_5": float(np.percentile(n_change, 97.5)),
        "hist": {str(int(k)): int(v) for k, v in zip(*np.unique(n_change, return_counts=True))}}
    return out


# -------------------------------------------------------------- selftest ----
def selftest():
    """The statistics on synthetic data with a known answer.  A planted change
    with wide margins must be established; a planted near-tie must not; a
    scenario with one clear winner under both accountings shows no change."""
    rng = np.random.default_rng(7)
    S, B, NT, A = 10, 16, 3, 2
    nb = 60
    n = B * nb
    blocks = np.repeat(np.arange(B), nb)

    def make(base, seed_sd, rx_sd):
        """base (A, M, NT) -> per-receiver rates with seed and receiver noise."""
        M = base.shape[1]
        Z = np.zeros((A, M, NT, S, B)); mu = np.zeros((A, M, NT, S))
        for s in range(S):
            se = rng.normal(0, seed_sd, size=(M,))               # a seed shifts a method
            rx = rng.normal(0, rx_sd, size=(M, n))               # shared by both accountings
            for a in range(A):
                for t in range(NT):
                    r = base[a, :, t, None] + se[:, None] + rx
                    mu[a, :, t, s] = r.mean(1)
                    for m in range(M):
                        Z[a, m, t, s] = np.bincount(blocks, weights=r[m], minlength=B)
        return {"Z": Z, "mu": mu, "counts": np.full(B, float(nb)), "n": n, "n_blocks": B}

    clear = np.array([[[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],      # (1)
                      [[3.0, 2.0, 3.0], [2.0, 3.0, 2.0], [1.0, 1.0, 1.0]]])     # (2)
    # a near-tie as the latency makes one: (2) lowers method 0 by a fixed amount at
    # t = 1, so the two methods' order flips by margins of 0.02 against noise of
    # about 0.01 in their difference -- a point-estimate change, not an established one
    tie = np.array([[[3.00, 3.00, 3.0], [2.98, 2.98, 2.0], [1.0, 1.0, 1.0]],
                    [[3.00, 2.96, 3.0], [2.98, 2.98, 2.0], [1.0, 1.0, 1.0]]])
    arrs = {"planted": make(clear, 0.01, 0.05), "near_tie": make(tie, 0.02, 0.1)}
    out = analyze(arrs, n_boot=4000, tag='winner-selftest')
    p, t = out["scenarios"]["planted"], out["scenarios"]["near_tie"]
    assert [r["change"] for r in p["per_T"]] == [False, True, False], p["per_T"]
    assert p["per_T"][1]["established"] and len(p["changes"]) == 1
    ch = p["changes"][0]
    assert (ch["winner_1"], ch["winner_2"]) == (0, 1) and ch["both_contrasts_established"]
    assert abs(ch["rate_lost"] - 1 / 3) < 0.02, ch["rate_lost"]
    assert ch["rate_lost_ci95_boot"][0] < ch["rate_lost"] < ch["rate_lost_ci95_boot"][1]
    assert ch["share_same_change"] > 0.99
    assert t["per_T"][1]["change"] and not t["per_T"][1]["established"], t["per_T"][1]
    assert not any(r["established"] for r in t["per_T"])
    assert 1.96 < out["q"] < out["q_t"], (out["q"], out["q_t"])
    assert out["n_identically_zero"] == 0
    # an infeasible pair (both rates exactly zero) is dropped, not tested
    z = make(clear, 0.01, 0.05)
    z["Z"][:, 1:, 0] = 0.0; z["mu"][:, 1:, 0] = 0.0
    o2 = analyze({"zero": z}, n_boot=2000, tag='winner-selftest')
    assert o2["n_identically_zero"] == 2, o2["n_identically_zero"]     # (1,2) under (1), (2)
    # the full shapes (ten methods, twelve periods, six scenarios, 20,000 replicates)
    # through analyze, r8 and the JSON writer, on synthetic rates
    NT_, M_ = 12, len(PUBS)
    big = rng.uniform(1.0, 3.0, size=(A, M_, NT_))
    Zs = {}
    for k in range(6):
        Zs[f"s{k}"] = {}
        bb = 15 + 2 * k
        Zk = np.abs(rng.normal(1.0, 0.1, size=(A, M_, NT_, S, bb))) * big[..., None, None] * 50
        Zs[f"s{k}"] = {"Z": Zk, "mu": Zk.sum(-1) / (50 * bb), "counts": np.full(bb, 50.0),
                       "n": 50 * bb, "n_blocks": bb}
    t0 = time.time()
    o3 = analyze(Zs, tag='winner-selftest')
    for nm, s in o3["scenarios"].items():
        for r in s["per_T"]:
            r["T_us"] = grid_us()[r["t"]]
        for c in s["changes"]:
            c["T_us"] = grid_us()[c["t"]]
    syn = {nm: Zs[f"s{k}"] for k, nm in enumerate(EVAL)}
    o4 = analyze(syn, tag='winner-selftest')
    rec = assemble(syn, o4, [{"site": nm, "holds": True} for nm in EVAL],
                   protocol.seeds(), grid_us())
    json.dumps(rec)
    assert set(rec["R8"]) >= {"n_point_changes", "n_established", "established"}
    assert o3["n_contrasts"] == 6 * A * NT_ * M_ * (M_ - 1) // 2 == 6480
    print(f"selftest OK: q = {out['q']:.3f}, q_t = {out['q_t']:.3f}, planted change "
          f"established, near-tie not, identically-zero contrasts dropped; full shapes "
          f"({o3['n_contrasts']} contrasts, {N_BOOT:,} replicates) in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ main ----
def r8(res, grid):
    """The R8 outcome the paper reads."""
    est, pts = [], []
    for nm, s in res["scenarios"].items():
        for c in s["changes"]:
            key = {"site": nm, "T_us": grid[c["t"]], "free": PUBS[c["winner_1"]],
                   "joint": PUBS[c["winner_2"]], "rate_lost": c["rate_lost"]}
            pts.append(key)
            if c["established"]:
                est.append(key)
    return {"n_point_changes": len(pts), "n_established": len(est), "established": est,
            "max_rate_lost_point": max((p["rate_lost"] for p in pts), default=None),
            "max_rate_lost_established": max((e["rate_lost"] for e in est), default=None),
            "wording": ("the paper may state a change of the best method, and the rate lost "
                        "by keeping the published accounting's choice as a decision cost, only "
                        "at the established changes; every other change is a point estimate "
                        "and is worded as one" + ("" if est else "; no change is established, "
                                                  "and the abstract and conclusion say so"))}


def assemble(arrs, res, rep, seeds, grid):
    """The output record: the gate, the statistics, R8, and the ten-method point
    winners against the printed (nine-method) Table I."""
    # the ten-method point winners against the printed (nine-method) Table I
    ag = {r['site']: r for r in json.load(open(os.path.join(HERE, 'aging.json')))['rows']
          if r['speed_kmh'] == 0.0}
    moved = []
    for nm in EVAL:
        ten = [{"T_us": grid[c["t"]], "free": PUBS[c["winner_1"]], "joint": PUBS[c["winner_2"]]}
               for c in res["scenarios"][nm]["changes"]]
        if ten != ag[nm]['accounting']['winner_changes']:
            moved.append({"site": nm, "ten_methods": ten,
                          "printed": ag[nm]['accounting']['winner_changes']})
    for nm in EVAL:
        res["scenarios"][nm]["site"] = nm
        res["scenarios"][nm]["n"] = int(arrs[nm]["n"])
        for r in res["scenarios"][nm]["per_T"]:
            r["T_us"] = grid[r["t"]]
        for c in res["scenarios"][nm]["changes"]:
            c["T_us"] = grid[c["t"]]
    P = protocol.load()
    return {"script_sha256": hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
            "protocol_sha256": P['self_sha256'],
            "code_commit": subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE,
                                          capture_output=True, text=True).stdout.strip(),
            "seeds": list(seeds), "grid_us": grid, "pubs": PUBS,
            "accountings": {"1": "probe airtime only (Graff Eq. 34; results_freecompute.json)",
                            "2": "inference latency included (results.json, heldout.json)"},
            "block_m": evaluate.BLOCK_M, "gate": rep,
            "gate_tolerances": {"stored_means": TOL_STORED, "aging_json": TOL_AGING},
            "q_definition": ("numpy quantile (linear) at 1 - alpha of the per-replicate maximum, "
                             "over the family, of |D* - D| / SE*; q_t = t_(S-1) quantile at "
                             "1 - (per_comparison_level)/2"),
            "winner_identity": {"ten_method_point_winners_equal_printed": not moved,
                                "moved": moved},
            **res, "R8": r8(res, grid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='winner_uncertainty.json')
    ap.add_argument('--check-only', action='store_true',
                    help="recompute the published point estimates and run the gate only")
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    t0 = time.time()
    lat = json.load(open(os.path.join(HERE, evaluate.LAT_FILE)))
    grid = grid_us()
    seeds = protocol.seeds()
    arrs = {nm: scenario_rates(nm, seeds, lat, grid) for nm in EVAL}
    rep, ok = gate(arrs, grid)
    for r in rep:
        print(f"  gate {r['site']:24s} joint {r['max_abs_joint_vs_stored']:.2e}  "
              f"free {r['max_abs_free_vs_results_freecompute'] if r['max_abs_free_vs_results_freecompute'] is None else format(r['max_abs_free_vs_results_freecompute'], '.2e')}  "
              f"aging {r['max_abs_joint_vs_aging']:.2e}/{r['max_abs_free_vs_aging']:.2e}  "
              f"winners {r['nine_method_winners_equal_aging']}/"
              f"{r['ten_method_winners_equal_accounting']}  "
              f"{'holds' if r['holds'] else 'FAILS'}")
    if not ok:
        raise SystemExit("reproduction gate failed: nothing resampled, nothing written")
    if a.check_only:
        print(f"reproduction gate holds in all {len(EVAL)} scenarios "
              f"({time.time() - t0:.0f}s); nothing resampled, nothing written")
        return
    res = analyze(arrs)
    out = assemble(arrs, res, rep, seeds, grid)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, a.out), 'w'), indent=1)
    R = out["R8"]
    print(f"\nq = {out['q']:.3f}, q_t = {out['q_t']:.3f} over {out['n_contrasts']} contrasts "
          f"({out['n_identically_zero']} identically zero)")
    print(f"R8: {R['n_established']} of {R['n_point_changes']} point-estimate changes "
          f"established; ten-method winners equal the printed ones: "
          f"{out['winner_identity']['ten_method_point_winners_equal_printed']}")
    print(f"wrote {a.out} in {out['seconds']}s")


if __name__ == '__main__':
    main()
