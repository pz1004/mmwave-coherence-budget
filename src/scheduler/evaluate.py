#!/usr/bin/env python3
"""Phase 4 experiment.  Every method, every coherence time, every seed in force, paired
on identical receivers with common random numbers, on the two test cities.

The held-out cities are NOT touched by the default run; they are opened once by
a second evaluate.py run (--sites ... --out heldout.json), and
whatever they show is reported.
"""
from __future__ import annotations
import argparse, functools, json, os, sys, time, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
import obs, feats, sim, methods
import models as M

import protocol
SEEDS = protocol.seeds()        # the ten seeds in force; --seeds overrides it
VALIDATION = 'validation.json'  # where the deployable baseline is picked; --validation
# One-shot ee{e}k{k} baselines pay ONE fresh invocation to exit e, not the restart
# sum; --oneshot-charge restart reproduces the restart sum, for the isolated diff
# that measures the difference.
ONESHOT_FRESH = True
LAT_FILE = 'lat_cpu1t_pcore.json'  # the charges; --lat (sweeps, isolated diffs)
TAU_SCALE = 1.0                  # rho: scales every INFERENCE charge; --tau-scale
TP_AIR_US = None                 # probe air time override; --tp-air-us
TABLES_SUFFIX = ''               # tables_<seed><suffix>.npz; --tables-suffix
ORACLE_MIX = None                # predictor quality q; --oracle-mix (None: trained)
RANK_CHARGE = False              # charge for picking the next codeword; --rank-charge
ZHANG_NP2 = False                # Zhang et al. as published (methods.zhang_np2); --zhang-np2
ZF = 'zhang-NP2'                 # its name; not a FIXED_PREFIXES entry, so not in the fixed menu
RANK_EXACT = False               # ranking counted after the first call only; --rank-exact
LOOKUP_AUG = 'file'              # augmented-table lookup: lat_lookup_aug.json, or 'original'
AUG_KEYS = ()                    # the variants whose tables are kappa-major (scheduler json)
ZHANG_GUARD = False              # both adaptive forms with a deadline guard; --zhang-guard
ZG, ZFG = 'zhang-G', 'zhang-NP2-G'
STRATA = [(-1e9, 0.0), (0.0, 10.0), (10.0, 20.0), (20.0, 1e9)]
STRAT_NAMES = ["below 0 dB", "0-10 dB", "10-20 dB", "above 20 dB"]
BLOCK_M = 50.0
# 20,000, not 2,000.  Holm's tightest threshold over 12 deadlines is 0.05/12 =
# 0.00417, which at 2,000 resamples sits about 4 resamples from the edge: the
# Monte-Carlo SE of such a p-value is ~49% of the threshold, so borderline
# significance decisions are close to coin flips and move when the random
# stream moves.  At 20,000 the MC SE is ~15% of the threshold.
N_BOOT = 20000
FIXED_PREFIXES = ('topk', 'sweep', 'hier', 'fc', 'cnn', 'ee')
# (reported name, table key, whether COMPUTE also re-ranks the remaining probes)
SCHEDULERS = [('sched', 'dpoff', True), ('sched-pi', 'dppi', True),
              ('sched-noreorder', 'dpnoff', False), ('sched-noreorder-pi', 'dpnpi', False),
              ('sched-nocompute', 'nocoff', True), ('sched-nocompute-pi', 'nocpi', True),
              ('osla', 'oslaoff', True), ('osla-pi', 'oslapi', True),
              # planned as though computation were free, charged honestly here
              ('sched-costblind', 'blindoff', True),
              ('sched-costblind-pi', 'blindpi', True),
              # may choose whether/when to compute, but not how deep
              ('sched-1exit', 'oneoff', True), ('sched-1exit-pi', 'onepi', True),
              # metareasoning comparator, stronger than the myopic rule
              ('blinkered', 'blnkoff', True), ('blinkered-pi', 'blnkpi', True),
              # repeated fresh exit-1 calls, and one pass to
              # exit 3 -- together with sched-1exit they separate invocation count
              # from depth (R2)
              ('sched-rep1', 'rep1off', True), ('sched-rep1-pi', 'rep1pi', True),
              ('sched-deep1', 'deep1off', True), ('sched-deep1-pi', 'deep1pi', True),
              # a per-K FC or CNN network as the compute action, against the anytime
              # predictor under the same restrictions (R3).  Names must not start
              # with a FIXED_PREFIXES entry, or they would join the fixed menu.
              ('sched-anyr', 'anyroff', False), ('sched-anyr-pi', 'anyrpi', False),
              ('sched-fcc', 'fccoff', False), ('sched-fcc-pi', 'fccpi', False),
              ('sched-cnnc', 'cnncoff', False), ('sched-cnnc-pi', 'cnncpi', False)]
assert not any(n.startswith(FIXED_PREFIXES) for n, _, _ in SCHEDULERS)
assert not ZF.startswith(FIXED_PREFIXES)
assert not ZG.startswith(FIXED_PREFIXES) and not ZFG.startswith(FIXED_PREFIXES)
PRIMARY = 'sched'
HEADLINE = 'sched-pi'


def load_nets(seed):
    ee = sim.load_predictor(seed)
    nets = {}
    for k in (4, 8, 16, 32, 64):
        m = M.BeamMLP(n_in=k).to(sim.DEV)
        m.load_state_dict(torch.load(os.path.join(HERE, 'models', f'fc_k{k}_{seed}.pt')))
        nets[f'fc{k}'] = m.eval()
        c = M.BeamCNN().to(sim.DEV)
        c.load_state_dict(torch.load(os.path.join(HERE, 'models', f'cnn_k{k}_{seed}.pt')))
        nets[f'cnn{k}'] = c.eval()
    return ee, nets


def stable_seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def block_ids(rx, block_m=None):
    xy = np.floor(rx[:, :2] / (BLOCK_M if block_m is None else block_m)).astype(np.int64)
    _, inv = np.unique(xy[:, 0] * 1000003 + xy[:, 1], return_inverse=True)
    return inv


def _ci(m):
    """95% percentile interval and a two-sided bootstrap p-value against zero."""
    frac = float((m <= 0).mean())
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)),
            max(2 * min(frac, 1 - frac), 1.0 / N_BOOT))


def paired_bootstrap(diffs, blocks, n_block, rng):
    """Block bootstrap of one comparison across the WHOLE deadline grid.

    `diffs` is one per-receiver paired difference vector per deadline, in grid
    order.  Returns (per_deadline, aggregate): a list of (lo, hi, p) for each
    deadline, and one (lo, hi, p) for the grid-averaged contrast.

    The blocks are resampled ONCE per replicate and the same
    resample is used for every deadline, so the 12 per-deadline statistics
    within a replicate describe the same bootstrap world and averaging them
    inside the replicate yields a distribution for the site-level contrast.
    An independent stream per deadline would be fine for a per-deadline
    interval but would make the deadlines uncombinable: there would be no
    replicate in which all 12 co-occur, so no aggregate interval would exist.
    """
    S = np.stack([np.bincount(blocks, weights=d, minlength=n_block) for d in diffs])
    c = np.bincount(blocks, minlength=n_block).astype(float)
    pick = rng.integers(0, n_block, size=(N_BOOT, n_block))
    den = c[pick].sum(1)                      # (N_BOOT,)  shared denominator
    per = S[:, pick].sum(2) / den             # (n_deadline, N_BOOT)
    return [_ci(per[i]) for i in range(len(diffs))], _ci(per.mean(0))


MIN_STRATUM_BLOCKS = 5     # a stratum's contrast gets an interval only above this


def paired_bootstrap_hier(dS, blocks, n_block, rng):
    """Seed-by-block bootstrap of one comparison across the grid.

    dS[t] is the (seeds, n) per-receiver paired difference at deadline t.  Each
    replicate resamples the seeds with replacement and the 50 m blocks once, the
    same block resample for every seed and every deadline -- a crossed design,
    mildly conservative on the seed-by-block interaction.  paired_bootstrap
    averages seeds first and so carries no seed variability; this one does.

    Also returns the t interval  D +/- t_(S-1) sqrt(s_seed^2/S + SE_block^2),
    with s_seed the SD of the S per-seed grid-averaged differences and SE_block
    the block-bootstrap SE of their mean.  With ten seeds a percentile interval
    is too narrow on the seed component, so the reading rules require both to
    exclude zero.

    Returns (per_deadline, aggregate) as paired_bootstrap does, plus a dict with
    the t interval and its ingredients.  Nothing is rounded.
    """
    from scipy import stats
    S = dS[0].shape[0]
    Z = np.stack([np.stack([np.bincount(blocks, weights=d[s], minlength=n_block)
                            for s in range(S)]) for d in dS])          # (T, S, B)
    c = np.bincount(blocks, minlength=n_block).astype(float)
    pick = rng.integers(0, n_block, size=(N_BOOT, n_block))
    sidx = rng.integers(0, S, size=(N_BOOT, S))
    rows = np.arange(N_BOOT)[:, None]
    w = np.zeros((N_BOOT, n_block)); np.add.at(w, (rows, pick), 1.0)   # block multiplicity
    m = np.zeros((N_BOOT, S)); np.add.at(m, (rows, sidx), 1.0)         # seed multiplicity
    den = w @ c                                                          # receivers drawn
    Zp = np.einsum('rb,tsb->rts', w, Z)                                  # (N, T, S)
    per = np.einsum('rs,rts->rt', m, Zp) / (S * den[:, None])            # seed x block
    per_b = Zp.sum(2) / (S * den[:, None])                               # blocks only
    D_s = np.array([np.mean([d[s].mean() for d in dS]) for s in range(S)])
    D = float(D_s.mean())
    se_block = float(per_b.mean(1).std(ddof=1))
    s_seed = float(D_s.std(ddof=1)) if S > 1 else float('nan')
    tint = {"ci95_t": [float('nan'), float('nan')], "p_t": float('nan'),
            "seed_sd": s_seed, "se_block": se_block, "per_seed": D_s.tolist(), "n_seeds": S}
    if S > 1:
        se = float(np.sqrt(s_seed ** 2 / S + se_block ** 2))
        q = float(stats.t.ppf(0.975, S - 1))
        tint.update(ci95_t=[D - q * se, D + q * se],
                    p_t=float(2 * stats.t.sf(abs(D) / se, S - 1)) if se > 0 else 0.0)
    return [_ci(per[:, i]) for i in range(len(dS))], _ci(per.mean(1)), D, tint


def hier_record(dS, blocks, n_block, rng, grid):
    """The *_aggregate_hier entry and the *_rows_hier list for one contrast."""
    pers, agg, D, tint = paired_bootstrap_hier(dS, blocks, n_block, rng)
    aggr = {"diff": D, "ci95": [agg[0], agg[1]], "p": agg[2], **tint}
    rows = [{"T_us": T, "diff": float(d.mean()), "ci95": [lo, hi], "p": pv}
            for T, d, (lo, hi, pv) in zip(grid, dS, pers)]
    return aggr, rows


COVERAGE_OF = ('sched', 'sched-pi')   # controllers whose decisions are classified
MIN_CELL = 20                           # sched.MIN_N; checked against scheduler.json


def load_coverage(seed):
    """Per-cell sample counts sched.py saved for the headline controller
    (coverage_<seed><suffix>.npz, untracked), or None if absent."""
    p = os.path.join(HERE, f'coverage_{seed}{TABLES_SUFFIX}.npz')
    return dict(np.load(p)) if os.path.exists(p) else None


def coverage_of(vis, cov, T, pi):
    """Share of the deployed policy's decisions made in cells whose payoff and
    transition estimates are the table's own (on-policy for -pi tables,
    off-policy for -off tables), an off-policy fallback (-pi only), or the
    analytic prior, measured on the decisions the controller actually takes on
    an evaluation site, not on the estimation rollouts."""
    v = vis.cpu().numpy()
    tot = v.sum()
    out = {"decisions": float(tot)}
    for nm, key in (("gm", "cnt"), ("tr", "tot")):
        off = cov[f'{key}_off'] >= MIN_CELL
        if pi:
            on = cov[f'{key}_on_{T}'] >= MIN_CELL
            prim, fb, an = on, ~on & off, ~on & ~off
        else:
            prim, fb, an = off, np.zeros_like(off), ~off
        out[f"{nm}_primary"] = float(v[prim].sum() / tot)
        out[f"{nm}_fallback_off"] = float(v[fb].sum() / tot)
        out[f"{nm}_analytic"] = float(v[an].sum() / tot)
    return out


def charges(lat, free_compute=False):
    """Every charge the executor pays, in one place.

    TAU      restart charge to reach exit e (the controller: e fresh passes)
    FRESH    one fresh pass to exit e (a one-shot compute-and-commit method)
    ONESHOT  what the one-shot ee{e}k{k} baselines are charged: FRESH, or TAU
             with --oneshot-charge restart
    ch       per-component charges; fc_k*/cnn are the fixed networks'
    TAU_SCALE (rho) multiplies every inference charge -- TAU, FRESH and the
    fixed networks -- and nothing else: the probe bookkeeping, the table
    lookup and the Neyman-Pearson test are not inference.  At rho = 1 and
    no probe override, no arithmetic is added, so every existing number
    reproduces bit for bit.
    """
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    TAU = np.array([0.0, *[lat['tau_restart_us'][str(i)] for i in (1, 2, 3)]])
    FRESH = np.array([0.0, *[lat['tau_fresh_us'][str(i)] for i in (1, 2, 3)]])
    if TAU_SCALE != 1.0:
        TAU, FRESH = TAU * TAU_SCALE, FRESH * TAU_SCALE
        ch = {k: (v * TAU_SCALE if (k.startswith('fc_k') or k == 'cnn') else v)
              for k, v in ch.items()}
    T_PROBE = (obs.T_P_US if TP_AIR_US is None else TP_AIR_US) + ch['probe_update']
    C_LOOK, C_DEC = sim.lookup_charge(lat), ch['np_test']   # real-table lookup on the primary profile
    if free_compute:
        # Graff et al.'s accounting, Eq. (34): the overhead term counts the time
        # spent transmitting SS and CSI-RS blocks and NOTHING else.  Exactly one
        # thing changes from the honest run -- inference becomes free.  The probe
        # charge, the channels, the noise draws and the receivers are identical,
        # so any difference between the two runs is the compute term and only the
        # compute term.
        TAU, FRESH = np.zeros_like(TAU), np.zeros_like(FRESH)
        C_LOOK = C_DEC = 0.0
        ch = {k: (0.0 if (k.startswith('fc_k') or k == 'cnn') else v)
              for k, v in ch.items()}
    ONESHOT = FRESH if ONESHOT_FRESH else TAU
    # the masked argmax that picks each next codeword once inference has
    # re-ranked the search, at its own timing; not scaled by rho (controller logic,
    # like the lookup), and free under Graff et al.'s accounting
    C_RANK = ch['sweep_select'] if (RANK_CHARGE and not free_compute) else 0.0
    # the kappa-augmented table's consult (timed on it), for the variants that use it,
    # free under Graff et al.'s accounting; and the deadline guard's check, a threshold
    # comparison charged as the Neyman-Pearson decision is (0 when that is free)
    C_LOOK_AUG = None
    if RANK_EXACT:
        C_LOOK_AUG = (0.0 if free_compute else
                      sim.lookup_charge(lat) if LOOKUP_AUG == 'original' else sim.lookup_charge_aug(lat))
    return dict(ch=ch, TAU=TAU, FRESH=FRESH, ONESHOT=ONESHOT, T_PROBE=T_PROBE,
                C_LOOK=C_LOOK, C_DEC=C_DEC, C_RANK=C_RANK, C_LOOK_AUG=C_LOOK_AUG,
                C_GUARD=C_DEC)


def check_charge(lat):
    """Fail the run if the charge is not the work the executor performs.

    Three invariants, all cheap:
      1. the restart charge is the running sum of the fresh invocations;
      2. one fresh pass to exit j costs strictly less than the incremental
         chain to the same exit, because it evaluates only head j -- if this
         inverts, tau_fresh was measured with something cached;
      3. the planner and the executor charge the same vector.
    """
    f = [lat['tau_fresh_us'][str(j)] for j in (1, 2, 3)]
    r = [lat['tau_restart_us'][str(e)] for e in (1, 2, 3)]
    c = [lat['tau_cumulative_us'][str(e)] for e in (1, 2, 3)]
    for e in (1, 2, 3):
        assert abs(r[e - 1] - sum(f[:e])) < 1e-6, (
            f"tau_restart[{e}] = {r[e-1]} != sum of fresh invocations {sum(f[:e])}")
    for j in (2, 3):
        assert f[j - 1] < c[j - 1], (
            f"tau_fresh[{j}] = {f[j-1]} >= tau_cumulative[{j}] = {c[j-1]}: a fresh "
            "pass evaluates only head j, so it must be cheaper than the chain "
            "that evaluates every head.  Was it timed against a cached activation?")
    rec = json.load(open(os.path.join(HERE, f'scheduler{TABLES_SUFFIX}.json')))
    q = charges(lat)
    charged = q['TAU'].tolist()
    assert max(abs(a - b) for a, b in zip(rec['tau_us'], charged)) < 1e-6, (
        f"the planner solved with tau={rec['tau_us']} but the executor charges "
        f"{charged}; the policy is optimal for a cost nobody pays")
    for vk, tv in rec.get('variant_tau_us', {}).items():
        mine = sim.variant_specs(lat, TAU_SCALE)[vk][1].tolist()
        assert max(abs(a - b) for a, b in zip(tv, mine)) < 1e-6, (
            f"variant {vk}: planner tau {tv} != executor tau {mine}")
    assert rec.get('min_cell', MIN_CELL) == MIN_CELL, "coverage threshold != sched.MIN_N"
    # 4. the planner and the executor also agree on the probe and the lookup
    assert abs(rec['t_probe_us'] - q['T_PROBE']) < 1e-9, (
        f"planner t_p {rec['t_probe_us']} != executor t_p {q['T_PROBE']}")
    assert abs(rec['c_lookup_us'] - q['C_LOOK']) < 1e-12, (
        f"planner lookup {rec['c_lookup_us']} != executor lookup {q['C_LOOK']}")
    # 5. one-shot methods pay one fresh invocation, and the restart sum only in the
    #    --oneshot-charge restart reproduction.
    want = q['FRESH'] if ONESHOT_FRESH else q['TAU']
    assert np.array_equal(q['ONESHOT'], want), "one-shot charge is not the stated one"
    if ONESHOT_FRESH:
        assert np.allclose(q['ONESHOT'][1:], np.array(f) * TAU_SCALE), (
            "the one-shot baselines are not charged tau_fresh")
    # 6. the planner's rollouts and the executor run the same predictor:
    #    a predictor-quality table is evaluated only at its own q.
    assert float(rec.get('oracle_mix') or 0.0) == float(ORACLE_MIX or 0.0), (
        f"tables solved at oracle_mix={rec.get('oracle_mix')} but evaluated at {ORACLE_MIX}")
    # 7. the planner and the executor charge the same re-ranking cost
    assert abs(float(rec.get('c_rank_us', 0.0)) - q['C_RANK']) < 1e-12, (
        f"tables solved with c_rank={rec.get('c_rank_us', 0.0)} but the executor charges "
        f"{q['C_RANK']}")
    # 8. both count the ranking the same way, and charge the augmented table's
    #    consult the same time; the kappa-major tables are the ones the planner solved
    assert (rec.get('c_rank_mode') == 'exact') == RANK_EXACT, (
        f"tables solved with ranking mode {rec.get('c_rank_mode')} but the executor's exact "
        f"count is {RANK_EXACT}")
    if RANK_EXACT:
        assert abs(float(rec['c_lookup_aug_us']) - q['C_LOOK_AUG']) < 1e-12, (
            f"planner lookup {rec['c_lookup_aug_us']} != executor lookup {q['C_LOOK_AUG']} "
            "on the augmented table")
        assert set(rec['augmented_variants']) == set(AUG_KEYS), (rec['augmented_variants'],
                                                                 AUG_KEYS)


def evaluate_site(name, grid, lat, tables, verbose=True, free_compute=False):
    q = charges(lat, free_compute)
    ch, TAU, ONESHOT = q['ch'], q['TAU'], q['ONESHOT']
    T_PROBE, C_LOOK, C_DEC = q['T_PROBE'], q['C_LOOK'], q['C_DEC']
    C_RANK = q['C_RANK']                      # 0 unless --rank-charge
    C_LOOK_AUG, C_GUARD = q['C_LOOK_AUG'], q['C_GUARD']
    NEWV = sim.variant_specs(lat, 0.0 if free_compute else TAU_SCALE)
    cov = {s: load_coverage(s) for s in SEEDS}
    coverage_acc = {}
    site = sim.Site(name)
    blocks = block_ids(site.rx); n_block = int(blocks.max()) + 1
    strat = np.full(site.n, -1)
    for i, (lo, hi) in enumerate(STRATA):
        strat[(site.strat_snr >= lo) & (site.strat_snr < hi)] = i
    assert (strat >= 0).all()
    oracle_mean = float(site.oracle_se.mean())

    acc = {}                     # method -> {T -> list over seeds of per-rx net rate}
    diag, alloc, alloc_h = {}, {}, {}
    for seed in SEEDS:
        ee, nets = load_nets(seed)
        # Deterministic across processes: Python's hash() of a str is salted per
        # interpreter, so a seed taken from hash((name, seed)) would give a
        # DIFFERENT pilot noise draw on every invocation, and no number would be
        # reproducible.
        gen = torch.Generator(device=sim.DEV).manual_seed(stable_seed(name, seed))
        gen2 = torch.Generator(device=sim.DEV).manual_seed(stable_seed(name, seed, 'r'))
        y = sim.measure(site.pbar, gen)                    # common random numbers
        # the predictor-quality condition, or None (the trained predictor)
        mix = sim.oracle_mix(site, name, seed, ORACLE_MIX)
        eefn = methods.fixed_net if mix is None else functools.partial(methods.fixed_net, mix=mix)

        runs = {}
        runs['oracle'] = (torch.zeros(site.n, device=sim.DEV), site.oracle_beam + obs.LAYER6_OFFSET)
        for k in (1, 2, 4, 8, 16, 32, 64):
            runs[f'sweep64' if k == 64 else f'topk{k}'] = methods.topk(site, y, k, T_PROBE)
        for L in (3, 4, 5, 6):
            runs[f'hier{L}'] = methods.hierarchical(site, y, L, T_PROBE)
        for k in (4, 8, 16, 32, 64):
            runs[f'fc{k}'] = methods.fixed_net(site, y, k, 'fc', nets[f'fc{k}'], ch[f'fc_k{k}'], T_PROBE)
            runs[f'cnn{k}'] = methods.fixed_net(site, y, k, 'cnn', nets[f'cnn{k}'], ch['cnn'], T_PROBE)
        for e in (1, 2, 3):
            for k in (2, 4, 8, 16, 32):
                runs[f'ee{e}k{k}'] = eefn(site, y, k, f'ee{e}', ee, ONESHOT[e], T_PROBE)

        for T in grid:
            for nm, (al, asl) in {'zhang-AHIBS': (True, False),
                                  'zhang-AHTBS': (False, True),
                                  'zhang-AHTBS+AHIBS': (True, True)}.items():
                guard_here = ZHANG_GUARD and nm == 'zhang-AHTBS+AHIBS'
                if guard_here:      # the guarded form replays this call's repeats
                    snap = gen2.get_state()
                d, c, slots, layer = methods.zhang(site, y, float(T), T_PROBE, C_DEC,
                                                   al, asl, gen=gen2)
                runs[(nm, T)] = (d, c)
                diag.setdefault((nm, T), []).append(
                    (float(slots.float().mean()), float(layer.float().mean())))
                if guard_here:
                    g3 = torch.Generator(device=sim.DEV)
                    g3.set_state(snap)
                    dgd, cgd, sgd, lgd, info = methods.zhang_guarded(
                        site, y, float(T), T_PROBE, C_DEC, g3, C_GUARD)
                    runs[(ZG, T)] = (dgd, cgd)
                    # [mean looks, mean layer, share stopped or cut short by the guard,
                    #  share without a beam, share overrunning T]
                    diag.setdefault((ZG, T), []).append(
                        (float(sgd.mean()), float(lgd.float().mean()), info[0], info[1],
                         float(((dgd >= T) & (lgd > 0)).float().mean())))
                    # the unguarded form's own overrun share, recorded beside it
                    diag.setdefault((nm + '-overrun', T), []).append(
                        (float((d >= T).float().mean()),))
            for nm, key, ro in SCHEDULERS:
                tb = tables[seed][f'{key}_{T}']
                vk = key[:-3] if key.endswith('off') else key[:-2]
                sp, tv = (NEWV[vk][0], NEWV[vk][1]) if vk in NEWV else (None, TAU)
                sp = sim.with_mix(sp, mix)
                vis = (torch.zeros((sim.N_EXIT + 1, sim.K_MAX + 1, sim.NB, sim.NC),
                                   dtype=torch.float64, device=sim.DEV)
                       if nm in COVERAGE_OF and cov[seed] is not None else None)
                aug = RANK_EXACT and vk in AUG_KEYS      # a kappa-major table
                kout = [] if aug else None
                d, c, k_, lv, up = methods.scheduled(site, y, ee, tb, tv, T_PROBE,
                                                     C_LOOK_AUG if aug else C_LOOK,
                                                     reorder=ro, spec=sp,
                                                     nets=nets, visits=vis,
                                                     c_rank=C_RANK, rank_exact=aug,
                                                     kappa_out=kout)
                if aug:
                    # [share that called the predictor, mean kappa among them, mean
                    #  ranking operations performed, mean that counting every probe adds]
                    kp = kout[0].double()
                    called = kp >= 0
                    diag.setdefault((nm + '-kappa', T), []).append(
                        (float(called.double().mean()),
                         float(kp[called].mean()) if bool(called.any()) else 0.0,
                         float(torch.where(called, k_.double() - kp, torch.zeros_like(kp)).mean())
                         if ro else 0.0,
                         float(torch.where(called, kp, torch.zeros_like(kp)).mean()) if ro else 0.0))
                if vis is not None:
                    coverage_acc.setdefault(nm, []).append(
                        coverage_of(vis, cov[seed], T, pi=nm.endswith('-pi')))
                runs[(nm, T)] = (d, c)
                kn, ln = k_.cpu().numpy(), lv.cpu().numpy()
                # [mean probes, mean exit level, frac committed on the
                #  prediction, mean tau us, P(computed at all), mean probe us]
                diag.setdefault((nm, T), []).append(
                    (float(kn.mean()), float(ln.mean()), float(up.float().mean()),
                     float(tv[ln].mean()), float((ln > 0).mean()),
                     float(kn.mean() * T_PROBE)))
                if nm in (PRIMARY, HEADLINE):
                    (alloc if nm == PRIMARY else alloc_h).setdefault(T, []).append(
                        [[float(kn[strat == i].mean()), float(ln[strat == i].mean()),
                          float(TAU[ln[strat == i]].mean())] for i in range(4)])

        for T in grid:
            vb, cbT = methods.oracle_stop(site, y, T_PROBE, float(T))
            acc.setdefault('bound-stopping', {}).setdefault(T, []).append(vb.cpu().numpy())
        for key, (d, c) in runs.items():
            se = site.se.gather(1, c[:, None])[:, 0].double()
            if isinstance(key, tuple):
                nm, T = key
                r = (torch.clamp(1 - d / T, min=0.0) * se).cpu().numpy()
                acc.setdefault(nm, {}).setdefault(T, []).append(r)
            else:
                for T in grid:
                    r = (torch.clamp(1 - d / T, min=0.0) * se).cpu().numpy()
                    acc.setdefault(key, {}).setdefault(T, []).append(r)
                acc.setdefault(key, {}).setdefault('cost', []).append(float(d.mean()))
                acc.setdefault(key, {}).setdefault('se', []).append(float(se.mean()))
        if ZHANG_NP2:
            # Zhang et al.'s descent as published, on its own draws.  Its first
            # looks are the raw statistic behind y (the same generator draws, so y
            # itself is untouched and must be reproduced exactly); its repeats come
            # from an array indexed by (receiver, codeword, look), common to every T.
            X1 = sim.measure_chi2(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
                stable_seed(name, seed)))
            assert torch.equal(sim.report_of(X1), y), "X1 does not reproduce y"
            zr = np.random.default_rng(stable_seed(name, seed, 'zf'))
            Zr = zr.standard_normal((site.n, site.pbar.shape[1], 4))
            Qr = zr.chisquare(2 * obs.K_F - 1, (site.n, site.pbar.shape[1], 4))
            zd = methods.zhang_np2_descent(site, X1, Zr, Qr, T_PROBE, C_DEC)
            for T in grid:
                d, c, lay, dg = methods.zhang_np2(zd, float(T))
                se = site.se.gather(1, c[:, None])[:, 0].double()
                r = (torch.clamp(1 - d / T, min=0.0) * se).cpu().numpy()
                acc.setdefault(ZF, {}).setdefault(T, []).append(r)
                diag.setdefault((ZF, T), []).append(dg)
                if ZHANG_GUARD:
                    # the same descent with the deadline guard, on the same draws;
                    # [mean looks, mean layer, share stopped or cut short by the guard,
                    #  share without a beam, share overrunning T]
                    d, c, lay, dg = methods.zhang_np2_guarded(site, X1, Zr, Qr, float(T),
                                                              T_PROBE, C_DEC, C_GUARD, zd)
                    se = site.se.gather(1, c[:, None])[:, 0].double()
                    r = (torch.clamp(1 - d / T, min=0.0) * se).cpu().numpy()
                    acc.setdefault(ZFG, {}).setdefault(T, []).append(r)
                    diag.setdefault((ZFG, T), []).append(dg)
        if verbose:
            print(f"  [{name} seed {seed}] done", flush=True)

    out = {"site": name, "n": site.n, "n_blocks": n_block,
           "oracle_se": round(oracle_mean, 4),
           "strata_counts": [int((strat == i).sum()) for i in range(4)],
           "strata_frac": [round(float((strat == i).mean()), 4) for i in range(4)],
           "methods": {}, "grid": list(grid)}
    for nm, byT in acc.items():
        m = {"per_T": {}}
        if 'cost' in byT:
            m["cost_us"] = round(float(np.mean(byT['cost'])), 3)
            m["eta"] = round(float(np.mean(byT['se'])) / oracle_mean, 4)
        for T in grid:
            r = np.stack(byT[T])                     # (seeds, n)
            mu = r.mean(1)
            m["per_T"][str(T)] = {
                "mean": round(float(mu.mean()), 5),
                "seed_spread": round(float(mu.max() - mu.min()), 5),
                "outage": round(float((r == 0).mean()), 5),
                "strata": [round(float(r[:, strat == i].mean()), 5) if (strat == i).any()
                           else None for i in range(4)],
                "seed_sd": round(float(mu.std(ddof=1)), 5) if len(mu) > 1 else None}
        m["seed_grid_means"] = [round(float(x), 5) for x in
                                np.mean([np.stack(byT[T]).mean(1) for T in grid], 0)]
        out["methods"][nm] = m

    # ---- the gate: every scheduler variant vs the post-hoc best FIXED --------
    # One generator per comparison, seeded from the comparison's own identity.
    # A single shared stream would make every reported CI depend on how many
    # other comparisons happened to run first: adding a variant to SCHEDULERS
    # could move a Holm count on bit-identical per-receiver data.
    def brng(*key):
        return np.random.default_rng(stable_seed('boot', name, *key))
    rng = np.random.default_rng(881)        # kept only for anything unkeyed
    fixed = [nm for nm in acc if isinstance(nm, str) and nm != 'oracle'
             and nm.startswith(FIXED_PREFIXES)]
    gate = {}
    gate_agg = {}
    hier_agg, hier_rows = {}, {}          # the seed-by-block counterparts, per family
    for variant, _, _ in SCHEDULERS:
        rows = []
        bestT = {T: max(fixed, key=lambda nm: out['methods'][nm]['per_T'][str(T)]['mean'])
                 for T in grid}
        dS = [np.stack(acc[variant][T]) - np.stack(acc[bestT[T]][T]) for T in grid]
        ds = [x.mean(0) for x in dS]
        hier_agg.setdefault('gate', {})[variant], hier_rows.setdefault('gate', {})[variant] = \
            hier_record(dS, blocks, n_block, brng('hier', 'gate', variant), grid)
        pers, agg = paired_bootstrap(ds, blocks, n_block, brng('gate', variant))
        gate_agg[variant] = {"diff": round(float(np.mean([d.mean() for d in ds])), 5),
                             "ci95": [round(agg[0], 5), round(agg[1], 5)],
                             "p": round(agg[2], 5)}
        for T, d, (lo, hi, pv) in zip(grid, ds, pers):
            best = bestT[T]
            rows.append({"T_us": T, "best_fixed": best, "p": round(pv, 5),
                         "best_fixed_mean": out['methods'][best]['per_T'][str(T)]['mean'],
                         "sched_mean": out['methods'][variant]['per_T'][str(T)]['mean'],
                         "paired_diff": round(float(d.mean()), 5),
                         "ci95": [round(lo, 5), round(hi, 5)],
                         "rel_to_oracle": round(float(d.mean()) / oracle_mean, 5)})
        gate[variant] = rows
    out["gate_rows"] = gate
    out["gate_aggregate"] = gate_agg

    # ---- pre-specified comparisons, with CIs -------------------------------
    # (a) does the compute action pay?  every with-compute controller against
    #     every without-compute one, so the choice can be made on VALIDATION
    #     rather than by taking a maximum over variants on the test city.
    WITH = [n for n, _, _ in SCHEDULERS if 'nocompute' not in n
            and 'osla' not in n and 'blinkered' not in n]
    WO = [n for n, _, _ in SCHEDULERS if 'nocompute' in n]
    pays, pays_agg = {}, {}
    for a_ in WITH:
        for b_ in WO:
            rows = []
            dS = [np.stack(acc[a_][T]) - np.stack(acc[b_][T]) for T in grid]
            ds = [x.mean(0) for x in dS]
            pers, agg = paired_bootstrap(ds, blocks, n_block, brng('pays', a_, b_))
            k_ = f"{a_}|{b_}"
            hier_agg.setdefault('compute_pays', {})[k_], \
                hier_rows.setdefault('compute_pays', {})[k_] = \
                hier_record(dS, blocks, n_block, brng('hier', 'pays', a_, b_), grid)
            for T, d, (lo, hi, pv) in zip(grid, ds, pers):
                rows.append({"T_us": T, "diff": round(float(d.mean()), 5),
                             "ci95": [round(lo, 5), round(hi, 5)], "p": round(pv, 5),
                             "with": out['methods'][a_]['per_T'][str(T)]['mean'],
                             "without": out['methods'][b_]['per_T'][str(T)]['mean']})
            pays[f"{a_}|{b_}"] = rows
            pays_agg[f"{a_}|{b_}"] = {
                "diff": round(float(np.mean([d.mean() for d in ds])), 5),
                "ci95": [round(agg[0], 5), round(agg[1], 5)], "p": round(agg[2], 5)}
    out["compute_pays_rows"] = pays
    out["compute_pays_aggregate"] = pays_agg

    # (c) MECHANISM: what is the compute action actually buying?  Each pair is
    #     paired per receiver under common random numbers, like (a).
    #       costblind -- same action set and model, planned as if computation
    #                    were free, charged honestly: isolates latency-AWARENESS
    #                    from predictor availability.
    #       1exit     -- may choose whether/when to compute but not how deep:
    #                    isolates selective invocation from depth allocation.
    #       noreorder -- compute may not re-rank the remaining probes: isolates
    #                    supplying a beam from ordering the search.
    #       blinkered/osla -- the metareasoning comparators.
    #       zhang     -- the published adaptive comparator, with the interval
    #                    every other contrast reports.  brng() is keyed by the
    #                    pair, so each pair draws an independent stream.
    mech, mech_agg = {}, {}
    for a_, b_ in (('sched-pi', 'sched-costblind-pi'), ('sched', 'sched-costblind'),
                   ('sched-pi', 'sched-1exit-pi'), ('sched', 'sched-1exit'),
                   ('sched-pi', 'sched-noreorder-pi'), ('sched', 'sched-noreorder'),
                   ('sched-pi', 'blinkered-pi'), ('sched', 'blinkered'),
                   ('sched-pi', 'osla-pi'), ('sched', 'osla'),
                   ('sched-pi', 'zhang-AHTBS+AHIBS'),
                   ('sched', 'zhang-AHTBS+AHIBS'),
                   # Zhang et al. as published; present only with --zhang-np2,
                   # keyed by pair, so the other pairs' streams are untouched
                   ('sched-pi', ZF), ('sched', ZF),
                   # both forms with the deadline guard; present only with
                   # --zhang-guard, keyed by pair, so the other pairs' streams are untouched
                   ('sched-pi', ZG), ('sched', ZG), ('sched-pi', ZFG), ('sched', ZFG),
                   # depth vs invocation count (R2) and
                   # predictor strength under identical restrictions (R3).  Keyed
                   # by pair, so the other pairs' streams are untouched.
                   ('sched-pi', 'sched-rep1-pi'), ('sched', 'sched-rep1'),
                   ('sched-pi', 'sched-deep1-pi'), ('sched', 'sched-deep1'),
                   ('sched-fcc-pi', 'sched-anyr-pi'), ('sched-fcc', 'sched-anyr'),
                   ('sched-cnnc-pi', 'sched-anyr-pi'), ('sched-cnnc', 'sched-anyr')):
        if a_ not in acc or b_ not in acc:
            continue
        rows = []
        dS = [np.stack(acc[a_][T]) - np.stack(acc[b_][T]) for T in grid]
        ds = [x.mean(0) for x in dS]
        pers, agg = paired_bootstrap(ds, blocks, n_block, brng('mech', a_, b_))
        k_ = f"{a_}|{b_}"
        hier_agg.setdefault('mechanism', {})[k_], hier_rows.setdefault('mechanism', {})[k_] = \
            hier_record(dS, blocks, n_block, brng('hier', 'mech', a_, b_), grid)
        for T, d, (lo, hi, pv) in zip(grid, ds, pers):
            rows.append({"T_us": T, "diff": round(float(d.mean()), 5),
                         "ci95": [round(lo, 5), round(hi, 5)], "p": round(pv, 5),
                         "a": out['methods'][a_]['per_T'][str(T)]['mean'],
                         "b": out['methods'][b_]['per_T'][str(T)]['mean']})
        mech[f"{a_}|{b_}"] = rows
        mech_agg[f"{a_}|{b_}"] = {
            "diff": round(float(np.mean([d.mean() for d in ds])), 5),
            "ci95": [round(agg[0], 5), round(agg[1], 5)], "p": round(agg[2], 5)}
    out["mechanism_rows"] = mech
    out["mechanism_aggregate"] = mech_agg

    # (b) against the DEPLOYABLE baseline: the fixed strategy chosen per deadline
    #     on the validation city, which is a pre-specified comparator, unlike the
    #     post-hoc best fixed the gate uses.
    vpath = os.path.join(HERE, VALIDATION)
    out["deployable_rows"] = {}
    if os.path.exists(vpath) and name != json.load(open(vpath))['sites'][0]['site']:
        vs = json.load(open(vpath))['sites'][0]
        vf = [m for m in vs['methods'] if m.startswith(FIXED_PREFIXES) and m != 'oracle'
              and m in acc]
        pick = {T: max(vf, key=lambda m: vs['methods'][m]['per_T'][str(T)]['mean'])
                for T in grid}
        out["deployable_choice"] = {str(T): pick[T] for T in grid}
        out["deployable_aggregate"] = {}
        for v, _, _ in SCHEDULERS:
            rows = []
            dS = [np.stack(acc[v][T]) - np.stack(acc[pick[T]][T]) for T in grid]
            ds = [x.mean(0) for x in dS]
            pers, agg = paired_bootstrap(ds, blocks, n_block, brng('deploy', v))
            hier_agg.setdefault('deployable', {})[v], hier_rows.setdefault('deployable', {})[v] = \
                hier_record(dS, blocks, n_block, brng('hier', 'deploy', v), grid)
            for T, d, (lo, hi, pv) in zip(grid, ds, pers):
                rows.append({"T_us": T, "baseline": pick[T],
                             "baseline_mean": out['methods'][pick[T]]['per_T'][str(T)]['mean'],
                             "sched_mean": out['methods'][v]['per_T'][str(T)]['mean'],
                             "diff": round(float(d.mean()), 5),
                             "ci95": [round(lo, 5), round(hi, 5)], "p": round(pv, 5)})
            out["deployable_rows"][v] = rows
            out["deployable_aggregate"][v] = {
                "diff": round(float(np.mean([d.mean() for d in ds])), 5),
                "ci95": [round(agg[0], 5), round(agg[1], 5)], "p": round(agg[2], 5)}
    for fam in hier_agg:
        out[f"{fam}_aggregate_hier"] = hier_agg[fam]
        out[f"{fam}_rows_hier"] = hier_rows[fam]

    # ---- descriptive: the prespecified oracle-SNR strata as CONTRASTS ----
    # The frozen protocol requires strata beside every pooled number.  Each
    # stratum's contrast resamples only the blocks holding its receivers, and
    # gets an interval only when it spans at least MIN_STRATUM_BLOCKS blocks.
    def diffs(a_, b_):
        if b_ is None:          # against the deployable fixed strategy, per deadline
            return [np.stack(acc[a_][T]) - np.stack(acc[pick[T]][T]) for T in grid]
        return [np.stack(acc[a_][T]) - np.stack(acc[b_][T]) for T in grid]
    SPAIRS = [('compute_pays', 'sched-pi', 'sched-nocompute-pi'),
              ('compute_pays', 'sched', 'sched-nocompute'),
              ('mechanism', 'sched-pi', 'sched-costblind-pi'),
              ('mechanism', 'sched', 'sched-costblind'),
              ('deployable', 'sched-pi', None), ('deployable', 'sched', None)]
    strata_c = {}
    for fam, a_, b_ in SPAIRS:
        if a_ not in acc or (b_ is not None and b_ not in acc) or \
                (b_ is None and "deployable_choice" not in out):
            continue
        dS = diffs(a_, b_)
        key = a_ if b_ is None else f"{a_}|{b_}"
        per = []
        for i in range(4):
            sel = strat == i
            ub, inv = np.unique(blocks[sel], return_inverse=True)
            rec = {"stratum": STRAT_NAMES[i], "n_rx": int(sel.sum()), "n_blocks": int(len(ub)),
                   "diff": float(np.mean([x[:, sel].mean() for x in dS])) if sel.any() else None}
            if len(ub) >= MIN_STRATUM_BLOCKS:
                pers_, agg_, D_, tint_ = paired_bootstrap_hier(
                    [x[:, sel] for x in dS], inv, len(ub), brng('strata', fam, key, i))
                rec.update(ci95=[agg_[0], agg_[1]], p=agg_[2], **tint_)
            per.append(rec)
        strata_c.setdefault(fam, {})[key] = per
    out["strata_contrasts"] = strata_c

    # ---- descriptive: block-size sensitivity of the headline contrasts ----
    bss = {}
    for bm in (25.0, 100.0):
        bl = block_ids(site.rx, bm); nb = int(bl.max()) + 1
        for fam, a_, b_ in SPAIRS[:4]:
            if a_ not in acc or b_ not in acc:
                continue
            _, agg_, D_, tint_ = paired_bootstrap_hier(diffs(a_, b_), bl, nb,
                                                      brng('blocksize', bm, a_, b_))
            bss.setdefault(f"{bm:g}", {})[f"{a_}|{b_}"] = {
                "diff": D_, "ci95": [agg_[0], agg_[1]], "p": agg_[2], "n_blocks": nb, **tint_}
    out["block_size_sensitivity"] = bss
    if coverage_acc:
        out["coverage"] = {nm: {k: float(np.mean([c_[k] for c_ in v])) for k in v[0]}
                           for nm, v in coverage_acc.items()}

    out["bootstrap"] = {"n_resamples": N_BOOT, "block_m": BLOCK_M,
                        "n_blocks": int(n_block), "p_floor": 1.0 / N_BOOT}
    out["diagnostics"] = {f"{k[0]}@{k[1]}": [round(x, 3) for x in np.mean(v, 0)]
                          for k, v in diag.items()}
    out["allocation"] = {str(T): np.mean(v, 0).round(3).tolist()
                         for T, v in alloc.items()}
    # same traces for the controller the headline number actually uses
    out["allocation_headline"] = {str(T): np.mean(v, 0).round(3).tolist()
                                  for T, v in alloc_h.items()}
    out["allocation_methods"] = {"allocation": PRIMARY,
                                 "allocation_headline": HEADLINE}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', nargs='+', default=['city_2_chicago_28', 'city_3_houston_28'])
    ap.add_argument('--out', default='results.json')
    ap.add_argument('--free-compute', action='store_true',
                    help="charge probe time only, as Graff et al. Eq. (34) does")
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help="override the seed list in force, e.g. 17 29 43 for "
                         "the prespecified analysis")
    ap.add_argument('--validation', default='validation.json',
                    help="the validation run the deployable baseline is picked from")
    ap.add_argument('--lat', default='lat_cpu1t_pcore.json', help="latency file")
    ap.add_argument('--tau-scale', type=float, default=1.0,
                    help="rho: scale every inference charge (sweeps)")
    ap.add_argument('--tp-air-us', type=float, default=None,
                    help="probe air time, us (sweeps); default obs.T_P_US")
    ap.add_argument('--tables-suffix', default='',
                    help="read tables_<seed><suffix>.npz and scheduler<suffix>.json")
    ap.add_argument('--oneshot-charge', choices=('fresh', 'restart'), default=None,
                    help="what the one-shot ee baselines pay: one fresh invocation (default) "
                         "or the restart sum")
    ap.add_argument('--oracle-mix', type=float, default=None,
                    help="predictor quality q; default: the trained predictor")
    ap.add_argument('--rank-charge', action='store_true',
                    help="charge picking the next codeword after an inference call")
    ap.add_argument('--zhang-np2', action='store_true',
                    help="also evaluate Zhang et al.'s descent as published")
    ap.add_argument('--rank-exact', action='store_true',
                    help="count the ranking on the probes after the first call only, on "
                         "kappa-major tables; implies --rank-charge")
    ap.add_argument('--c-lookup-aug', choices=('file', 'original'), default='file',
                    help="the augmented table's lookup charge (as sched.py)")
    ap.add_argument('--zhang-guard', action='store_true',
                    help="also evaluate both adaptive forms with a deadline guard")
    a = ap.parse_args()
    ORACLE_MIX = a.oracle_mix
    RANK_CHARGE, ZHANG_NP2 = a.rank_charge, a.zhang_np2
    RANK_EXACT, LOOKUP_AUG, ZHANG_GUARD = a.rank_exact, a.c_lookup_aug, a.zhang_guard
    if RANK_EXACT:
        RANK_CHARGE = True
    if a.seeds:
        SEEDS = tuple(a.seeds)
    VALIDATION = a.validation
    LAT_FILE, TAU_SCALE, TP_AIR_US = a.lat, a.tau_scale, a.tp_air_us
    TABLES_SUFFIX = a.tables_suffix
    if a.oneshot_charge is not None:
        ONESHOT_FRESH = a.oneshot_charge == 'fresh'
    t0 = time.time()
    P = json.load(open(os.path.join(HERE, 'protocol.json')))
    grid = P['coherence_grid_us']
    lat = json.load(open(os.path.join(HERE, LAT_FILE)))
    tables = {s: dict(np.load(os.path.join(HERE, f'tables_{s}{TABLES_SUFFIX}.npz')))
              for s in SEEDS}
    # A variant is evaluated only where every seed has its tables for every
    # deadline, so a run solving a subset (the sweeps) evaluates that subset
    # instead of failing on the first missing key.
    SCHEDULERS = [row for row in SCHEDULERS
                  if all(f'{row[1]}_{T}' in tables[s] for s in SEEDS for T in grid)]
    if RANK_EXACT:                     # the variants solved on kappa-major tables
        AUG_KEYS = tuple(json.load(open(os.path.join(
            HERE, f'scheduler{TABLES_SUFFIX}.json'))).get('augmented_variants', ()))
    check_charge(lat)
    res = {"protocol_sha256": P['self_sha256'], "profile": lat['env']['profile'],
           "accounting": "probe-only (Graff Eq. 34)" if a.free_compute else "joint",
           "grid_us": grid, "seeds": list(SEEDS), "sites": []}
    # oneshot_fresh: the one-shot baselines pay one fresh invocation (true) or the restart sum
    extra = {"lat_file": LAT_FILE, "tau_scale": TAU_SCALE, "tp_air_us": TP_AIR_US,
             "tables_suffix": TABLES_SUFFIX, "oneshot_fresh": ONESHOT_FRESH}
    if extra != {"lat_file": 'lat_cpu1t_pcore.json', "tau_scale": 1.0, "tp_air_us": None,
                 "tables_suffix": '', "oneshot_fresh": False}:
        res["charging"] = extra        # absent only with every charge at its original setting
    if ORACLE_MIX is not None:         # absent otherwise, so no schema moves
        res.setdefault("charging", extra)["oracle_mix"] = ORACLE_MIX
    if RANK_CHARGE:                    # absent otherwise, so no schema moves
        res.setdefault("charging", extra)["c_rank_us"] = charges(lat)['C_RANK']
    if ZHANG_NP2:
        res.setdefault("charging", extra)["zhang_np2"] = True
    if RANK_EXACT:                     # absent otherwise, so no schema moves
        q_ = charges(lat)
        res.setdefault("charging", extra).update(
            c_rank_mode="exact", c_lookup_aug_us=q_['C_LOOK_AUG'], c_lookup_aug_source=LOOKUP_AUG,
            augmented_variants=list(AUG_KEYS))
    if ZHANG_GUARD:
        res.setdefault("charging", extra).update(zhang_guard=True,
                                                 c_guard_us=charges(lat)['C_GUARD'])
    try:
        import subprocess
        res["code_commit"] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE,
                                            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    for s in a.sites:
        print(f"== {s} ==", flush=True)
        res['sites'].append(evaluate_site(s, grid, lat, tables,
                                          free_compute=a.free_compute))
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(HERE, a.out), 'w'), indent=1)
    print(f"\nwrote {a.out} in {res['seconds']}s")
