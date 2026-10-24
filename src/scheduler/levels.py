#!/usr/bin/env python3
"""The lower bounds at the level each claim needs, rule R12 at the corrected CSI-RS thresholds,
the configured sweep at the mandatory eight resources per slot, and where the bootstrap's lower
tails come from.  Declared in protocol.json (analysis 'levels') before it ran; gate L9 follows
the amendment recorded there.

1. Levels.  The paper states each exclusion for every non-training scenario, so it needs a lower
   bound on the lowest of the seven efficiencies.  The smallest of seven per-scenario bounds, each
   one-sided at alpha, is one at alpha whatever the dependence (the scenario that has the lowest
   efficiency is one of the seven); csirs.py and csirs_cfg.py use 0.05/7 per scenario and, for the
   fast references, refbounds.py 0.05/28.  Here each configuration's per-scenario bound at
   ALPHA_NEED = 0.05 and each reference's at ALPHA_NEED_REF = 0.05/4 (the table picks one of four
   references per implementation), by
   effbound.lower_bound's scan and bisection on csirs.py's and refbounds.py's streams, its bootstrap
   test asking only that at most alpha of the replicates fall at or below zero (gate L9).  Rule R18
   reads them.
2. R12 at the corrected thresholds.  nrtime.py gives the CSI-RS threshold of ResNet-18 on the GPU
   under both computation-time conditions of TS 38.214 5.4, at the declarable UE timing most
   favourable to it, for a 120 kHz or a 30 kHz control carrier and 12 or 8 resources per slot.
   R12's test (csirs.at_threshold, on its stream '|thr') runs at each, for csirs.py's exact comparison
   and the configured variants of the same resources per slot, with Holm across the seven scenarios
   as csirs.py reads it.
3. Eight per slot.  A UE must measure at least eight one-port resources in a slot (TS 38.306,
   maxNumberSSB-CSI-RS-ResourceOneTx, mandatory >= 8 in FR2); twelve need a larger capability.  At
   eight the sweep is eight sets of eight under two triggers.  csirs_cfg.py's comparison (its choose,
   'worst' ties, the 1 dB grid) with sets of eight, in the spread order and in contiguous runs.
4. Tails.  For the cells that set the declared bounds (density 3, coherent, Phoenix, in the three
   12-per-slot comparisons; Top-8, Top-16 and Top-32 in Philadelphia; Hier-6 in Phoenix): per 50 m
   block, its users, its share of the genie-aided rate, its efficiency, and the efficiency and the
   declared-level bootstrap bound without it.

Gates, before anything is written: L1 csirs.py's gates (i) and (ii); L2 the exact comparison
reproduces csirs.json (eta, se_lin and R12's test at csirs.py's threshold in every configuration and
scenario; the bound at 0.05/7 at density 3) and L3 the 12-per-slot variants csirs_cfg.json the same
way, floats to 1e-12, flags exactly; L4 the references reproduce refbounds.json (eta, se_lin, the
bound at 0.05/28); L5 nrtime's zprime_only mode reproduces csirs.nr_threshold to 1e-12, and the
corrected thresholds equal the declared values; L6 the sets of eight partition the 64 beams and their
unquantized choice is methods.topk's; L7 per user the 'worst' choice is never better than the exact
one; L8 every R12 contrast's seed share below 10% (calib_holm.gate3's formula); L9 the bound used
here, whose bootstrap test asks only that at most alpha of the replicates fall at or below zero,
equals effbound.lower_bound at the declared level on every reproduction cell (effbound also asks the
two-sided 95% interval to clear zero, which binds only above alpha = 0.025), both scanning the window
the reproduced file scanned; the scans' asserts.
--check-only runs L1-L7's reproduction parts and computes no new statistic.

usage:  python3 levels.py [--check-only]       -> levels.json
"""
import hashlib, json, os, sys, time
import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, obs, methods, evaluate, protocol, csirs, csirs_cfg, effbound, refbounds, nrtime
from reading_rules import holm

SCEN = csirs.SCEN
L6 = obs.LAYER6_OFFSET
KEYS = csirs_cfg.KEYS
BOUND_KEYS = ['coherent|198', 'coherent|66', 'coherent|33', 'coherent|1000',
              'per_re|198', 'per_re|66', 'per_re|33', 'per_re|1000']
REPRO_LB_KEYS = ['coherent|198', 'per_re|198']
ALPHA_NEED = 0.05
ALPHA_NEED_REF = 0.05 / 4
WINDOW = 1.0
VAR12 = csirs_cfg.VARIANTS                       # 'spread|worst', 'contiguous|worst'
VAR8 = ('spread8|worst', 'contiguous8|worst')
SETS8 = (8,) * 8
# the corrected thresholds as declared (nrtime.worst at ResNet-18's fastest GPU latency)
DECLARED = {'120k12': 0.9393066873786408, '30k12': 0.9480869285714285,
            '120k8': 0.97332584305835, '30k8': 0.9740620829493087}
THR_SOURCES = {'zp': ('exact',) + VAR12, '120k12': ('exact',) + VAR12, '30k12': ('exact',) + VAR12,
               '120k8': ('exact',) + VAR8, '30k8': ('exact',) + VAR8}
LOBO_CELLS = [('exact', 'coherent|198', 'city_4_phoenix_28'),
              ('spread|worst', 'coherent|198', 'city_4_phoenix_28'),
              ('contiguous|worst', 'coherent|198', 'city_4_phoenix_28'),
              ('topk8', 'ref', 'city_5_philadelphia_28'), ('topk16', 'ref', 'city_5_philadelphia_28'),
              ('topk32', 'ref', 'city_5_philadelphia_28'), ('hier6', 'ref', 'city_4_phoenix_28')]
TOL_EQ = csirs_cfg.TOL_EQ


def sets8(assign):
    """Eight sets of eight beams, each in the frozen spread order."""
    order = np.asarray(obs.spread_order()) if assign == 'spread' else np.arange(64)
    out = []
    for k in range(0, 64, 8):
        st = np.asarray(order[k:k + 8])
        out.append(st[np.argsort(csirs_cfg.POS[st], kind='stable')])
    assert sorted(np.concatenate(out).tolist()) == list(range(64)), assign
    return out


def scenario(name, seeds, t_probe):
    """Per seed and configuration the selected rates of the exact choice, the four configured
    variants and the four references, in csirs_cfg.scenario's order of draws; gates L1, L6, L7."""
    site = sim.Site(name)
    S12 = {a: csirs_cfg.sets_of(a) for a in csirs_cfg.ASSIGN}
    S8 = {a: sets8(a) for a in csirs_cfg.ASSIGN}
    sel = {v: {} for v in ('exact',) + VAR12 + VAR8}
    ref, bit, l6, l7, rates, se_seed = {}, True, True, True, {}, []
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    for s in seeds:
        y_ref = sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        z, q = csirs.draws(site, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        rng = np.random.default_rng(evaluate.stable_seed('csirs', 'per_re', name, s))
        pbar64 = site.pbar[:, L6:].cpu().numpy()
        for m, (d, c) in refbounds.runs(site, y_ref, t_probe).items():
            if m != 'sweep64':
                ref.setdefault(m, []).append(site.se.gather(1, c[:, None])[:, 0].double().cpu().numpy())
        for est, ns in ([('coherent', n) for n in csirs.COHERENT_N]
                        + [('per_re', n) for n in csirs.PER_RE_N]):
            if est == 'coherent':
                y = csirs.coherent(site.pbar, z, q, ns)
                if ns == csirs.N_PROBE:
                    bit &= bool(torch.equal(y, y_ref))
            else:
                y = torch.zeros_like(site.pbar)
                y[:, L6:] = torch.tensor(csirs.per_re(pbar64, rng, ns), device=sim.DEV)
            key = f"{est}|{ns}"
            d, c = methods.topk(site, y, 64, t_probe)
            yn = y[:, L6:]
            se_x = site.se.gather(1, c[:, None])[:, 0].double()
            sel['exact'].setdefault(key, []).append(se_x.cpu().numpy())
            if est == 'coherent' and ns == csirs.N_PROBE:
                se_seed.append(float(se_x.mean()))
                for T in grid:
                    rates.setdefault(T, []).append(float((torch.clamp(1 - d / T, min=0.0) * se_x).mean()))
            for sets_by, names in ((S12, VAR12), (S8, VAR8)):
                for a, v in zip(csirs_cfg.ASSIGN, names):
                    b_ex, k_ex, _, _ = csirs_cfg.choose(yn, site.sn, sets_by[a], 'exact')
                    if names is VAR8:
                        l6 &= bool(torch.equal(L6 + b_ex, c))
                    b_w, _, _, _ = csirs_cfg.choose(yn, site.sn, sets_by[a], 'worst')
                    se_w = site.se.gather(1, (L6 + b_w)[:, None])[:, 0].double()
                    Q = csirs_cfg.quantize(csirs_cfg.rsrp_dbm(csirs_cfg.winners(yn, sets_by[a])[1]))
                    l7 &= bool((se_w <= se_x + 1e-12).all()) and bool(
                        (Q.gather(1, k_ex[:, None])[:, 0] == Q.max(1).values).all())
                    sel[v].setdefault(key, []).append(se_w.cpu().numpy())
        print(f"  [{name} seed {s}] done", flush=True)
    st = next(x for x in json.load(open(os.path.join(HERE, csirs.STORED[name])))['sites']
              if x['site'] == name)['methods']['sweep64']
    worst_T = max(abs(float(np.mean(rates[T])) - st['per_T'][str(T)]['mean']) for T in grid)
    eta_rec = round(float(np.mean(se_seed)) / float(site.oracle_se.mean()), 4)
    gates = {"L1_csirs_gates": bool(bit and worst_T <= csirs.TOL and eta_rec == st['eta']),
             "L6_sets_of_eight": l6, "L7_worst_never_better": l7}
    blocks = evaluate.block_ids(site.rx)
    return ({v: {k: np.stack(a) for k, a in by.items()} for v, by in sel.items()},
            {m: np.stack(a) for m, a in ref.items()}, site.oracle_se.double().cpu().numpy(),
            blocks, int(blocks.max()) + 1, gates)


def same(a, b):
    return csirs_cfg.same(a, b)


def passes_at(r, alpha):
    """At most alpha of the bootstrap replicates of the contrast at or below zero.  effbound's
    passes_boot also asks the two-sided 95% interval to clear zero, which binds above alpha = 0.025;
    at the declared levels the two agree (gate L9)."""
    return r['diff'] > 0 and r['p'] / 2 <= alpha


def bound_at(selk, gse, blocks, nb, eta, se_lin, tag, alpha, window):
    """effbound.lower_bound with passes_at: min(e_boot, e_t), both one-sided at alpha."""
    lo = max(0.0, eta - window)
    grid = np.linspace(lo, eta, effbound.SCAN)
    ok = [passes_at(csirs.test(selk, gse, blocks, nb, e, tag), alpha) for e in grid]
    assert ok[0], (tag, "the scan's lowest point does not pass")
    flips = sum(a != b for a, b in zip(ok, ok[1:]))
    assert flips <= 1, (tag, "more than one pass-to-fail transition", ok)
    if all(ok):
        e_b = float(grid[-1])
    else:
        i = ok.index(False)
        a, b = float(grid[i - 1]), float(grid[i])
        while b - a > 1e-6:
            m = 0.5 * (a + b)
            a, b = (m, b) if passes_at(csirs.test(selk, gse, blocks, nb, m, tag), alpha) else (a, m)
        e_b = a
    e_t = eta - float(stats.t.ppf(1 - alpha, selk.shape[0] - 1)) * se_lin
    return min(e_b, e_t), e_b, e_t, flips


def without(selk, gse, blocks, b):
    """The data with block b's users removed, blocks renumbered."""
    keep = blocks != b
    _, inv = np.unique(blocks[keep], return_inverse=True)
    return selk[:, keep], gse[keep], inv, int(inv.max()) + 1


def lobo(selk, gse, blocks, nb, n, key, alpha):
    """Per block: users, share of the genie-aided rate, efficiency, and the efficiency and the
    bootstrap bound at alpha without it, on the full bound's stream tag."""
    eta = float(selk.mean(1).mean() / gse.mean())
    out = []
    for b in range(nb):
        m = blocks == b
        s2, g2, b2, nb2 = without(selk, gse, blocks, b)
        e2 = float(s2.mean(1).mean() / g2.mean())
        sl2 = csirs.lin_se(s2, g2, b2, nb2, e2, f"{n}|{key}|lin")
        _, e_b, _, _ = bound_at(s2, g2, b2, nb2, e2, sl2, f"{n}|{key}|lb", alpha, WINDOW)
        out.append({"block": b, "users": int(m.sum()), "genie_share": float(gse[m].sum() / gse.sum()),
                    "eta_block": float(selk[:, m].mean(0).sum() / gse[m].sum()),
                    "eta_without": e2, "eta_lb_boot_without": e_b})
    return {"eta": eta, "alpha": alpha, "blocks": out}


def main():
    check = '--check-only' in sys.argv
    seeds = protocol.seeds()
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))['T_PROBE']
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))
    CG = json.load(open(os.path.join(HERE, 'csirs_cfg.json')))
    RB = json.load(open(os.path.join(HERE, 'refbounds.json')))
    thr_zp, tau = csirs.nr_threshold(grid)
    t_max = max(grid)
    thr = {'zp': thr_zp, '120k12': nrtime.worst(tau, t_max, 12, 120)[0],
           '30k12': nrtime.worst(tau, t_max, 12, 30)[0], '120k8': nrtime.worst(tau, t_max, 8, 120)[0],
           '30k8': nrtime.worst(tau, t_max, 8, 30)[0]}
    l5 = (abs(nrtime.threshold(tau, t_max, 12, 120, 0, 0, zprime_only=True) - thr_zp) < 1e-12
          and all(abs(thr[k] - v) < 1e-12 for k, v in DECLARED.items())
          and thr['zp'] == CS['threshold']['eta'] == CG['threshold']['eta'])
    t0 = time.time()
    S, chi = len(seeds), stats.chi2.ppf(0.001, len(seeds) - 1)
    gate = {"L5_thresholds": bool(l5), "per_scenario": {}}
    l2 = l3 = l4 = True
    l9 = []
    out_src = {v: {k: {} for k in KEYS} for v in ('exact',) + VAR12 + VAR8}
    out_ref = {m: {} for m in refbounds.REFS}
    lobo_out = {}
    for n in SCEN:
        sel, ref, gse, blocks, nb, g = scenario(n, seeds, t_probe)
        gate["per_scenario"][n] = g
        for v in ('exact',) + VAR12 + VAR8:
            stored = (CS['configs'] if v == 'exact' else
                      CG['variants'][v]['configs'] if v in VAR12 else None)
            for key in KEYS:
                selk = sel[v][key]
                eta = float(selk.mean(1).mean() / gse.mean())
                se_lin = csirs.lin_se(selk, gse, blocks, nb, eta, f"{n}|{key}|lin")
                at_zp = csirs.at_threshold(selk, gse, blocks, nb, eta, se_lin, thr_zp, f"{n}|{key}|thr")
                if stored is not None:               # L2 / L3
                    r = stored[key]['per_scenario'][n]
                    ok = same({"eta": eta, "se_lin": se_lin}, {"eta": r['eta'], "se_lin": r['se_lin']})
                    ok &= same(at_zp, {f: r['at_threshold'][f] for f in at_zp})
                    if key in REPRO_LB_KEYS and not check:
                        # the window the reproduced file scanned, csirs.py's 0.25 or csirs_cfg.py's;
                        # L9 compares on it too, since bisections from different brackets end apart
                        # by less than their 1e-6 tolerance
                        w_lb = 0.25 if v == 'exact' else WINDOW
                        lb, e_b, e_t, _ = effbound.lower_bound(
                            selk, gse, blocks, nb, eta, se_lin, f"{n}|{key}|lb", csirs.ALPHA_LB, w_lb)
                        ok &= same({"eta_lb": lb, "eta_lb_boot": e_b, "eta_lb_t": e_t},
                                   {f: r[f] for f in ('eta_lb', 'eta_lb_boot', 'eta_lb_t')})
                        l9.append(bound_at(selk, gse, blocks, nb, eta, se_lin, f"{n}|{key}|lb",
                                           csirs.ALPHA_LB, w_lb)[:3] == (lb, e_b, e_t))
                    if v == 'exact':
                        l2 &= ok
                    else:
                        l3 &= ok
                if check:
                    continue
                rec = {"eta": eta, "eta_seeds": (selk.mean(1) / gse.mean()).tolist(), "se_lin": se_lin,
                       "at_threshold": {}}
                for tk, srcs in THR_SOURCES.items():
                    if v in srcs:
                        rec["at_threshold"][tk] = (at_zp if tk == 'zp' else csirs.at_threshold(
                            selk, gse, blocks, nb, eta, se_lin, thr[tk], f"{n}|{key}|thr"))
                if key in BOUND_KEYS:
                    lb, e_b, e_t, fl = bound_at(selk, gse, blocks, nb, eta, se_lin,
                                                f"{n}|{key}|lb", ALPHA_NEED, WINDOW)
                    rec.update(eta_lb_need=lb, eta_lb_need_boot=e_b, eta_lb_need_t=e_t,
                               scan_transitions_need=fl)
                    if v in VAR8:
                        lb, e_b, e_t, fl = bound_at(selk, gse, blocks, nb, eta, se_lin,
                                                    f"{n}|{key}|lb", csirs.ALPHA_LB, WINDOW)
                        rec.update(eta_lb=lb, eta_lb_boot=e_b, eta_lb_t=e_t, scan_transitions=fl)
                out_src[v][key][n] = rec
        for m in refbounds.REFS:
            sk = ref[m]
            e = float(sk.mean(1).mean() / gse.mean())
            sl = csirs.lin_se(sk, gse, blocks, nb, e, f"{n}|ref|{m}|lin")
            r = RB['references'][m]['per_scenario'][n]
            ok = same({"eta": e, "se_lin": sl}, {"eta": r['eta'], "se_lin": r['se_lin']})
            if not check:
                lb, e_b, e_t, _ = effbound.lower_bound(sk, gse, blocks, nb, e, sl, f"{n}|ref|{m}|lb",
                                                       refbounds.ALPHA_REF, refbounds.WINDOW)
                ok &= same({"eta_lb": lb, "eta_lb_boot": e_b, "eta_lb_t": e_t},
                           {f: r[f] for f in ('eta_lb', 'eta_lb_boot', 'eta_lb_t')})
                lb, e_b, e_t, fl = bound_at(sk, gse, blocks, nb, e, sl, f"{n}|ref|{m}|lb",
                                            ALPHA_NEED_REF, WINDOW)
                out_ref[m][n] = {"eta": e, "se_lin": sl, "eta_lb_need": lb, "eta_lb_need_boot": e_b,
                                 "eta_lb_need_t": e_t, "scan_transitions_need": fl}
            l4 &= ok
        if not check:
            for src, key, site in LOBO_CELLS:
                if site != n:
                    continue
                if key == 'ref':
                    lobo_out[f"{src}|{n}"] = lobo(ref[src], gse, blocks, nb, n, f"ref|{src}",
                                                  refbounds.ALPHA_REF)
                else:
                    lobo_out[f"{src}|{key}|{n}"] = lobo(sel[src][key], gse, blocks, nb, n, key,
                                                        csirs.ALPHA_LB)
        del sel, ref
        print(f"[{n}] done ({time.time() - t0:.0f} s)", flush=True)
    gate.update(L1_csirs_gates=all(g['L1_csirs_gates'] for g in gate['per_scenario'].values()),
                L2_exact_reproduces_csirs_json=bool(l2), L3_variants_reproduce_csirs_cfg_json=bool(l3),
                L4_references_reproduce_refbounds_json=bool(l4),
                L6_sets_of_eight=all(g['L6_sets_of_eight'] for g in gate['per_scenario'].values()),
                L7_worst_never_better=all(g['L7_worst_never_better'] for g in gate['per_scenario'].values()),
                L9_bound_at_is_effbound=all(l9))
    print(json.dumps({k: v for k, v in gate.items() if k != 'per_scenario'}, indent=1))
    if not all(v for k, v in gate.items() if k != 'per_scenario'):
        raise SystemExit("gate failed: nothing written")
    if check:
        print(f"gates hold; no new statistic computed, nothing written ({time.time() - t0:.0f} s)")
        return
    # R12 at each threshold: Holm across the seven scenarios, as csirs.py reads it (L8: seed shares)
    worst_share, r12 = 0.0, {}
    for tk, srcs in THR_SOURCES.items():
        for v in srcs:
            for key in KEYS:
                rows = [out_src[v][key][n]['at_threshold'][tk] for n in SCEN]
                rej = holm([max(r['p'], r['p_t']) for r in rows])
                for n, r, h in zip(SCEN, rows, rej):
                    r["holm_rejects"] = bool(h)
                    r["established"] = bool(h and r['ci95'][0] > 0 and r['t_lo'] > 0)
                    ps = {"at_threshold": r, "eta_seeds": out_src[v][key][n]['eta_seeds']}
                    r["seed_share_upper"] = csirs_cfg.seed_share(ps, S, chi)
                    worst_share = max(worst_share, r["seed_share_upper"])
                r12.setdefault(tk, {}).setdefault(v, {})[key] = {
                    "n_established": sum(r['established'] for r in rows),
                    "supported": all(r['established'] for r in rows)}
    assert worst_share < 0.10, ("L8: an R12 contrast's seed share reaches 10%", worst_share)
    gate["L8_seed_share_max_upper"] = worst_share
    summ = {v: {key: {"eta_min": min(r['eta'] for r in out_src[v][key].values()),
                      **({"eta_lb_need_min": min(r['eta_lb_need'] for r in out_src[v][key].values()),
                          "eta_lb_need_min_at": min(out_src[v][key], key=lambda n: out_src[v][key][n]['eta_lb_need'])}
                         if key in BOUND_KEYS else {}),
                      **({"eta_lb_min": min(r['eta_lb'] for r in out_src[v][key].values()),
                          "eta_lb_min_at": min(out_src[v][key], key=lambda n: out_src[v][key][n]['eta_lb'])}
                         if key in BOUND_KEYS and v in VAR8 else {})}
                for key in KEYS} for v in out_src}
    out = {"script_sha256": hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "nrtime_sha256": hashlib.sha256(open(os.path.join(HERE, 'nrtime.py'), 'rb').read()).hexdigest(),
           "seeds": list(seeds), "scenarios": SCEN, "alpha_need": ALPHA_NEED,
           "alpha_need_ref": ALPHA_NEED_REF, "alpha_declared": csirs.ALPHA_LB,
           "alpha_declared_ref": refbounds.ALPHA_REF, "window": WINDOW,
           "thresholds": thr, "tau_us": tau, "t_max_us": t_max,
           "threshold_capability": {k: list(nrtime.worst(tau, t_max, n_, c_)[1:])
                                    for k, (n_, c_) in {'120k12': (12, 120), '30k12': (12, 30),
                                                        '120k8': (8, 120), '30k8': (8, 30)}.items()},
           "sets8": {a: [s.tolist() for s in sets8(a)] for a in csirs_cfg.ASSIGN},
           "rule": ("R18: the needed-level bounds are reported beside the declared ones and "
                    "change no verdict; R12 at the corrected thresholds as R12, R14 and R16 read it"),
           "gate": gate, "sources": out_src, "summary": summ, "r12": r12,
           "references": {m: {"per_scenario": out_ref[m],
                              "eta_lb_need_min": min(r['eta_lb_need'] for r in out_ref[m].values()),
                              "eta_lb_need_min_at": min(out_ref[m], key=lambda n: out_ref[m][n]['eta_lb_need'])}
                          for m in refbounds.REFS},
           "lobo": lobo_out, "seconds": round(time.time() - t0, 1)}
    p = os.path.join(HERE, 'levels.json')
    assert not os.path.exists(p), "levels.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote levels.json in {out['seconds']}s")


if __name__ == '__main__':
    main()
