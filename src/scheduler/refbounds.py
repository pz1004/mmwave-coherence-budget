#!/usr/bin/env python3
"""Nominal simultaneous lower bounds on the efficiencies of the fast fixed references the screen
of Appendix B.1 uses, Top-8, Top-16, Top-32 and Hier-6.  Declared in protocol.json (analysis
'csirs_cfg_refbounds') before it ran.

The paper screens an untrained implementation against a fast reference j with the condition
d_L >= (1 - eta_j) T_max + eta_j d_j, valid for any lower bound 0 <= eta_j <= min(eta_j_true, 1).
This script gives each reference the bound the paper already gives the sweep (csirs.py's
coherent|1000 configuration): one-sided, the smaller of the crossed seed-by-block
bootstrap's bound and the delta method's, in every scenario, here at level 0.05/28 so that the
bounds hold together over the seven scenarios and the four references at a nominal 95%, because
the screen reports, per implementation, the reference that gives the longest horizon.

The selections are the evaluation's own: sim.measure on the generator seeded stable_seed(scenario,
seed) (evaluate.py), methods.topk for K = 8, 16, 32 and 64, and methods.hierarchical with six layers
and no decision charge, as evaluate.py calls them; the rate of the selected codeword is site.se.

Gates, before anything is written: H1 every reference's efficiency (to four decimals), cost and
per-period mean rates reproduce results.json, validation.json and heldout.json and their _rx
versions; H2 effbound.lower_bound at csirs.py's level and window reproduces csirs.json's
coherent|1000 bound and delta-method standard error in all seven scenarios, the sweep's selections
being sim.measure's; H3 bound <= eta < 1 for every reference and scenario, a non-positive bound
recorded as giving no horizon (rule R17).  --check-only runs
H1 and H2 and computes no reference's bound.

usage:  python3 refbounds.py [--check-only]       -> refbounds.json
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, methods, evaluate, protocol, csirs, effbound

SCEN = csirs.SCEN
REFS = {'topk8': 8, 'topk16': 16, 'topk32': 32, 'hier6': 12}     # codewords probed
ALPHA_REF = 0.05 / 28             # one-sided: 7 scenarios x 4 references
WINDOW = 1.0                      # the scan starts at zero, where every efficiency's test passes
FILES = {n: (f, f.replace('.json', '_rx.json')) for n, f in csirs.STORED.items()}


def runs(site, y, t_probe):
    out = {f'topk{k}': methods.topk(site, y, k, t_probe) for k in (8, 16, 32)}
    out['sweep64'] = methods.topk(site, y, 64, t_probe)
    out['hier6'] = methods.hierarchical(site, y, 6, t_probe)
    return out


def scenario(name, seeds, t_probe, grid):
    """Per method the (seeds, users) selected rates, and gate H1 against both stored files."""
    site = sim.Site(name)
    sel, cost, se_m, rates = {}, {}, {}, {}
    for s in seeds:
        y = sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        for m, (d, c) in runs(site, y, t_probe).items():
            se = site.se.gather(1, c[:, None])[:, 0].double()
            sel.setdefault(m, []).append(se.cpu().numpy())
            cost.setdefault(m, []).append(float(d.mean()))
            se_m.setdefault(m, []).append(float(se.mean()))
            for T in grid:
                rates.setdefault((m, T), []).append(
                    (torch.clamp(1 - d / T, min=0.0) * se).cpu().numpy())
        print(f"  [{name} seed {s}] done", flush=True)
    oracle_mean = float(site.oracle_se.mean())
    h1, worst = True, 0.0
    for f in FILES[name]:
        st = next(x for x in json.load(open(os.path.join(HERE, f)))['sites']
                  if x['site'] == name)['methods']
        for m in list(REFS) + ['sweep64']:
            h1 &= round(float(np.mean(se_m[m])) / oracle_mean, 4) == st[m]['eta']
            h1 &= abs(round(float(np.mean(cost[m])), 3) - st[m]['cost_us']) <= 1e-3
            for T in grid:
                mu = float(np.stack(rates[(m, T)]).mean(1).mean())
                worst = max(worst, abs(mu - st[m]['per_T'][str(T)]['mean']))
    blocks = evaluate.block_ids(site.rx)
    return ({m: np.stack(v) for m, v in sel.items()}, site.oracle_se.double().cpu().numpy(),
            blocks, int(blocks.max()) + 1, {"stored_reproduced": bool(h1 and worst <= csirs.TOL),
                                             "per_T_max_abs": worst},
            {m: round(float(np.mean(se_m[m])) / oracle_mean, 4) for m in list(REFS) + ['sweep64']})


def main():
    check = '--check-only' in sys.argv
    seeds = protocol.seeds()
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))
    t0 = time.time()
    gate, per = {"H1": {}, "H2": {}}, {m: {} for m in REFS}
    for n in SCEN:
        sel, gse, blocks, nb, h1, eta_st = scenario(n, seeds, t_probe, grid)
        gate["H1"][n] = h1
        # H2: the helper at csirs.py's level and window gives csirs.json's sweep bound
        sw = sel['sweep64']
        eta = float(sw.mean(1).mean() / gse.mean())
        se_lin = csirs.lin_se(sw, gse, blocks, nb, eta, f"{n}|coherent|1000|lin")
        lb, e_b, e_t, _ = effbound.lower_bound(sw, gse, blocks, nb, eta, se_lin,
                                               f"{n}|coherent|1000|lb", csirs.ALPHA_LB, 0.25)
        ref = CS['configs']['coherent|1000']['per_scenario'][n]
        gate["H2"][n] = bool(eta == ref['eta'] and se_lin == ref['se_lin'] and lb == ref['eta_lb']
                             and e_b == ref['eta_lb_boot'] and e_t == ref['eta_lb_t'])
        if not check:
            for m in REFS:
                sk = sel[m]
                e = float(sk.mean(1).mean() / gse.mean())
                sl = csirs.lin_se(sk, gse, blocks, nb, e, f"{n}|ref|{m}|lin")
                lb, e_b, e_t, flips = effbound.lower_bound(sk, gse, blocks, nb, e, sl,
                                                           f"{n}|ref|{m}|lb", ALPHA_REF, WINDOW)
                per[m][n] = {"eta": e, "eta_stored": eta_st[m],
                             "eta_seeds": (sk.mean(1) / gse.mean()).tolist(), "se_lin": sl,
                             "eta_lb": lb, "eta_lb_boot": e_b, "eta_lb_t": e_t,
                             "scan_transitions": flips}
        del sel
    ok = (all(g["stored_reproduced"] for g in gate["H1"].values()) and all(gate["H2"].values()))
    print(json.dumps({"H1": all(g["stored_reproduced"] for g in gate["H1"].values()),
                      "H2": all(gate["H2"].values())}, indent=1))
    if not ok:
        raise SystemExit("gate failed: nothing written")
    if check:
        print(f"gates hold; no reference's bound computed, nothing written ({time.time() - t0:.0f} s)")
        return
    h3 = all(r['eta_lb'] <= r['eta'] < 1 for m in REFS for r in per[m].values())
    assert h3, "H3: a bound above eta or an efficiency of one"
    gate["H3"] = h3
    gate["H3_nonpositive_bounds"] = sorted(f"{m}|{n}" for m in REFS for n, r in per[m].items()
                                           if r['eta_lb'] <= 0)
    out = {"script_sha256": hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "seeds": list(seeds), "scenarios": SCEN, "t_probe_us": t_probe, "alpha": ALPHA_REF,
           "family": len(SCEN) * len(REFS), "window": WINDOW,
           "rule": ("R17: a fast reference's horizon is stated at its lowest bound over the "
                    "seven scenarios, one-sided at 0.05/28; its lowest measured efficiency is "
                    "reported beside as a point value"),
           "gate": gate, "references": {}}
    for m, k in REFS.items():
        lo_at = min(per[m], key=lambda n: per[m][n]['eta_lb'])
        mn_at = min(per[m], key=lambda n: per[m][n]['eta_stored'])
        out["references"][m] = {"n_probes": k, "d_us": k * t_probe, "per_scenario": per[m],
                                "eta_lb_min": per[m][lo_at]['eta_lb'], "eta_lb_min_at": lo_at,
                                "eta_stored_min": per[m][mn_at]['eta_stored'],
                                "eta_stored_min_at": mn_at}
        print(f"{m}: bound {out['references'][m]['eta_lb_min']:.4f} ({lo_at}), lowest stored "
              f"{out['references'][m]['eta_stored_min']:.4f}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, 'refbounds.json')
    assert not os.path.exists(p), "refbounds.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote refbounds.json in {out['seconds']}s")


if __name__ == '__main__':
    main()
