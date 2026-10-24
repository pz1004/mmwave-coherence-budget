#!/usr/bin/env python3
"""What the accounting change does to a choice made without the evaluation scenarios.
Declared in protocol.json (analysis 'exact') before it ran; post hoc, and it replaces neither
the primary criterion nor Table II or R8.

Two selectors are frozen on the validation city, Los Angeles, from its 10-seed means:
  a(T)  the fixed method with the highest rate under airtime-only accounting (inference,
        the table lookup and the test decision free; the probe time as in every run),
  b(T)  the fixed method with the highest rate with inference latency included,
over two candidate sets: S9, the nine fixed methods of Table II (aging.PUBS), and S36, the
36 fixed methods every deployable choice is made from (evaluate.FIXED_PREFIXES; its b(T) is
the stored deployable choice).  On each evaluation scenario s,
  Delta_s = (1/12) sum_T [R_s(b(T)) - R_s(a(T))],
both rates with inference latency included, is estimated with the paper's paired
seed-by-block bootstrap and t interval (evaluate.paired_bootstrap_hier, 20,000 replicates, a
stream keyed by set and scenario).  Rule R11: Delta_s is established in a scenario when both
intervals exclude zero and the larger p-value survives Holm across the six scenarios, per set.

Per-user rates are recomputed with the evaluation's own code (aging.menu, the stored models,
the common measurement noise); the recomputed means must reproduce the stored ones, which are
rounded to five decimals, within 1e-5.

usage:  python3 selector.py [--check-only]       -> selector.json
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, evaluate, aging, protocol
from reading_rules import holm

VAL = 'city_1_losangeles_28'
EVAL = ['city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28',
        'city_5_philadelphia_28', 'city_6_miami_28', 'city_7_sandiego_28']
STORED = {VAL: 'validation.json', **{n: 'results.json' for n in EVAL[:2]},
          **{n: 'heldout.json' for n in EVAL[2:]}}
TOL = 1e-5      # the stored means are rounded to five decimals (aging.py's gate uses 1e-5)


def scenario(name, seeds, grid, lat, keep=()):
    """Seed means of every fixed method under both accountings, {m: {T: [S]}}, and the
    per-user rates with latency included, (S, n), of the methods in `keep`."""
    site = sim.Site(name)
    qf, qa = evaluate.charges(lat), evaluate.charges(lat, free_compute=True)
    mf_, ma_, rates = {}, {}, {}
    for s in seeds:
        gen = torch.Generator(device=sim.DEV).manual_seed(evaluate.stable_seed(name, s))
        y = sim.measure(site.pbar, gen)
        ee, nets = evaluate.load_nets(s)
        mf = aging.menu(site, y, y, ee, nets, qf)
        ma = aging.menu(site, y, y, ee, nets, qa)
        for m, (df, cf) in mf.items():
            if m == 'oracle':
                continue
            da, ca = ma[m]
            assert torch.equal(cf, ca), (name, s, m)      # the same beam either way
            se = site.se.gather(1, cf[:, None])[:, 0].double()
            for T in grid:
                rf = (max(0.0, 1 - df / T) * se).cpu().numpy()
                ra = (max(0.0, 1 - da / T) * se).cpu().numpy()
                mf_.setdefault(m, {}).setdefault(T, []).append(float(rf.mean()))
                ma_.setdefault(m, {}).setdefault(T, []).append(float(ra.mean()))
                if m in keep:
                    rates.setdefault(m, {}).setdefault(T, []).append(rf)
        print(f"  [{name} seed {s}] done", flush=True)
    rates = {m: {T: np.stack(v) for T, v in d.items()} for m, d in rates.items()}
    blocks = evaluate.block_ids(site.rx)
    return mf_, ma_, rates, blocks, int(blocks.max()) + 1


def gate(name, mf, grid):
    """The recomputed means against the stored ones, which are rounded to five decimals."""
    st = next(s for s in json.load(open(os.path.join(HERE, STORED[name])))['sites']
              if s['site'] == name)
    worst = max(abs(float(np.mean(mf[m][T])) - st['methods'][m]['per_T'][str(T)]['mean'])
                for m in mf for T in grid)
    return {"max_abs": worst, "holds": bool(worst <= TOL)}


def main():
    check = '--check-only' in sys.argv
    seeds = protocol.seeds()
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
    t0 = time.time()
    mf_v, ma_v, _, _, _ = scenario(VAL, seeds, grid, lat)
    gates = {VAL: gate(VAL, mf_v, grid)}
    S36 = sorted(mf_v)
    assert len(S36) == 36 and all(m.startswith(evaluate.FIXED_PREFIXES) for m in S36), S36
    S9 = list(aging.PUBS)
    mean = lambda M, m, T: float(np.mean(M[m][T]))
    sel = {}
    for set_name, cands in (('S9', S9), ('S36', S36)):
        sel[set_name] = (cands, [max(cands, key=lambda m: mean(ma_v, m, T)) for T in grid],
                         [max(cands, key=lambda m: mean(mf_v, m, T)) for T in grid])
    keep = sorted({m for c, a, b in sel.values() for m in a + b})
    data = {}
    for n in EVAL:
        data[n] = scenario(n, seeds, grid, lat, keep)
        gates[n] = gate(n, data[n][0], grid)
    print(json.dumps(gates, indent=1))
    if not all(g['holds'] for g in gates.values()):
        raise SystemExit("reproduction gate failed: nothing selected, nothing written")
    if check:
        print("reproduction gate holds; nothing written")
        return
    out = {"script_sha256": hashlib.sha256(
               open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "seeds": list(seeds), "grid_us": grid, "validation": VAL, "gate": gates,
           "rule": ("R11: established when the seed-by-block bootstrap and the t interval "
                    "both exclude zero and the larger p-value survives Holm across the six "
                    "evaluation scenarios, per candidate set"), "sets": {}}
    for set_name, (cands, a, b) in sel.items():
        margin = lambda M, T: (lambda v: v[0] - v[1])(sorted((mean(M, m, T) for m in cands),
                                                             reverse=True))
        rec = {"methods": cands, "a": a, "b": b,
               "differ_at_T": [T for T, x, y in zip(grid, a, b) if x != y],
               "a_margin": [margin(ma_v, T) for T in grid],
               "b_margin": [margin(mf_v, T) for T in grid], "scenarios": {}}
        rows = []
        for n in EVAL:
            _, _, rates, blocks, nb = data[n]
            dS = [rates[b[t]][T] - rates[a[t]][T] for t, T in enumerate(grid)]
            rng = np.random.default_rng(evaluate.stable_seed('selector', set_name, n))
            per, agg, D, tint = evaluate.paired_bootstrap_hier(dS, blocks, nb, rng)
            base = float(np.mean([rates[a[t]][T].mean() for t, T in enumerate(grid)]))
            r = {"diff": D, "rel": D / base, "base_rate": base,
                 "ci95": [agg[0], agg[1]], "p": agg[2], **tint,
                 "per_T": [{"T_us": T, "a": a[t], "b": b[t], "diff": float(dS[t].mean())}
                           for t, T in enumerate(grid)]}
            rec["scenarios"][n] = r
            rows.append(r)
        rej = holm([max(r['p'], r['p_t']) for r in rows])
        for n, r, h in zip(EVAL, rows, rej):
            lo = min(r['ci95'][0], r['ci95_t'][0])
            hi = max(r['ci95'][1], r['ci95_t'][1])
            r["holm_rejects"] = bool(h)
            r["verdict"] = +1 if (h and lo > 0) else -1 if (h and hi < 0) else 0
        rec["n_positive"] = sum(1 for r in rows if r['diff'] > 0)
        rec["n_negative"] = sum(1 for r in rows if r['diff'] < 0)
        rec["n_established_gain"] = sum(1 for r in rows if r['verdict'] > 0)
        rec["n_established_loss"] = sum(1 for r in rows if r['verdict'] < 0)
        out["sets"][set_name] = rec
        print(f"{set_name}: a != b at {rec['differ_at_T']}; Delta "
              f"{min(r['rel'] for r in rows):+.2%}..{max(r['rel'] for r in rows):+.2%}, "
              f"positive {rec['n_positive']}/6, established gain {rec['n_established_gain']}, "
              f"loss {rec['n_established_loss']}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, 'selector.json')
    assert not os.path.exists(p), "selector.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote selector.json in {out['seconds']}s")


if __name__ == '__main__':
    main()
