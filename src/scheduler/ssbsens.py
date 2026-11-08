#!/usr/bin/env python3
"""Rule R34: rule R31's SS-block-guided reference with its SS blocks measured on fewer samples.
Declared in protocol.json (analysis 'ue_ssb') before it ran.

ssbref.py measures the SS blocks once with the probe model (N = 1000 samples), as Section V's probes
are, which favors the reference.  Here the SS-block stage takes csirs.py's coherent law at 198, 66
and 33 samples, the counts of the CSI-RS laws at densities 3, 1 and 0.5, on csirs.py's draws; only the
coherent law draws the wide codewords (csirs.per_re draws the 64 narrow beams alone).  Each SS law is
paired with the six CSI-RS laws of the paper at both layers; windows, selection, draws, scenarios
and seeds are ssbref.py's (ssbref.windows, ssbref.guided).

Statistics, per SS law, layer, CSI-RS law and scenario: eta and the per-seed efficiencies, the
delta-method standard error (csirs.lin_se), the one-sided lower bound at 0.05/14, simultaneous over
the seven scenarios and two layers as in R31 (effbound.lower_bound, both halves stored), the share
of users whose genie-aided beam lies in the measured block's window.  Per SS law and layer, the
lowest efficiency over the scenarios and the six laws, density 3's coherent bound and the smallest
bound over the laws, each with its t half.

Gates, before anything is written: S1 csirs.py's gate (i), the coherent law at N = 1000 reproducing
sim.measure bit for bit; G2 with the SS stage at coherent|1000, every law's per-seed efficiencies
reproduce ssbref.json's to 1e-12, so the selection is ssbref.py's; S3 each layer's windows
partition the 64 beams; S4 bound <= eta < 1, and the scans' own asserts.  --check-only runs S1, G2
and S3 and computes no statistic of the SS laws.

usage:  python3 ssbsens.py [--check-only]      -> ssbsens.json
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, methods, evaluate, protocol, csirs, effbound, ssbref

SCEN, LAYERS, L6 = ssbref.SCEN, ssbref.LAYERS, ssbref.L6
SS_LAWS = ['coherent|198', 'coherent|66', 'coherent|33']
LAWS6 = [k for k in ssbref.LAWS if k != 'coherent|1000']
ALPHA, WINDOW, TOL = ssbref.ALPHA, ssbref.WINDOW, ssbref.TOL


def scenario(name, seeds, t_probe, ref):
    """Per seed, SS law, layer and CSI-RS law, the selected rates; gates S1 and G2; window hits."""
    site = sim.Site(name)
    keys = [('coherent|1000', l, k) for l in LAYERS for k in ssbref.LAWS] + \
        [(s, l, k) for s in SS_LAWS for l in LAYERS for k in LAWS6]
    sel = {key: [] for key in keys}
    hit = {(s, l): [] for s in SS_LAWS for l in LAYERS}
    bit = True
    for s in seeds:
        y_ref = sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        z, q = csirs.draws(site, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        rng = np.random.default_rng(evaluate.stable_seed('csirs', 'per_re', name, s))
        pbar64 = site.pbar[:, L6:].cpu().numpy()
        y = {}
        for est, ns in ([('coherent', n) for n in csirs.COHERENT_N]
                        + [('per_re', n) for n in csirs.PER_RE_N]):
            key = f"{est}|{ns}"
            if est == 'coherent':
                if key not in ssbref.LAWS:
                    continue                                 # deterministic in (z, q): no stream used
                yy = csirs.coherent(site.pbar, z, q, ns)
                if ns == csirs.N_PROBE:
                    bit &= bool(torch.equal(yy, y_ref))
            else:
                yy = torch.zeros_like(site.pbar)              # every law drawn, in csirs.py's order
                yy[:, L6:] = torch.tensor(csirs.per_re(pbar64, rng, ns), device=sim.DEV)
                if key not in ssbref.LAWS:
                    continue
            y[key] = yy
        for sl, l, k in keys:
            b, _ = ssbref.guided(y[sl], y[k][:, L6:], l)
            sel[(sl, l, k)].append(site.sn.gather(1, b[:, None])[:, 0].double().cpu().numpy())
        for sl, l in hit:
            blk = ssbref.guided(y[sl], y[sl][:, L6:], l)[1]
            hit[(sl, l)].append(float((((site.oracle_beam - ssbref.windows(l)[blk][:, 0]) % 64)
                                       < 2 ** (6 - l)).double().mean()))
        print(f"  [{name} seed {s}] done", flush=True)
    gse = site.oracle_se.double().cpu().numpy()
    g2 = True
    for l in LAYERS:
        for k in ssbref.LAWS:
            st = np.stack(sel[('coherent|1000', l, k)])
            r = ref[f'guided{l}'][k]['per_scenario'][name]
            g2 &= bool(np.allclose(st.mean(1) / gse.mean(), r['eta_seeds'], rtol=0, atol=TOL))
            g2 &= abs(float(st.mean(1).mean() / gse.mean()) - r['eta']) <= TOL
    blocks = evaluate.block_ids(site.rx)
    return ({key: np.stack(v) for key, v in sel.items() if key[0] in SS_LAWS}, gse, blocks,
            int(blocks.max()) + 1,
            {"S1_bit_for_bit": bit, "G2_reproduces_ssbref_json": bool(g2)},
            {key: float(np.mean(v)) for key, v in hit.items()})


def main():
    check = '--check-only' in sys.argv
    seeds = protocol.seeds()
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    REF = json.load(open(os.path.join(HERE, 'ssbref.json')))
    assert (REF['seeds'], REF['scenarios'], REF['layers'], REF['laws']) == \
        (list(seeds), SCEN, list(LAYERS), ssbref.LAWS)
    s3 = all(sorted(ssbref.windows(l).flatten().tolist()) == list(range(64))
             and ssbref.windows(l).shape == (2 ** l, 2 ** (6 - l)) for l in LAYERS)
    t0 = time.time()
    gate, per = {"S3_windows_partition": s3, "per_scenario": {}}, {}
    for n in SCEN:
        sel, gse, blocks, nb, g, hit = scenario(n, seeds, t_probe, REF['references'])
        gate["per_scenario"][n] = g
        if not (g["S1_bit_for_bit"] and g["G2_reproduces_ssbref_json"] and s3):
            raise SystemExit(f"gate failed in {n}: {g}; nothing written")
        if check:
            continue
        for (sl, l, k), sk in sel.items():
            e = float(sk.mean(1).mean() / gse.mean())
            tag = f"{n}|ss:{sl}|guided{l}|{k}"
            se = csirs.lin_se(sk, gse, blocks, nb, e, f"{tag}|lin")
            lb, e_b, e_t, flips = effbound.lower_bound(sk, gse, blocks, nb, e, se, f"{tag}|lb",
                                                       ALPHA, WINDOW)
            assert lb <= e < 1, (tag, lb, e)
            per.setdefault(sl, {}).setdefault(f"guided{l}", {}).setdefault(k, {})[n] = {
                "eta": e, "eta_seeds": (sk.mean(1) / gse.mean()).tolist(), "se_lin": se,
                "eta_lb": lb, "eta_lb_boot": e_b, "eta_lb_t": e_t, "scan_transitions": flips,
                "window_hit": hit[(sl, l)]}
        del sel
        print(f"{n}: done ({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps({"S3": s3, **gate["per_scenario"]}, indent=1))
    if check:
        print(f"gates hold; no statistic of the SS laws computed, nothing written "
              f"({time.time() - t0:.0f} s)")
        return
    gate["S4_bound_below_eta_below_one"] = True
    h = lambda f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
    out = {"script_sha256": h(os.path.basename(__file__)), "ssbref_json_sha256": h('ssbref.json'),
           "csirs_json_sha256": h('csirs.json'), "seeds": list(seeds), "scenarios": SCEN,
           "layers": list(LAYERS), "ss_laws": SS_LAWS, "laws": LAWS6, "alpha": ALPHA,
           "family": len(SCEN) * len(LAYERS), "window": WINDOW,
           "rule": ("R34: rule R31's reference with its SS blocks measured under csirs.py's coherent "
                    "law at 198, 66 and 33 samples; per SS law and layer the lowest efficiency over "
                    "the seven scenarios and six CSI-RS laws, density 3's coherent bound and the "
                    "smallest bound, one-sided at 0.05/14, each bound's bootstrap and t halves "
                    "stored"),
           "gate": gate, "references": {}, "summary": {}}
    for sl, refs in per.items():
        out["references"][sl], out["summary"][sl] = {}, {}
        for ref, laws in refs.items():
            rec = {}
            for k, ps in laws.items():
                lo_at = min(ps, key=lambda n: ps[n]['eta_lb'])
                mn_at = min(ps, key=lambda n: ps[n]['eta'])
                rec[k] = {"per_scenario": ps, "eta_min": ps[mn_at]['eta'], "eta_min_at": mn_at,
                          "eta_lb_min": ps[lo_at]['eta_lb'], "eta_lb_min_at": lo_at,
                          "eta_lb_boot_at_min": ps[lo_at]['eta_lb_boot'],
                          "eta_lb_t_at_min": ps[lo_at]['eta_lb_t'],
                          "eta_lb_t_min": min(r['eta_lb_t'] for r in ps.values())}
            d3 = rec['coherent|198']
            out["references"][sl][ref] = rec
            out["summary"][sl][ref] = {
                "eta_min": min(r['eta_min'] for r in rec.values()),
                "eta_lb_d3": d3['eta_lb_min'], "eta_lb_d3_at": d3['eta_lb_min_at'],
                "eta_lb_t_d3": d3['eta_lb_t_min'],
                "eta_lb_min": min(r['eta_lb_min'] for r in rec.values()),
                "eta_lb_t_min": min(r['eta_lb_t_min'] for r in rec.values()),
                "window_hit_min": min(r['window_hit'] for r in d3['per_scenario'].values())}
            print(sl, ref, {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in out["summary"][sl][ref].items()}, flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, 'ssbsens.json')
    assert not os.path.exists(p), "ssbsens.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote ssbsens.json in {out['seconds']}s")


if __name__ == '__main__':
    main()
