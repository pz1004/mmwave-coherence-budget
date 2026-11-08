#!/usr/bin/env python3
"""Rule R31: an SS-block-guided CSI-RS reference, its efficiency and its simultaneous lower bounds.
Declared in protocol.json (analysis 'tdd_ssb') before it ran.

The reference.  NR beam management commonly selects a wide beam from the SS burst and refines it
with CSI-RS.  Here the SS blocks are the 2^l codewords of the model's layer l (l = 3 or 4: 8 or 16
blocks, each on the array's first 2^l elements), measured once with the probe model of Section III
(sim.measure, N = 1000 samples), before the acquisition starts, so that the refinement alone is
timed (nrtdd.guided); the refinement is one aperiodic CSI-RS set of m = 2^(6-l) resources (8 or 4),
the narrow beams whose directions lie nearest the strongest measured block's, b m - m/2, ...,
b m + m/2 - 1 (mod 64): a layer-l codeword b points where narrow beam b m does (obs.py's codebook),
so its descendants b m, ..., b m + m - 1 cover only half its beam.  The UE reports the CRI of its
largest measurement, one report, so the 1 dB grid cannot change the choice.  Each resource is
measured under csirs.py's laws (coherent at 1000, 198, 66 and 33 samples; energy detection per
element at 198, 66 and 33) on csirs.py's draws, so every selection is paired with csirs.json's sweep.

Statistics, per reference, law and scenario (the seven non-training scenarios, all eligible users, the
10 seeds): eta, the mean over seeds and users of the selected beam's rate over the genie-aided
beam's (csirs.py's definition); the delta-method standard error (csirs.lin_se); the one-sided lower
bound at ALPHA = 0.05/14, simultaneous over the seven scenarios and the two references, the smaller
of the crossed bootstrap's and the delta method's (effbound.lower_bound, its scan from zero), both
halves stored; descriptive shares: users whose genie-aided beam lies in the measured block's window,
users whose measured block is the noiseless strongest.

Gates, before anything is written: S1 csirs.py's gate (i), the coherent law at N = 1000 reproducing
sim.measure bit for bit, every scenario and seed; S2 the sweep under every law used reproduces
csirs.json's efficiency and per-seed efficiencies (to 1e-12), so the draws are csirs.py's; S3 for each
l the 2^l windows partition the 64 beams, m each; S4 bound <= eta < 1 everywhere, and the scans'
own asserts.  --check-only runs S1-S3 and computes no statistic of the reference.

usage:  python3 ssbref.py [--check-only]       -> ssbref.json
"""
import hashlib, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, obs, methods, evaluate, protocol, csirs, effbound

SCEN = csirs.SCEN
L6 = obs.LAYER6_OFFSET
LAYERS = (3, 4)
LAWS = ['coherent|1000', 'coherent|198', 'coherent|66', 'coherent|33',
        'per_re|198', 'per_re|66', 'per_re|33']
ALPHA = 0.05 / (len(SCEN) * len(LAYERS))
WINDOW = 1.0                    # the scan starts at zero, where every efficiency's test passes
TOL = 1e-12


def windows(l):
    """Per layer-l codeword b, the m = 2^(6 - l) narrow beams nearest its direction, as a (2^l, m)
    tensor of narrow-beam indices."""
    m = 2 ** (obs.N_LAYER - l)
    b = torch.arange(2 ** l, device=sim.DEV)[:, None]
    return (b * m - m // 2 + torch.arange(m, device=sim.DEV)[None, :]) % obs.M_BEAMS


def guided(y_ssb, y_csi, l):
    """The strongest measured SS block (layer l, under y_ssb), then the strongest of its window under
    y_csi (narrow beams, (n, 64)); first maxima, as methods.topk.  Returns the narrow beam and the
    block."""
    lo = 2 ** l - 2
    blk = y_ssb[:, lo:lo + 2 ** l].argmax(1)
    win = windows(l)[blk]                                   # (n, m)
    pick = y_csi.gather(1, win).argmax(1)
    return win.gather(1, pick[:, None])[:, 0], blk


def scenario(name, seeds, t_probe):
    """Per seed, law and reference, the selected rates (S, n); the genie-aided rate; the blocks;
    gates S1 and S2; descriptive shares."""
    site = sim.Site(name)
    sel = {(l, k): [] for l in LAYERS for k in LAWS}
    sw = {k: [] for k in LAWS}
    hit = {l: [] for l in LAYERS}
    blk_ok = {l: [] for l in LAYERS}
    bit = True
    best_blk = {l: site.pbar[:, 2 ** l - 2:2 ** (l + 1) - 2].argmax(1) for l in LAYERS}
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
                if key not in LAWS:
                    continue                                 # deterministic in (z, q): no stream used
                yy = csirs.coherent(site.pbar, z, q, ns)
                if ns == csirs.N_PROBE:
                    bit &= bool(torch.equal(yy, y_ref))
            else:
                yy = torch.zeros_like(site.pbar)              # every law drawn, in csirs.py's order
                yy[:, L6:] = torch.tensor(csirs.per_re(pbar64, rng, ns), device=sim.DEV)
                if key not in LAWS:
                    continue
            y[key] = yy
        y_ssb = y['coherent|1000']
        for k in LAWS:
            _, c = methods.topk(site, y[k], 64, t_probe)
            sw[k].append(site.se.gather(1, c[:, None])[:, 0].double().cpu().numpy())
            for l in LAYERS:
                b, blk = guided(y_ssb, y[k][:, L6:], l)
                sel[(l, k)].append(site.sn.gather(1, b[:, None])[:, 0].double().cpu().numpy())
        for l in LAYERS:             # the block depends on the SS-block measurement alone
            blk = guided(y_ssb, y_ssb[:, L6:], l)[1]
            hit[l].append(float((((site.oracle_beam - windows(l)[blk][:, 0]) % 64)
                                 < 2 ** (6 - l)).double().mean()))
            blk_ok[l].append(float((blk == best_blk[l]).double().mean()))
        print(f"  [{name} seed {s}] done", flush=True)
    gse = site.oracle_se.double().cpu().numpy()
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))['configs']
    s2 = True
    for k in LAWS:
        st = np.stack(sw[k])
        r = CS[k]['per_scenario'][name]
        s2 &= abs(float(st.mean(1).mean() / gse.mean()) - r['eta']) <= TOL
        s2 &= bool(np.allclose(st.mean(1) / gse.mean(), r['eta_seeds'], rtol=0, atol=TOL))
    blocks = evaluate.block_ids(site.rx)
    return ({key: np.stack(v) for key, v in sel.items()}, gse, blocks, int(blocks.max()) + 1,
            {"S1_bit_for_bit": bit, "S2_sweep_reproduces_csirs_json": bool(s2)},
            {l: float(np.mean(v)) for l, v in hit.items()},
            {l: float(np.mean(v)) for l, v in blk_ok.items()})


def main():
    check = '--check-only' in sys.argv
    seeds = protocol.seeds()
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    s3 = all(sorted(windows(l).flatten().tolist()) == list(range(64))
             and windows(l).shape == (2 ** l, 2 ** (6 - l)) for l in LAYERS)
    t0 = time.time()
    gate, per = {"S3_windows_partition": s3, "per_scenario": {}}, {}
    for n in SCEN:
        sel, gse, blocks, nb, g, hit, blk_ok = scenario(n, seeds, t_probe)
        gate["per_scenario"][n] = g
        if not (g["S1_bit_for_bit"] and g["S2_sweep_reproduces_csirs_json"] and s3):
            raise SystemExit(f"gate failed in {n}: {g}; nothing written")
        if check:
            continue
        for l in LAYERS:
            for k in LAWS:
                sk = sel[(l, k)]
                e = float(sk.mean(1).mean() / gse.mean())
                tag = f"{n}|guided{l}|{k}"
                sl = csirs.lin_se(sk, gse, blocks, nb, e, f"{tag}|lin")
                lb, e_b, e_t, flips = effbound.lower_bound(sk, gse, blocks, nb, e, sl, f"{tag}|lb",
                                                           ALPHA, WINDOW)
                assert lb <= e < 1, (tag, lb, e)
                per.setdefault(f"guided{l}", {}).setdefault(k, {})[n] = {
                    "eta": e, "eta_seeds": (sk.mean(1) / gse.mean()).tolist(), "se_lin": sl,
                    "eta_lb": lb, "eta_lb_boot": e_b, "eta_lb_t": e_t, "scan_transitions": flips,
                    "window_hit": hit[l], "block_is_noiseless_best": blk_ok[l]}
        del sel
        print(f"{n}: done ({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps({"S3": s3, **{n: g for n, g in gate["per_scenario"].items()}}, indent=1))
    if check:
        print(f"gates hold; no statistic of the reference computed, nothing written "
              f"({time.time() - t0:.0f} s)")
        return
    gate["S4_bound_below_eta_below_one"] = True
    out = {"script_sha256": hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "csirs_json_sha256": hashlib.sha256(
               open(os.path.join(HERE, 'csirs.json'), 'rb').read()).hexdigest(),
           "seeds": list(seeds), "scenarios": SCEN, "layers": list(LAYERS), "laws": LAWS,
           "alpha": ALPHA, "family": len(SCEN) * len(LAYERS), "window": WINDOW,
           "windows": {str(l): windows(l).tolist() for l in LAYERS},
           "rule": ("R31: the SS-block-guided reference's efficiency per law is stated at its lowest "
                    "over the seven scenarios, and its bound at its lowest simultaneous bound, "
                    "one-sided at 0.05/14, each bound's bootstrap and t halves stored"),
           "gate": gate, "references": {}}
    for ref, laws in per.items():
        rec = {}
        for k, ps in laws.items():
            lo_at = min(ps, key=lambda n: ps[n]['eta_lb'])
            mn_at = min(ps, key=lambda n: ps[n]['eta'])
            rec[k] = {"per_scenario": ps, "eta_min": ps[mn_at]['eta'], "eta_min_at": mn_at,
                      "eta_lb_min": ps[lo_at]['eta_lb'], "eta_lb_min_at": lo_at,
                      "eta_lb_boot_at_min": ps[lo_at]['eta_lb_boot'],
                      "eta_lb_t_at_min": ps[lo_at]['eta_lb_t'],
                      "eta_lb_t_min": min(r['eta_lb_t'] for r in ps.values())}
            print(f"{ref} {k}: eta {rec[k]['eta_min']:.4f} ({mn_at}), bound "
                  f"{rec[k]['eta_lb_min']:.4f} ({lo_at}; boot {rec[k]['eta_lb_boot_at_min']:.4f}, "
                  f"t {rec[k]['eta_lb_t_at_min']:.4f}), lowest t {rec[k]['eta_lb_t_min']:.4f}",
                  flush=True)
        out["references"][ref] = rec
    out["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, 'ssbref.json')
    assert not os.path.exists(p), "ssbref.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote ssbref.json in {out['seconds']}s")


if __name__ == '__main__':
    main()
