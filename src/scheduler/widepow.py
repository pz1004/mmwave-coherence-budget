#!/usr/bin/env python3
"""Rule R45: wide beams under a per-element power limit (protocol.json analysis 'widepow', declared
before it ran; R45 is post hoc).

The model's codewords are unit-norm with the total transmit power fixed (obs.hierarchical_codebook):
a layer-l codeword, formed on 2^l of the 64 elements, drives each active element at 64 / 2^l times a
narrow beam's per-element power.  With every element limited to a narrow beam's level, a layer-l
codeword's power falls by 2^l / 64 (-3 dB at layer 5 to -15 dB at layer 1).  R45 scales every wide
codeword's band-averaged power so before its measurement, and changes nothing else:

(a) Hier-6 (methods.hierarchical, six layers, under the probe law), as refbounds.py runs it;
(b) the SS-block-guided refinement (ssbref.guided, layers 3 and 4, SS blocks scaled by 1/8 and
    1/4): SS blocks under the probe law with 1,000 samples, the refinement's narrow beams under each
    of ssbref.py's seven laws, on ssbref.py's draws in its order.

Measured efficiencies only (the mean over seeds of the selected narrow beam's user-mean rate over
the genie-aided rate's), per scenario of the seven non-training ones, on all users and on rule
R37's covered population; no bound is computed.  The draws are the unscaled readings' (common random
numbers): csirs.draws on evaluate.stable_seed(scenario, seed) and csirs.py's per-element stream.

Gate W1, before anything is written: unscaled, (a) reproduces refbounds.json's Hier-6 efficiency and
(b) ssbref.json's for every layer and law, in every scenario, to 1e-12.

usage:  python3 widepow.py [--gate]      -> widepow.json   (--gate runs W1 and writes nothing)
"""
import hashlib, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KEY = 'widepow'
OUT = os.path.join(HERE, 'widepow.json')
CQI1 = 2 * 78 / 1024
TOL = 1e-12


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def scenario(name, seeds, t_probe, scaled):
    """Per method key, the (S, n) selected narrow-beam rates; the genie-aided rate."""
    import torch, sim, obs, csirs, methods, evaluate, ssbref
    L6 = obs.LAYER6_OFFSET
    site = sim.Site(name)
    lay = torch.tensor(obs.layer_of(np.arange(obs.N_CODE)), device=sim.DEV, dtype=torch.float64)
    scale = (2.0 ** lay) / 64.0 if scaled else torch.ones_like(lay)
    pbar = site.pbar * scale[None, :]
    sel = {}
    for s in seeds:
        z, q = csirs.draws(site, torch.Generator(device=sim.DEV).manual_seed(evaluate.stable_seed(name, s)))
        rng = np.random.default_rng(evaluate.stable_seed('csirs', 'per_re', name, s))
        pbar64 = site.pbar[:, L6:].cpu().numpy()
        y = {}
        for est, ns in ([('coherent', n) for n in csirs.COHERENT_N]
                        + [('per_re', n) for n in csirs.PER_RE_N]):
            key = f'{est}|{ns}'
            if est == 'coherent':
                if key not in ssbref.LAWS:
                    continue
                yy = csirs.coherent(site.pbar, z, q, ns)
            else:
                yy = torch.zeros_like(site.pbar)          # every law drawn, in csirs.py's order
                yy[:, L6:] = torch.tensor(csirs.per_re(pbar64, rng, ns), device=sim.DEV)
                if key not in ssbref.LAWS:
                    continue
            y[key] = yy
        y_probe = csirs.coherent(pbar, z, q, csirs.N_PROBE)       # wide codewords scaled, or not
        _, c = methods.hierarchical(site, y_probe, 6, t_probe)
        sel.setdefault('hier6', []).append(site.se.gather(1, c[:, None])[:, 0].double().cpu().numpy())
        for k in ssbref.LAWS:
            for l in ssbref.LAYERS:
                b, _ = ssbref.guided(y_probe, y[k][:, L6:], l)
                sel.setdefault(f'guided{l}|{k}', []).append(
                    site.sn.gather(1, b[:, None])[:, 0].double().cpu().numpy())
    return {k: np.stack(v) for k, v in sel.items()}, site.oracle_se.double().cpu().numpy()


def main():
    import evaluate, protocol, csirs
    gate_only = '--gate' in sys.argv
    if not gate_only:
        P = json.load(open(os.path.join(HERE, 'protocol.json')))
        A = [a for a in P['analyses'] if a['key'] == KEY]
        assert len(A) == 1 and A[0]['scripts_sha256']['widepow.py'] == sha(os.path.abspath(__file__)), \
            'widepow.py is not the declared script'
        assert not os.path.exists(OUT), f'{OUT} exists: written once'
    seeds = protocol.seeds()
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))['T_PROBE']
    RB = json.load(open(os.path.join(HERE, 'refbounds.json')))['references']['hier6']['per_scenario']
    SB = json.load(open(os.path.join(HERE, 'ssbref.json')))['references']
    t0 = time.time()
    w1, per = {}, {}
    for name in csirs.SCEN:
        sel0, gse = scenario(name, seeds, t_probe, scaled=False)
        eta = lambda S, m=None: float(S[:, m].mean(1).mean() / gse[m].mean()) if m is not None \
            else float(S.mean(1).mean() / gse.mean())
        w1[name] = abs(eta(sel0['hier6']) - RB[name]['eta']) <= TOL and all(
            abs(eta(v) - SB[f"guided{k.split('|')[0][6:]}"]['|'.join(k.split('|')[1:])]['per_scenario'][name]['eta'])
            <= TOL for k, v in sel0.items() if k.startswith('guided'))
        if not gate_only:
            sel1, _ = scenario(name, seeds, t_probe, scaled=True)
            keep = gse >= CQI1
            per[name] = {k: {'eta_all': eta(sel1[k]), 'eta_covered': eta(sel1[k], keep),
                             'eta_all_unscaled': eta(sel0[k]), 'eta_covered_unscaled': eta(sel0[k], keep)}
                         for k in sel1}
        print(f'  {name}: done ({time.time() - t0:.0f} s)', flush=True)
    print(json.dumps({'W1': all(w1.values())}), flush=True)
    if not all(w1.values()):
        raise SystemExit(f'gate W1 failed: {w1}; nothing written')
    if gate_only:
        print('gate W1 holds; nothing written')
        return
    summary = {}
    keys = sorted(next(iter(per.values())))
    for k in keys:
        for pop in ('all', 'covered'):
            summary[f'{k}|{pop}'] = {
                'eta_min': min(per[n][k][f'eta_{pop}'] for n in per),
                'eta_min_unscaled': min(per[n][k][f'eta_{pop}_unscaled'] for n in per),
                'argmin': min(per, key=lambda n: per[n][k][f'eta_{pop}'])}
    for l in (3, 4):
        for pop in ('all', 'covered'):
            ks = [k for k in keys if k.startswith(f'guided{l}|')]
            summary[f'guided{l}|lowest_law|{pop}'] = {
                'eta_min': min(summary[f'{k}|{pop}']['eta_min'] for k in ks),
                'eta_min_unscaled': min(summary[f'{k}|{pop}']['eta_min_unscaled'] for k in ks)}
    out = {'rule': 'R45', 'script_sha256': sha(os.path.abspath(__file__)), 'seeds': list(seeds),
           'cqi1': CQI1, 'gate': {'W1': w1}, 'per_scenario': per, 'summary': summary,
           'inputs_sha256': {'refbounds.json': sha(os.path.join(HERE, 'refbounds.json')),
                             'ssbref.json': sha(os.path.join(HERE, 'ssbref.json'))},
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, v in summary.items():
        if 'lowest_law' in k or k.startswith('hier6'):
            print(f'  {k}: {v["eta_min_unscaled"]:.4f} -> {v["eta_min"]:.4f}')
    print(f'wrote {OUT} in {out["seconds"]} s')


if __name__ == '__main__':
    main()
