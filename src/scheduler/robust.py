#!/usr/bin/env python3
"""Rule R40: the stored inputs' robustness, read from the stored channels (protocol.json analysis
'robust', declared before it ran; R40 is post hoc).

Three readings, in the seven non-training scenarios, on all eligible users and on rule R37's
operating population (C* >= 2 x 78/1024 bit/s/Hz):

(a) Report error.  The sweep selects the largest of the 64 narrow beams' reported band-averaged
    powers, each the stored power in dB plus an independent Gaussian error of sigma dB, reported in
    1 dB steps (the floor of the dB value, as an L1-RSRP range [k, k + 1) dBm, TS 38.133), ties
    broken toward the lowest rate among the tied beams; sigma = 0, 1, 2 and 3 dB, ten realizations
    each (numpy default_rng(stable_seed('robust', 'report', scenario, sigma, r))), no other
    measurement noise.  Reported: the efficiency (the mean over realizations of the selected rate's
    user mean over the genie-aided rate's) and the per-probe sweep's slack
    (1 - eta)(T_max - 64 t_p), t_p = 20.34 us, T_max = 10 ms.
(b) Band-averaged calibration.  By Jensen's inequality a beam's rate, the mean over sub-bands of
    log2(1 + SNR_j), is at most log2(1 + mean_j SNR_j), so G_J = mean_u max_b log2(1 + P_TX
    pbar(u, b) / sigma_d^2) bounds the genie-aided rate G from above from band-averaged powers alone;
    from 1 dB reports, the upper edge of each report's range, G_J1 (the floor of the dB value plus
    1 dB), bounds it too.  Reported: x = G_J / G - 1 and x1 = G_J1 / G - 1; a lower bound on an
    efficiency calibrated with such an upper bound on G is the one computed with G times 1/(1 + x).
(c) Rate cap.  Rates capped at 948/1024 x 8 = 7.4063 bit/s/Hz, the highest entry of TS 38.214
    V19.5.0's CQI Table 2 (Table 5.2.2.1-3, also MCS Table 5.1.3.1-2's): the per-probe sweep's
    efficiency (sim.measure with the evaluation's generators and methods.topk, as refbounds.py
    selects) with every rate capped, the genie-aided rate capped likewise, against the uncapped; the
    share of users whose genie-aided rate exceeds the cap and the share of G it removes.

Gates, before anything is written: R1 uncapped, the per-probe sweep's efficiency reproduces
csirs.json's coherent|1000 efficiency in each scenario to 1e-12 (its last bits follow the order of
summation); R2 with sigma = 0 and no steps the report-error selection is the argmax of the
band-averaged power for every user; R3 per user G_J >= G - 1e-6, the stored single-precision
rates' rounding, and G_J1 >= G_J.

usage:  python3 robust.py [--gate]      -> robust.json   (--gate runs R1-R3 and writes nothing)
"""
import hashlib, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KEY = 'robust'
OUT = os.path.join(HERE, 'robust.json')
CQI1 = 2 * 78 / 1024
CAP = 948 / 1024 * 8
SIGMAS = (0, 1, 2, 3)
REALIZATIONS = 10
T_MAX_US, T_P_US = 10_000.0, 20.34


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def report_error(n, se, pdb, g, keep):
    """(a): per sigma, the efficiency per realization on all users and on the population."""
    import evaluate
    rows = np.arange(se.shape[0])
    out = {}
    for sig in SIGMAS:
        e_all, e_pop = [], []
        for r in range(REALIZATIONS):
            rng = np.random.default_rng(evaluate.stable_seed(KEY, 'report', n, sig, r))
            q = np.floor(pdb + sig * rng.standard_normal(pdb.shape))
            top = q == q.max(1, keepdims=True)
            rate = np.where(top, se, np.inf).min(1)                 # ties to the lowest rate
            e_all.append(float(rate.mean() / g.mean()))
            e_pop.append(float(rate[keep].mean() / g[keep].mean()))
        out[str(sig)] = {'eta_all': float(np.mean(e_all)), 'eta_pop': float(np.mean(e_pop)),
                         'eta_all_realizations': e_all, 'eta_pop_realizations': e_pop}
    return out


def jensen(pbar, se, keep):
    """(b): the genie-aided rate and its two band-averaged upper bounds."""
    import obs
    snr = obs.P_TX * pbar / obs.SIGMA_D2
    g = se.max(1)
    gj = np.log2(1 + snr).max(1)
    gj1 = np.log2(1 + 10 ** ((np.floor(10 * np.log10(snr)) + 1) / 10)).max(1)
    ok = bool((gj >= g - 1e-6).all() and (gj1 >= gj).all())
    rec = {}
    for name, m in (('all', np.ones(g.size, bool)), ('pop', keep)):
        rec[name] = {'G': float(g[m].mean()), 'G_J': float(gj[m].mean()),
                     'G_J1': float(gj1[m].mean()), 'x': float(gj[m].mean() / g[m].mean() - 1),
                     'x1': float(gj1[m].mean() / g[m].mean() - 1)}
    return rec, ok


def cap(n, seeds, t_probe):
    """(c): the per-probe sweep's efficiency uncapped and capped, csirs.py's arithmetic."""
    import torch, sim, methods, evaluate, obs
    site = sim.Site(n)
    L6 = obs.LAYER6_OFFSET
    se_c = torch.clamp(site.se, max=CAP)
    sel, sel_c = [], []
    for s in seeds:
        y = sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(n, s)))
        d, c = methods.topk(site, y, 64, t_probe)
        sel.append(site.se.gather(1, c[:, None])[:, 0].double().cpu().numpy())
        sel_c.append(se_c.gather(1, c[:, None])[:, 0].double().cpu().numpy())
    sel, sel_c = np.stack(sel), np.stack(sel_c)
    gse = site.oracle_se.double().cpu().numpy()
    gcap = se_c[:, L6:].max(1).values.double().cpu().numpy()
    keep = gse >= CQI1
    rec = {}
    for name, m in (('all', np.ones(gse.size, bool)), ('pop', keep)):
        rec[name] = {'eta': float(sel[:, m].mean(1).mean() / gse[m].mean()),
                     'eta_capped': float(sel_c[:, m].mean(1).mean() / gcap[m].mean()),
                     'G': float(gse[m].mean()), 'G_capped': float(gcap[m].mean()),
                     'users_above_cap': float((gse[m] > CAP).mean())}
    return rec


def main():
    import csirs, evaluate, protocol
    gate_only = '--gate' in sys.argv
    if not gate_only:
        P = json.load(open(os.path.join(HERE, 'protocol.json')))
        A = [a for a in P['analyses'] if a['key'] == KEY]
        assert len(A) == 1 and A[0]['scripts_sha256']['robust.py'] == sha(
            os.path.abspath(__file__)), "robust.py is not the declared script"
        assert not os.path.exists(OUT), f"{OUT} exists: written once"
    seeds = protocol.seeds()
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))['configs']['coherent|1000']
    t0 = time.time()
    gates, per = {'R1': {}, 'R2': {}, 'R3': {}}, {}
    for n in csirs.SCEN:
        z = np.load(os.path.join(HERE, 'data', f'{n}.npz'))
        pbar = z['pbar'][:, 62:].astype(np.float64)
        se = z['se'][:, 62:].astype(np.float64)
        g = se.max(1)
        keep = g >= CQI1
        pdb = 10 * np.log10(np.maximum(pbar, 1e-300))
        c = cap(n, seeds, t_probe)
        gates['R1'][n] = abs(c['all']['eta'] - CS['per_scenario'][n]['eta']) <= 1e-12
        argmax = np.where(pbar == pbar.max(1, keepdims=True), se, np.inf).min(1)
        gates['R2'][n] = bool(np.array_equal(argmax, se[np.arange(se.shape[0]), pbar.argmax(1)]))
        jr, gates['R3'][n] = jensen(pbar, se, keep)
        if not gate_only:
            per[n] = {'report_error': report_error(n, se, pdb, g, keep), 'jensen': jr, 'cap': c,
                      'pop_users_share': float(keep.mean())}
        print(f"  {n}: done ({time.time() - t0:.0f} s)", flush=True)
    ok = all(all(v.values()) for v in gates.values())
    print(json.dumps({k: all(v.values()) for k, v in gates.items()}), flush=True)
    if not ok:
        raise SystemExit(f"gate failed: {gates}; nothing written")
    if gate_only:
        print("gates R1-R3 hold; nothing written")
        return
    slack = lambda e: (1 - e) * (T_MAX_US - 64 * T_P_US)
    summary = {}
    for sig in map(str, SIGMAS):
        for pop in ('all', 'pop'):
            es = [per[n]['report_error'][sig][f'eta_{pop}'] for n in per]
            summary[f'report|{sig}|{pop}'] = {'eta_min': min(es), 'eta_max': max(es),
                                              'slack_max_us': slack(min(es)),
                                              'slack_min_us': slack(max(es))}
    for pop in ('all', 'pop'):
        summary[f'jensen|{pop}'] = {f: max(per[n]['jensen'][pop][f] for n in per)
                                    for f in ('x', 'x1')}
        summary[f'cap|{pop}'] = {
            'eta_shift_max': max(per[n]['cap'][pop]['eta'] - per[n]['cap'][pop]['eta_capped']
                                 for n in per),
            'G_cut_max': max(1 - per[n]['cap'][pop]['G_capped'] / per[n]['cap'][pop]['G']
                             for n in per),
            'G_cut_min': min(1 - per[n]['cap'][pop]['G_capped'] / per[n]['cap'][pop]['G']
                             for n in per),
            'users_above_cap': [min(per[n]['cap'][pop]['users_above_cap'] for n in per),
                                max(per[n]['cap'][pop]['users_above_cap'] for n in per)]}
    out = {'rule': 'R40', 'script_sha256': sha(os.path.abspath(__file__)),
           'cqi1': CQI1, 'cap': CAP, 'sigmas_db': list(SIGMAS), 'realizations': REALIZATIONS,
           't_max_us': T_MAX_US, 't_p_us': T_P_US, 'seeds': list(seeds), 'gate': gates,
           'per_scenario': per, 'summary': summary,
           'inputs_sha256': {'csirs.json': sha(os.path.join(HERE, 'csirs.json')),
                             **{f'data/{n}.npz': sha(os.path.join(HERE, 'data', f'{n}.npz'))
                                for n in csirs.SCEN}},
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT} in {out['seconds']} s")


if __name__ == '__main__':
    main()
