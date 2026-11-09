#!/usr/bin/env python3
"""Rule R46: report errors at RAN4's L1-RSRP accuracy (protocol.json analysis 'rsrpacc', declared
before it ran; R46 is post hoc).

Rule R40 read the sweep under per-beam report errors of 0-3 dB.  TS 38.133 V19.6.0 requires CSI-RS
based L1-RSRP in FR2 to meet a relative accuracy of 6.5 dB under normal conditions and 9.5 dB under
extreme ones (Table 10.1.20.2.2-1, the L1-RSRP of one CSI-RS against the largest), in at least 90% of
reported cases (A.7.7).  Read as independent Gaussian errors per beam, the difference of two beams'
errors stays within A dB with probability 0.9 when sigma = A / (z_0.95 sqrt 2): 2.794 dB for 6.5 dB
and 4.084 dB for 9.5 dB.  R46 applies robust.py's reading (a) at those sigma, unchanged: the sweep
selects the largest of the 64 narrow beams' band-averaged powers in dB plus the error, reported in
1 dB steps, ties to the lowest rate; ten realizations on numpy default_rng(evaluate.stable_seed(
'robust', 'report', scenario, sigma, r)); the efficiency and the per-probe sweep's slack
(1 - eta)(T_max - 64 t_p), in the seven non-training scenarios, on all users and on rule R37's
covered population.

Gate A1, before anything is written: at sigma = 3 dB the reading reproduces robust.json's
realizations exactly.

usage:  python3 rsrpacc.py [--gate]      -> rsrpacc.json   (--gate runs A1 and writes nothing)
"""
import hashlib, json, math, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KEY = 'rsrpacc'
OUT = os.path.join(HERE, 'rsrpacc.json')
CQI1 = 2 * 78 / 1024
Z95 = 1.6448536269514722
ACCURACY_DB = (6.5, 9.5)                      # TS 38.133 Table 10.1.20.2.2-1: normal, extreme
SIGMAS = tuple(a / (Z95 * math.sqrt(2)) for a in ACCURACY_DB)
REALIZATIONS = 10
T_MAX_US, T_P_US = 10_000.0, 20.34


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def report_error(n, se, pdb, g, keep, sig):
    """robust.report_error's reading at one sigma: per realization, all users and the population."""
    import evaluate
    e_all, e_pop = [], []
    for r in range(REALIZATIONS):
        rng = np.random.default_rng(evaluate.stable_seed('robust', 'report', n, sig, r))
        q = np.floor(pdb + sig * rng.standard_normal(pdb.shape))
        top = q == q.max(1, keepdims=True)
        rate = np.where(top, se, np.inf).min(1)                 # ties to the lowest rate
        e_all.append(float(rate.mean() / g.mean()))
        e_pop.append(float(rate[keep].mean() / g[keep].mean()))
    return {'eta_all': float(np.mean(e_all)), 'eta_pop': float(np.mean(e_pop)),
            'eta_all_realizations': e_all, 'eta_pop_realizations': e_pop}


def main():
    import csirs
    gate_only = '--gate' in sys.argv
    if not gate_only:
        P = json.load(open(os.path.join(HERE, 'protocol.json')))
        A = [a for a in P['analyses'] if a['key'] == KEY]
        assert len(A) == 1 and A[0]['scripts_sha256']['rsrpacc.py'] == sha(os.path.abspath(__file__)), \
            'rsrpacc.py is not the declared script'
        assert not os.path.exists(OUT), f'{OUT} exists: written once'
    RB = json.load(open(os.path.join(HERE, 'robust.json')))
    t0 = time.time()
    a1, per = {}, {}
    for n in csirs.SCEN:
        z = np.load(os.path.join(HERE, 'data', f'{n}.npz'))
        pbar = z['pbar'][:, 62:].astype(np.float64)
        se = z['se'][:, 62:].astype(np.float64)
        g = se.max(1)
        keep = g >= CQI1
        pdb = 10 * np.log10(np.maximum(pbar, 1e-300))
        r3 = report_error(n, se, pdb, g, keep, 3)
        st = RB['per_scenario'][n]['report_error']['3']
        a1[n] = (r3['eta_all_realizations'] == st['eta_all_realizations'] and
                 r3['eta_pop_realizations'] == st['eta_pop_realizations'])
        if not gate_only:
            per[n] = {f'{s:.3f}': report_error(n, se, pdb, g, keep, s) for s in SIGMAS}
    print(json.dumps({'A1': all(a1.values())}), flush=True)
    if not all(a1.values()):
        raise SystemExit(f'gate A1 failed: {a1}; nothing written')
    if gate_only:
        print('gate A1 holds; nothing written')
        return
    slack = lambda e: (1 - e) * (T_MAX_US - 64 * T_P_US)
    summary = {}
    for a, s in zip(ACCURACY_DB, SIGMAS):
        for pop in ('all', 'pop'):
            es = [per[n][f'{s:.3f}'][f'eta_{pop}'] for n in per]
            summary[f'{a:g}|{pop}'] = {'sigma_db': s, 'eta_min': min(es), 'eta_max': max(es),
                                       'slack_max_us': slack(min(es)), 'slack_min_us': slack(max(es))}
    out = {'rule': 'R46', 'script_sha256': sha(os.path.abspath(__file__)), 'cqi1': CQI1,
           'accuracy_db': list(ACCURACY_DB), 'sigmas_db': list(SIGMAS), 'z95': Z95,
           'realizations': REALIZATIONS, 't_max_us': T_MAX_US, 't_p_us': T_P_US, 'gate': {'A1': a1},
           'per_scenario': per, 'summary': summary,
           'inputs_sha256': {'robust.json': sha(os.path.join(HERE, 'robust.json')),
                             **{f'data/{n}.npz': sha(os.path.join(HERE, 'data', f'{n}.npz')) for n in csirs.SCEN}},
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, v in summary.items():
        print(f"  {k}: sigma {v['sigma_db']:.3f} dB, eta {v['eta_min']:.4f}-{v['eta_max']:.4f}, "
              f"slack {v['slack_min_us']:.0f}-{v['slack_max_us']:.0f} us")
    print(f'wrote {OUT} in {out["seconds"]} s')


if __name__ == '__main__':
    main()
