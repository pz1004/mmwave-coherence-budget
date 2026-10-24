#!/usr/bin/env python3
"""Sensitivity sweeps for the controller (declared with the ten-seed analysis in protocol.json).

One variable at a time, around the measured baseline:
  rho  in {0.1, 0.3, 3}      every INFERENCE charge scaled (the controller's tau,
                             the one-shot early-exit methods, the FC and CNN
                             networks); the probe, lookup and test charges not
  t_p  in {50, 125, 250} us  the probe's air time (pilot + feedback), for every
                             method; the per-probe bookkeeping is unchanged

Each condition gets its own solve of the headline controller and its two
twins (the variant keys dp, blind, noc, each off- and on-policy), its own
validation run (so the deployable baseline is picked under that condition's
charges), and one evaluation of all six evaluation sites. The summary reads
the seed-by-block contrasts at the headline controller's estimation suffix and
writes sweeps.json. The baseline is the primary run itself (results.json and
heldout.json).

Descriptive only: no reading rule consults these conditions.

A third axis is the predictor's quality: q in
{0.125, 0.25, 0.5, 1}, an oracle-mixed anytime predictor (sim.oracle_mix) at
unchanged charges, solved, validated and evaluated exactly as above.  Those
conditions are built only with --predq and summarised into predq.json, so
sweeps.json -- whose condition set the paper asserts -- is never rewritten by
them.  Reading rule R7 reads predq.json.

usage: python3 sweeps.py [--only COND ...] [--summarize-only]
       python3 sweeps.py --predq [--only COND ...] [--summarize-only]
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CONDITIONS = {'rho0.1': ['--tau-scale', '0.1'], 'rho0.3': ['--tau-scale', '0.3'],
              'rho3': ['--tau-scale', '3'],
              'tp50': ['--tp-air-us', '50'], 'tp125': ['--tp-air-us', '125'],
              'tp250': ['--tp-air-us', '250']}
PREDQ = {'oq0.125': ['--oracle-mix', '0.125'], 'oq0.25': ['--oracle-mix', '0.25'],
         'oq0.5': ['--oracle-mix', '0.5'], 'oq1': ['--oracle-mix', '1']}
# A driver (rerank.py) sets TAG = '_rr' and EXTRA = ['--rank-charge'] and calls
# build() and summarize() itself, so every file it reads or writes carries the tag
# (tables_<seed>_rr_<cond>.npz, sweep_rr_<cond>.json, sweeps_rr.json, predq_rr.json)
# and no file of the recorded sweeps is touched.  Run as a script, both stay empty.
TAG, EXTRA = '', []
VAL = 'city_1_losangeles_28'
EVAL = ['city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28',
        'city_5_philadelphia_28', 'city_6_miami_28', 'city_7_sandiego_28']


def run(args):
    print('  $', ' '.join(args), flush=True)
    subprocess.run([PY, *args], cwd=HERE, check=True)


def build(cond):
    flags, suf = {**CONDITIONS, **PREDQ}[cond], f'{TAG}_{cond}'
    run(['sched.py', '--variants', 'dp,blind,noc', *flags, *EXTRA, '--out-suffix', suf])
    run(['evaluate.py', '--tables-suffix', suf, *flags, *EXTRA, '--sites', VAL,
         '--out', f'validation{suf}.json'])
    run(['evaluate.py', '--tables-suffix', suf, *flags, *EXTRA, '--validation',
         f'validation{suf}.json', '--sites', *EVAL, '--out', f'sweep{suf}.json'])


def contrasts(sites, ctl, blind, noc):
    """Per site: relative grid-averaged gain and both intervals, for the three
    contrasts the reading rules use."""
    out = {}
    for s in sites:
        G = s['grid']
        mean = lambda nm: sum(s['methods'][nm]['per_T'][str(T)]['mean'] for T in G) / len(G)
        row = {}
        for tag, fam, key, ref in (('aware_vs_blind', 'mechanism', f'{ctl}|{blind}', blind),
                                   ('compute_pays', 'compute_pays', f'{ctl}|{noc}', noc),
                                   ('vs_deployable', 'deployable', ctl, None)):
            h = s.get(f'{fam}_aggregate_hier', {}).get(key)
            if h is None:
                continue
            base = mean(ref) if ref else mean(ctl) - h['diff']
            row[tag] = {"rel": h['diff'] / base, "diff": h['diff'],
                        "ci95": h['ci95'], "ci95_t": h['ci95_t'], "p": h['p'], "p_t": h['p_t']}
        out[s['site']] = row
    return out


def summarize(conds=None, path='sweeps.json', definitions=None):
    rep = json.load(open(os.path.join(HERE, f'report{TAG}.json')))
    ctl = rep['headline_controller']
    suf = '-pi' if ctl.endswith('-pi') else ''
    blind, noc = f'sched-costblind{suf}', rep['nocompute_controller']
    base = [s for f in (f'results{TAG}.json', f'heldout{TAG}.json')
            for s in json.load(open(os.path.join(HERE, f)))['sites']]
    out = {"headline_controller": ctl, "conditions": {"base": contrasts(base, ctl, blind, noc)},
           "definitions": definitions or {"rho": "scales every inference charge",
                                          "t_p": "probe air time, us"}}
    for cond in (CONDITIONS if conds is None else conds):
        p = os.path.join(HERE, f'sweep{TAG}_{cond}.json')
        if os.path.exists(p):
            out['conditions'][cond] = contrasts(json.load(open(p))['sites'], ctl, blind, noc)
    json.dump(out, open(os.path.join(HERE, path), 'w'), indent=1)
    for cond, rows in out['conditions'].items():
        g = [r['aware_vs_blind']['rel'] * 100 for r in rows.values() if 'aware_vs_blind' in r]
        p = [r['compute_pays']['rel'] * 100 for r in rows.values() if 'compute_pays' in r]
        if g and p:
            print(f"{cond:7s} aware vs blind {min(g):+6.2f}..{max(g):+6.2f}%   "
                  f"compute pays {min(p):+6.2f}..{max(p):+6.2f}%")


if __name__ == '__main__':
    pq = '--predq' in sys.argv
    argv = [a for a in sys.argv if a != '--predq']
    pool = PREDQ if pq else CONDITIONS
    if '--summarize-only' not in argv:
        only = argv[argv.index('--only') + 1:] if '--only' in argv else list(pool)
        assert set(only) <= set(pool), f"unknown conditions {set(only) - set(pool)}"
        for cond in only:
            print(f"== {cond}", flush=True)
            build(cond)
    if pq:
        summarize(PREDQ, 'predq.json',
                  {"q": "oracle-mixed anytime predictor: the rate-optimal beam ranks "
                        "first for a fraction q of receivers; charges unchanged"})
    else:
        summarize()
