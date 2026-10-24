#!/usr/bin/env python3
"""The reading rules R1-R7 applied to a set of tagged runs.

reading_rules.py is applied unchanged.  Its module-level J() and PREDQ are looked
up when its functions run, so this wrapper rebinds them: the controller's runs
(report, results, heldout) and the predictor-quality sweeps are read from their
tagged files, and everything else (rho.json, the timing files, aging.json,
protocol.json) from the untagged ones.  r7's per-q keys are mapped back to the
untagged names make_draft.py reads.

usage: python3 rules_suffix.py _rr        -> rules_rr.json
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SUF = sys.argv[1] if len(sys.argv) > 1 else ''
assert SUF.startswith('_') and len(SUF) > 1, "usage: rules_suffix.py _<tag>"
sys.argv = [sys.argv[0]]
import reading_rules as rr

CONTROLLER_RUNS = ('report.json', 'results.json', 'heldout.json')
_J, _r7, _PREDQ = rr.J, rr.r7, rr.PREDQ
rr.J = lambda n: _J(n[:-5] + SUF + '.json' if n in CONTROLLER_RUNS else n)


def r7(CTL, NOC, blind_key):
    rr.PREDQ = tuple((SUF[1:] + '_' + c, q) for c, q in _PREDQ)   # sweep_<tag>_<cond>.json
    try:
        out = _r7(CTL, NOC, blind_key)
    finally:
        rr.PREDQ = _PREDQ
    if not out.get('pending'):
        out['per_q'] = {k.split('_', 1)[1]: v for k, v in out['per_q'].items()}
    return out


rr.r7 = r7
sys.argv = ['reading_rules.py', '--out', f'rules{SUF}.json']
if __name__ == '__main__':
    rr.main()
