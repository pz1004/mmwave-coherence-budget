#!/usr/bin/env python3
"""Rule R39 (i): the UE-side candidate of rule R32 under Release 19's report timing for predicted
beams.  Declared in protocol.json (analysis 'nr_r19') before it ran; make_draft.py imports it.
nrue.py, nrtdd.py and nrtime.py are imported unchanged.

TS 38.214 V19.5.0 5.4, read 2026-11-08: a CSI report whose reportQuantity-r19 is 'p-CRI-r19',
'p-CRI-RSRP-r19', 'p-SSB-Index-r19' or 'p-SSB-Index-RSRP-r19' (a beam predicted by a UE-side
model) has (Z, Z') = (Z3 + d, Z'3 + d'), with (Z3, Z'3) of Table 5.4-2 and d, d' according to UE
reported capability; TS 38.306 V19.4.0 4.2.7.2 (aiml-BM-Case1-r19) defines relaxationTimelineD-r19,
'the value of d for the relaxation of Z3 timeline', and relaxationTimelineD1-r19, the value of d'
for Z'3.  The declarable values (TS 38.331's field ranges) were not read.  So the UE's inference
time enters the procedure as a declared relaxation, in symbols of the control carrier: the
predicted-beam report starts no earlier than nrtdd.report's conditions with Z + d and Z' + d', and,
as in nrue.py, no earlier than the end of its inference.

Reading.  For each row of nrue.table() (paired, DDDSU and 7DS8U at 120 kHz, every phase and
declarable timing, 12 resources per slot, K = 4) and a relaxation d = d' (the inference adding to
both timelines alike; d alone and d' alone are given too), s_L19(d) is the candidate's report at
tau = 0.  The candidate cannot outperform the sweep at efficiency one once s_L19(d) >= s_S + W
(nrtdd.py's bounding inputs), nor the SS-guided refinement once s_L19(d) >= s_G + W; d_excl is the
least such d, in symbols and in us.  At an efficiency eta < 1, Proposition 1(a) needs
s_L19(d) - W >= s_S + (1 - eta)(T_max - s_S): relaxed() lets make_draft.py read that.

Gate V: with d = d' = 0 the relaxed report equals nrue.ue_report at tau = 0 and at tau = F (R32's
free window) for every row, and s_L19 never decreases in d.

`python3 nrue19.py --gate` runs gate V alone; `python3 nrue19.py` also prints the values.
"""
import math

import nrtdd as T
import nrtime as N
import nrue as U

D_MAX = 1000                         # symbols searched, 8.9 ms at 120 kHz


def relaxed(car, e, x, kb, d, dp, tau=0.0):
    """The predicted-beam report: nrtdd.report's arithmetic with (Z + d, Z' + dp), no earlier than
    the end of inference e + tau (the candidate's trigger at time 0, its last resource ending e)."""
    z, zp = N.z_pair(120, x, kb)
    t = max((N.PDCCH_SYM + z + d) * N.SYM, e + (zp + dp) * N.SYM)
    return T.ul_start(max(t, e + tau), 120, car)


def d_excl(row, target, mode='both'):
    """The least relaxation at which the report reaches target (us), or None within D_MAX."""
    car = T.Carrier(row['pattern'], row['phase'])
    for d in range(D_MAX + 1):
        a, b = {'both': (d, d), 'd': (d, 0), 'dp': (0, d)}[mode]
        if relaxed(car, row['e_L'], row['x'], row['kb'], a, b) >= target - 1e-9:
            return d
    return None


def gate_v():
    """Gate V alone: no relaxation's reading computed."""
    ok = True
    for r in U.table():
        car = T.Carrier(r['pattern'], r['phase'])
        s0 = relaxed(car, r['e_L'], r['x'], r['kb'], 0, 0)
        ok &= (abs(s0 - U.ue_report(car, r['e_L'], r['x'], r['kb'], 0.0)) < 1e-9
               and abs(relaxed(car, r['e_L'], r['x'], r['kb'], 0, 0, r['free'])
                       - U.ue_report(car, r['e_L'], r['x'], r['kb'], r['free'])) < 1e-9
               and abs(s0 - r['s_L']) < 1e-9)
        prev = s0
        for d in range(1, 200):
            s = relaxed(car, r['e_L'], r['x'], r['kb'], d, d)
            ok &= s >= prev - 1e-9
            prev = s
    return bool(ok)


def table():
    rows = []
    for r in U.table():
        car = T.Carrier(r['pattern'], r['phase'])
        s0 = relaxed(car, r['e_L'], r['x'], r['kb'], 0, 0)
        gate = (abs(s0 - U.ue_report(car, r['e_L'], r['x'], r['kb'], 0.0)) < 1e-9
                and abs(relaxed(car, r['e_L'], r['x'], r['kb'], 0, 0, r['free'])
                        - U.ue_report(car, r['e_L'], r['x'], r['kb'], r['free'])) < 1e-9
                and abs(s0 - r['s_L']) < 1e-9)
        prev = s0
        for d in range(1, 200):
            s = relaxed(car, r['e_L'], r['x'], r['kb'], d, d)
            gate &= s >= prev - 1e-9
            prev = s
        out = {k: r[k] for k in ('pattern', 'phase', 'x', 'kb', 's_S', 's_L', 's_G', 'W', 'e_L',
                                 'free', 'budget')}
        for mode in ('both', 'd', 'dp'):
            for ref, tgt in (('sweep', r['s_S'] + r['W']), ('guided', r['s_G'] + r['W'])):
                k = d_excl(r, tgt, mode)
                out[f'd_excl_{ref}_{mode}'] = k
                out[f'd_excl_{ref}_{mode}_us'] = None if k is None else k * N.SYM
        out['gate_V'] = bool(gate)
        rows.append(out)
    return rows


if __name__ == '__main__':
    import sys
    if '--gate' in sys.argv:
        assert gate_v(), "gate V"
        print("gate V holds")
        sys.exit(0)
    rows = table()
    assert all(r['gate_V'] for r in rows), "gate V"
    print("gate V holds")
    for p in U.PATTERNS:
        rs = [r for r in rows if r['pattern'] == p]
        for f in ('d_excl_sweep_both', 'd_excl_guided_both', 'd_excl_sweep_dp', 'd_excl_sweep_d'):
            v = [r[f] for r in rs]
            print(p, f, (min(v), max(v)) if None not in v else v)
        print(p, 'budget us', (round(min(r['budget'] for r in rs), 1),
                               round(max(r['budget'] for r in rs), 1)))
