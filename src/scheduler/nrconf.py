#!/usr/bin/env python3
"""Rule R39 (ii): the CSI-RS budgets when the network-side candidate confirms its prediction with a
second measurement.  Declared in protocol.json (analysis 'nr_r19') before it ran; make_draft.py
imports it.  nrtddmu.py, nrtime.py and nrmu.py are imported unchanged.

TR 38.843 V18.0.0 6.3.1, read 2026-11-04 and 2026-11-08, counts in BM-Case1's RS overhead 'the
beams (pairs) required for additional measurements before/after the prediction if applicable'.  The
candidate of Section VI-A reports its K = 4 measurements and the network infers; here the network
then measures the K_c predicted beams with one more aperiodic CSI-RS set before data start:
  trigger    a PDCCH in the first two symbols of the first slot that opens with downlink symbols
             and starts no earlier than the end of inference, e = s_L + tau (the report's
             transmission and decoding, common to both procedures, taken as zero, in the
             candidate's favor; no preparation time for the second trigger);
  resources  its K_c resources at the sweep's first positions in the earliest slot whose symbols
             there are downlink, as nrtddmu.reduced places the candidate's K;
  report     nrtddmu.report's conditions (Z after that PDCCH, Z' after the last resource);
  data       after that report: t_L >= s_C + delta, so the bounding input is d~_L = s_C - W.
Carriers as nrtddmu.py: target and control at one spacing and slot format, 120 kHz (paired, DDDSU,
7DS8U) and 60 kHz (paired, DDDSU, DDDSUUUU), every phase and declarable timing, 12 and 8 resources
per slot; K_c = 1 and 4.

Budget.  An exclusion at efficiency one holds where s_C >= s_S + W (against the refinement,
s_G + W).  s_C depends on tau only through the trigger slot, so the budget is the latest trigger
slot start whose s_C is earlier, less s_L; None if even the first trigger slot at or after s_L is
too late (no latency is within budget).  The added delay at tau = 0, s_C - s_L, is reported too.

Gates.  K1: a trigger at the first slot gives nrtddmu.times(..., k = K_c)'s s_L.  K2: on paired
spectrum s_C(j) = j slot + nrtddmu.times('paired', ..., k = K_c)['s_L'] for every trigger slot j
used.  K3: s_C never decreases in the trigger slot, and a whole period shifts it by one period.

`python3 nrconf.py --gate` runs the gates alone; `python3 nrconf.py` also prints the values.
"""
import math
import sys

import nrmu as M
import nrtddmu as TM
import nrtime as N

KC = (1, 4)
SCS = (120, 60)


def confirm(name, scs, phase, n, x, kb, e, kc):
    """(s_C, trigger slot): the confirmation's report for inference ending at e (us)."""
    car = TM.Carrier(name, scs, phase)
    j = math.ceil(e / car.slot - 1e-9)
    while not car.opens_dl(j):
        j += 1
    rep, _ = TM.reduced(n, TM.Carrier(name, scs, phase + j), x, kb, kc)
    return j * car.slot + rep, j


def budget(row, kc, target):
    """The latest inference end keeping s_C < target, less s_L; None if none; and s_C at tau=0."""
    car = TM.Carrier(row['pattern'], row['scs'], row['phase'])
    s0, j0 = confirm(row['pattern'], row['scs'], row['phase'], row['n'], row['x'], row['kb'],
                     row['s_L'], kc)
    best, j = None, j0
    while j * car.slot < target:
        if car.opens_dl(j):
            sc, _ = confirm(row['pattern'], row['scs'], row['phase'], row['n'], row['x'],
                            row['kb'], j * car.slot, kc)
            if sc < target - 1e-9:
                best = j * car.slot - row['s_L']
            else:
                break
        j += 1
    return best, s0


def rows():
    out = []
    for scs in SCS:
        for name in ('paired',) + tuple(TM.PATTERNS[scs]):
            for n in (12, 8):
                for x, kb in M.capabilities(scs):
                    for ph in ([0] if name == 'paired' else TM.phases(name, scs)):
                        t = TM.times(name, scs, ph, n, x, kb)
                        r = {k: t[k] for k in ('pattern', 'scs', 'phase', 'n', 'x', 'kb', 's_S',
                                               's_L', 'W', 'budget', 'guided4', 'guided8')}
                        for kc in KC:
                            for ref, tgt in (('sweep', t['s_S'] + t['W']),
                                             ('guided4', t['guided4'] + t['W']),
                                             ('guided8', t['guided8'] + t['W'])):
                                b, s0 = budget(r, kc, tgt)
                                r[f'conf{kc}_{ref}'] = b
                            r[f'conf{kc}_added'] = s0 - t['s_L']
                        out.append(r)
    return out


def gates():
    ok = {'K1': True, 'K2': True, 'K3': True}
    for scs in SCS:
        for name in ('paired',) + tuple(TM.PATTERNS[scs]):
            car0 = TM.Carrier(name, scs)
            for n in (12, 8):
                for x, kb in M.capabilities(scs):
                    for ph in ([0] if name == 'paired' else TM.phases(name, scs)):
                        car = TM.Carrier(name, scs, ph)
                        for kc in KC:
                            s, j = confirm(name, scs, ph, n, x, kb, 0.0, kc)
                            base = TM.times(name, scs, ph, n, x, kb, kc)['s_L']
                            ok['K1'] &= j == 0 and abs(s - base) < 1e-9
                            prev = -math.inf
                            for jj in range(0, 3 * car0.P + 8):
                                if not car.opens_dl(jj):
                                    continue
                                sc, _ = confirm(name, scs, ph, n, x, kb, jj * car.slot, kc)
                                ok['K3'] &= sc >= prev - 1e-9
                                prev = sc
                                sp, _ = confirm(name, scs, ph, n, x, kb,
                                                (jj + car0.P) * car.slot, kc)
                                ok['K3'] &= abs(sp - sc - car0.P * car.slot) < 1e-9
                                if name == 'paired':
                                    ok['K2'] &= abs(sc - jj * car.slot - base) < 1e-9
    return {k: bool(v) for k, v in ok.items()}


if __name__ == '__main__':
    g = gates()
    assert all(g.values()), g
    print("gates K1-K3 hold")
    if '--gate' in sys.argv:
        sys.exit(0)
    R = rows()
    for scs in SCS:
        for name in ('paired',) + tuple(TM.PATTERNS[scs]):
            rs = [r for r in R if r['scs'] == scs and r['pattern'] == name and r['n'] == 12]
            for kc in KC:
                for ref in ('sweep', 'guided4', 'guided8'):
                    v = [r[f'conf{kc}_{ref}'] for r in rs]
                    vv = [a for a in v if a is not None]
                    print(scs, name, kc, ref, 'budget us', (round(min(vv), 1), round(max(vv), 1))
                          if vv else None, 'none at', sum(a is None for a in v), 'of', len(v))
                a = [r[f'conf{kc}_added'] for r in rs]
                print(scs, name, kc, 'added us', round(min(a), 1), round(max(a), 1))
            print(scs, name, 'budget us', round(min(r['budget'] for r in rs), 1),
                  round(max(r['budget'] for r in rs), 1))
