#!/usr/bin/env python3
"""Rules R32 and R33: a candidate that infers at the user equipment (UE), and the SS-block-guided
refinement of rule R31 against the sweep.  Declared in protocol.json (analysis 'ue_ssb') with
ssbsens.py; make_draft.py imports it.  It reads stored outputs alone and imports nrtdd.py unchanged.

R32.  nrtdd.py places the predictor at the base station: the candidate reports its K measurements and
the network infers, so data start after the report and the inference.  A UE-side candidate (TR 38.843
5.2, Alt. 2) infers from the end of its last resource, e_L, and reports the beam it selects.  Its
report starts no earlier than nrtdd.report's conditions and the end of inference, so inference within
the free window F = s_L - e_L delays nothing, and its data start t_L >= max(s_L, e_L + tau) + delta.
With nrtdd.py's t_S <= s_S + delta + W, an exclusion holds where e_L + tau >= s_S + W: the budget is
s_S + W - e_L = tau* + F.  Against the refinement, whose report is s_G, s_G replaces s_S.  A
candidate whose input is the SS-block measurements (K = 0), as the refinement's SS blocks are known
at the origin, infers on demand from the trigger: e_L = 0, and against the refinement its budget is
s_G + W.

R33.  At the measured efficiencies, the refinement's (ssbref.json) and the sweep's (the highest of
csirs.py's exact comparison and csirs_cfg.py's two configured variants), the period up to which the
refinement delivers more than the sweep whatever the delay delta >= 0 common to both.  Data start at
the first slot that opens with downlink symbols at or after a report plus delta, t_G(delta) and
t_S(delta), and with eta_S > eta_G the refinement delivers more at T > t_G iff
T < T*(delta) = (eta_S t_S - eta_G t_G) / (eta_S - eta_G).  T* is the least over delta, which moves
t_G or t_S only where a report plus delta crosses a slot start; a whole period of the pattern shifts
both, and T* with them, so one period of delta suffices.

Gate U: at tau = 0 and at tau = F the UE-side report is nrtdd's s_L, one symbol after F it is later,
and the candidate's resources are nrtdd.reduced's, for every pattern, phase and declarable timing.
"""
import json
import math
import os

import nrtdd as T
import nrtime as N

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERNS = ('paired',) + tuple(T.PATTERNS)
LAWS6 = [f"{e}|{n}" for e in ('coherent', 'per_re') for n in (198, 66, 33)]
VARIANTS = ('spread|worst', 'contiguous|worst')


def last_end(car, n, k=4):
    """The end of the candidate's last resource, us, and its slot, as nrtdd.reduced places them."""
    pos = [T.NSYM - n + j for j in range(k)]
    first = T.triggers(120, car)[0][1]
    slot = next(s for s in range(first, first + T.HORIZON)
                if pos[0] >= N.PDCCH_SYM and all(car.dl(s, j) for j in pos))
    return slot * T.SLOT + (pos[-1] + 1) * T.SYM, slot


def ue_report(car, e, x, kb, tau):
    """The UE-side candidate's report: nrtdd.report's conditions, no earlier than e + tau."""
    return T.report(0.0, e, 120, car, x, kb, e + tau)


def times(name, phase, x, kb, n=12, k=4):
    """R32's quantities at one pattern, phase and timing (120 kHz control carrier), us."""
    car = T.Carrier(name, phase)
    t = T.times(name, phase, n, 120, x, kb, k)
    e, slot = last_end(car, n, k)
    s_G = t['guided4']
    assert abs(t['guided8'] - s_G) < 1e-9, "one report time for both refinements"
    F = t['s_L'] - e
    gate = (slot == t['reduced_slot'] and abs(ue_report(car, e, x, kb, 0.0) - t['s_L']) < 1e-9
            and abs(ue_report(car, e, x, kb, F) - t['s_L']) < 1e-9
            and ue_report(car, e, x, kb, F + T.SYM) > t['s_L'] + 1e-9 and F >= 0)
    return {"pattern": name, "phase": phase, "x": x, "kb": kb, "s_S": t['s_S'], "s_L": t['s_L'],
            "s_G": s_G, "W": t['W'], "e_L": e, "free": F, "budget": t['budget'],
            "ue_sweep": t['s_S'] + t['W'] - e, "ue_guided": s_G + t['W'] - e,
            "k0_guided": s_G + t['W'], "k0_sweep": t['s_S'] + t['W'], "gate_U": bool(gate)}


def table(n=12, k=4):
    return [times(p, ph, x, kb, n, k) for p in PATTERNS
            for ph in ([0] if p == 'paired' else T.phases(p, 120))
            for x, kb in N.capabilities(120)]


def data_start(car, s):
    """The first slot start at or after s that opens with downlink symbols, us."""
    j = math.ceil(s / T.SLOT - 1e-9)
    while not car.opens_dl(j):
        j += 1
    return j * T.SLOT


def crossover(car, s_G, s_S, e_G, e_S):
    """R33's T*, us: the least over delta >= 0 of the period up to which the refinement delivers
    more; inf if it never delivers less."""
    P = car.P * T.SLOT
    deltas = {0.0}
    for s in (s_G, s_S):
        for j in range(math.ceil((s + P) / T.SLOT) + 1):
            if car.opens_dl(j) and 0 <= j * T.SLOT - s < P:
                deltas.add(j * T.SLOT - s)
    out = math.inf
    for d in deltas:
        t_G, t_S = data_start(car, s_G + d), data_start(car, s_S + d)
        assert t_G <= t_S
        out = min(out, math.inf if e_S <= e_G else (e_S * t_S - e_G * t_G) / (e_S - e_G))
    return out


def efficiencies():
    """Per (layer, law, scenario): the refinement's and the sweep's measured efficiencies."""
    SB = json.load(open(os.path.join(HERE, 'ssbref.json')))
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))['configs']
    CG = json.load(open(os.path.join(HERE, 'csirs_cfg.json')))['variants']
    src = [CS] + [CG[v]['configs'] for v in VARIANTS]
    return {(l, k, sc): (SB['references'][f'guided{l}'][k]['per_scenario'][sc]['eta'],
                         max(c[k]['per_scenario'][sc]['eta'] for c in src))
            for l in SB['layers'] for k in LAWS6 for sc in SB['scenarios']}


def crossovers(rows=None):
    """R33: per pattern, phase and timing, the least T* over layers, laws and scenarios, ms, and
    where it falls."""
    eff = efficiencies()
    out = []
    for r in rows or table():
        car = T.Carrier(r['pattern'], r['phase'])
        v, at = min((crossover(car, r['s_G'], r['s_S'], *e), key) for key, e in eff.items())
        out.append({"pattern": r['pattern'], "phase": r['phase'], "x": r['x'], "kb": r['kb'],
                    "t_star_ms": v / 1000, "at": list(at)})
    return out


if __name__ == '__main__':
    rows = table()
    assert all(r['gate_U'] for r in rows), "gate U"
    print("gate U holds")
    for p in PATTERNS:
        rs = [r for r in rows if r['pattern'] == p]
        print(p, {f: (round(min(r[f] for r in rs), 1), round(max(r[f] for r in rs), 1))
                  for f in ('budget', 'free', 'ue_sweep', 'ue_guided', 'k0_guided', 'k0_sweep')})
    for c in crossovers(rows):
        print(c['pattern'], c['phase'], c['x'], c['kb'], round(c['t_star_ms'], 2), c['at'])
