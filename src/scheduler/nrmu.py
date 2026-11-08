#!/usr/bin/env python3
"""Rule R35: nrtime.py's paired CSI-RS procedure at a 60 kHz subcarrier spacing, and the per-probe
sweep with analog receive beams.  Declared in protocol.json (analysis 'mu_rx') before it ran;
make_draft.py imports it.  nrtime.py is imported unchanged.

Numerology.  nrtime.py times the procedure on a 120 kHz target carrier.  Here the target and the
control carrier share one subcarrier spacing, 60 or 120 kHz, and every other rule of nrtime.py
stands: one-port, one-symbol resources, n per slot in the slot's last n symbols (a short last set
ending with its slot), one aperiodic set per slot, at most four sets per trigger (trigger A in the
first control slot, B in the next), each set measured by one cri-RSRP report setting, A's reports on
one uplink transmission and B's on another, never in one slot.  A report starts at the first symbol
no earlier than Z after its trigger's PDCCH and Z' after the last resource it covers (TS 38.214
V18.11.0 5.4), (Z, Z') = (Z3, Z'3) of Table 5.4-2 at mu = min(mu_PDCCH, mu_CSI-RS, mu_UL):
    mu = 2 (60 kHz):  (min(44, X2 + KB1), X2)
    mu = 3 (120 kHz): (min(97, X3 + KB2), X3)
X from beamReportTiming and KB from beamSwitchTiming (TS 38.331 V18.11.0 6.3.3): at 60 kHz X2 in
{8, 14, 28} and KB1 in {14, 28, 48, 224, 336} symbols.  The candidate measures the sweep's first K
resources under its own trigger in the first control slot and sends one report.  Data start at the
target's first slot boundary after a report, so nrtime.py's bounding inputs d~_S = s_S and
d~_L = s_L - h + tau, h the slot, give the budget tau* = s_S - s_L + h.  The per-slot capability
maxNumberSSB-CSI-RS-ResourceOneTx (TS 38.306 V18.11.0 4.2.7.2) does not depend on the subcarrier
spacing, so 12 and 8 resources per slot apply at both.  The efficiencies are not simulated here: at
60 kHz a resource of density 3 spans 132 resource blocks of a 100 MHz carrier, twice the 120 kHz
count, and the paper reads the budgets at the 120 kHz measurement's efficiencies and bounds.

Gate M: at 120 kHz the schedules reproduce nrtime.times for 12 and 8 resources per slot and every
declarable timing to 1e-9.

Receive beams.  With N_r analog receive beams and an exhaustive transmit-receive search, the
per-probe sweep takes M N_r probes: d_S = M N_r t_p and tau* = (M N_r - K) t_p, with t_p one
P-core's (rho.json).  Proposition 1(a) needs T_max > d_S, so where M N_r t_p >= T_max nothing is
excluded but by feasibility.  Under the CSI-RS procedure the user equipment applies the QCL
assumptions of the indicated TCI states to the aperiodic CSI-RS resources (TS 38.214 5.2.1.5.1),
the spatial QCL source of the paper's Section VI-A, so it receives every resource of the sweep and of
the candidate on that source's beam: both keep their resources and no budget is computed for it.

`python3 nrmu.py --gate` runs gate M alone; `python3 nrmu.py` also prints R35's values.
"""
import json
import math
import os
import sys

import nrtime as N

HERE = os.path.dirname(os.path.abspath(__file__))
Z_CAP = {60: 44, 120: 97}                            # TS 38.214 Table 5.4-2: Z3 = min(Z_CAP, X + KB)
X_MU = {60: (8, 14, 28), 120: N.X_MU[120]}           # beamReportTiming, symbols (TS 38.331 6.3.3)
KB = {60: (14, 28, 48, 224, 336), 120: N.KB_120}     # beamSwitchTiming, symbols (TS 38.331 6.3.3)
RX = (2, 4, 8)                                       # receive beams


def res_end(j, n, scs, total=N.N_BEAMS):
    """nrtime.res_end on a carrier of subcarrier spacing scs."""
    m = min(n, total - (j // n) * n)
    return (j // n) * N.slot_us(scs) + (14 - m + j % n + 1) * N.sym_us(scs)


def z_pair(scs, x, kb):
    """(Z, Z') in symbols, mu being the one subcarrier spacing of both carriers."""
    return min(Z_CAP[scs], x + kb), x


def _report(trig, last_end, scs, x, kb):
    z, zp = z_pair(scs, x, kb)
    t = max(last_end + zp * N.sym_us(scs), trig + (N.PDCCH_SYM + z) * N.sym_us(scs))
    return N._fit(N._next_symbol(t, scs), scs)


def sweep_reports(n, scs, x, kb):
    """Start of each of the sweep's uplink transmissions, us from trigger A's slot start."""
    assert n <= 14 - N.PDCCH_SYM, "an offset-0 set starts after the PDCCH"
    n_sets = math.ceil(N.N_BEAMS / n)
    assert n_sets <= 2 * N.SETS_PER_TRIGGER, "two triggers of at most four report settings each"
    out = []
    for g, tr in enumerate((0.0, N.slot_us(scs))):
        sets = list(range(g * N.SETS_PER_TRIGGER, min(n_sets, (g + 1) * N.SETS_PER_TRIGGER)))
        if not sets:
            continue
        assert sets[0] * N.slot_us(scs) >= tr - 1e-9, (n, scs)
        t = _report(tr, res_end(min(N.N_BEAMS, (sets[-1] + 1) * n) - 1, n, scs), scs, x, kb)
        if out and N._slot(t, scs) <= N._slot(out[-1], scs):
            t = (N._slot(out[-1], scs) + 1) * N.slot_us(scs)
        out.append(t)
    return out


def reduced_report(n, scs, x, kb, k=4):
    """Start of the candidate's one transmission, us from its trigger's slot start."""
    return _report(0.0, res_end(k - 1, n, scs), scs, x, kb)


def times(n, scs, x, kb, k=4):
    s_S, s_L = max(sweep_reports(n, scs, x, kb)), reduced_report(n, scs, x, kb, k)
    return {"n": n, "scs": scs, "x": x, "kb": kb, "s_S": s_S, "s_L": s_L, "h": N.slot_us(scs),
            "budget": s_S - s_L + N.slot_us(scs)}


def capabilities(scs):
    """Every (beamReportTiming, beamSwitchTiming) a UE can declare at this subcarrier spacing."""
    return [(x, kb) for x in X_MU[scs] for kb in KB[scs]]


def budgets(scs, n, k=4):
    """Every declarable timing's times and budget."""
    return [times(n, scs, x, kb, k) for x, kb in capabilities(scs)]


def gate_m():
    """Gate M: at 120 kHz the schedules reproduce nrtime.times."""
    ok = True
    for n in (12, 8):
        for x, kb in N.capabilities(120):
            t, r = times(n, 120, x, kb), N.times(n, 120, x, kb)
            ok &= abs(t['s_S'] - r['d_S']) < 1e-9 and abs(t['s_L'] - r['reduced']) < 1e-9
            ok &= abs(t['budget'] - (r['saving'] + N.SLOT)) < 1e-9
    return bool(ok)


def rx_probe(n_r, t_p, m=N.N_BEAMS, k=4, t_max=10000.0):
    """The per-probe sweep over every transmit-receive pair: d_S, tau* and whether it finishes
    within T_max, us."""
    d = m * n_r * t_p
    return {"n_r": n_r, "d_S": d, "budget": (m * n_r - k) * t_p, "within": d < t_max}


def t_p_pcore():
    """One P-core's time per probe, us (rho.json, as the paper's Table IV reads it)."""
    return json.load(open(os.path.join(HERE, 'rho.json')))['t_p_us']


if __name__ == '__main__':
    assert gate_m(), "gate M: the 120 kHz schedules do not reproduce nrtime.py"
    print("gate M holds")
    if '--gate' in sys.argv[1:]:
        sys.exit(0)
    for n in (12, 8):
        rows = budgets(60, n)
        lo, hi = min(rows, key=lambda r: r['budget']), max(rows, key=lambda r: r['budget'])
        print(f"60 kHz, {n} per slot: budget {lo['budget']:.1f}-{hi['budget']:.1f} us; largest at "
              f"X2 = {hi['x']}, KB1 = {hi['kb']}: s_S {hi['s_S']:.1f}, s_L {hi['s_L']:.1f}")
    tp = t_p_pcore()
    for n_r in RX:
        r = rx_probe(n_r, tp)
        print(f"{n_r} receive beams, t_p {tp:.3f} us: d_S {r['d_S']:.1f}, budget {r['budget']:.1f}, "
              f"within T_max {r['within']}")
