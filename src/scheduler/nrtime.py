#!/usr/bin/env python3
"""When the aperiodic CSI-RS procedure of Section VI-A delivers its reports, under both
computation-time conditions of TS 38.214 5.4.  Declared in protocol.json (analysis 'levels');
make_draft.py, figures_pdf.py and levels.py import it.

The report's delay after the last resource, Z'3, is common to the sweep and the reduced set, so on
its own it cancels.  TS 38.214 5.4 also requires a report to start no earlier than Z symbols after
the end of the PDCCH that triggered it (Table 5.4-2: Z3 for cri-RSRP).  The reduced
set's report, four resources after its trigger, is held by Z; the sweep's last report, 64 resources
after, mostly by Z'.  The delays therefore differ, and this module computes both procedures'
report times from the clauses read on 2026-10-18:

  target carrier  120 kHz, 125 us slots of 14 symbols; one-port, one-symbol CSI-RS resources, n per
                  slot in the slot's last n symbols (make_draft._nr_end), one aperiodic set per slot
                  (5.2.1.5.1: one triggering offset per set)
  control carrier triggers and reports, at 120 or 30 kHz; a two-symbol PDCCH opens each triggering
                  slot; reports go on that carrier's uplink as two-symbol transmissions that do not
                  cross a slot boundary
  first CSI-RS    with a 120 kHz control carrier the trigger's own slot (an offset-0 set starts after
                  the PDCCH); with 30 kHz the first 120 kHz slot that starts at least N_csirs = 5
                  PDCCH symbols after the PDCCH's end (5.2.1.5.1a)
  sweep           ceil(64/n) sets, at most four per trigger (maxNumberAperiodicCSI-PerBWP-
                  ForBeamReport, TS 38.331): trigger A in control slot 0 for sets 1-4, trigger B in
                  control slot 1 for the rest, each set measured by one cri-RSRP report setting;
                  A's reports on one uplink transmission, B's on another
  reduced set     trigger C in control slot 0, the sweep's first K resources at offset 0, one report
  report times    a transmission starts at the first symbol of the control carrier no earlier than
                  Z after the end of its trigger's PDCCH and Z' after the end of the last resource
                  any of its reports covers (5.4), with (Z, Z') = (Z3, Z'3) of Table 5.4-2 at
                  mu = min(mu_PDCCH, mu_CSI-RS, mu_UL): mu = 3, (min(97, X3 + KB2), X3); mu = 1,
                  (33, X1); X from beamReportTiming and KB from beamSwitchTiming (TS 38.331); two
                  aperiodic report transmissions never share a slot of the control carrier (5.2.1.5.1)
  completion      the sweep's d_S is its later transmission's start; the reduced set's candidate
                  infers after its one report, d_L = its start - one target slot + tau, the slot
                  boundary at which data start taken in the candidate's favour; the transmissions'
                  length and the base station's decoding, common to both, are taken as zero, which
                  can only make an exclusion harder

Not modelled.  The uplink data channel's preparation time (TS 38.214 6.4, re-read 2026-10-19: no
earlier than N2 symbols after the scheduling PDCCH, N2 = 36 at mu = 3 and 12 at mu = 1) holds only
the reduced set's report later than Z does, the sweep's being held later still by Z', so leaving it
out can only make an exclusion harder.  Time-division duplexing's uplink slots and timing advance
can delay either procedure's report; unshared_room gives the delay of the sweep's report that an
exclusion tolerates.  With zprime_only and X = 0 the module is csirs.py's model (gate: it
reproduces csirs.nr_threshold to 1e-12).
"""
import math

SLOT = 125.0                    # target carrier's slot, us
SYM = SLOT / 14
PDCCH_SYM, PUSCH_SYM = 2, 2
X_MU = {120: (14, 28, 56), 30: (4, 8, 14, 28)}     # beamReportTiming, symbols (TS 38.331)
KB_120 = (14, 28, 48, 224, 336)                    # beamSwitchTiming at 120 kHz, symbols
N_CSIRS = {30: 5}                                  # TS 38.214 Table 5.2.1.5.1a, mu_PDCCH = 1
N_BEAMS, SETS_PER_TRIGGER = 64, 4


def sym_us(scs):
    return 1000.0 / (scs / 15) / 14


def slot_us(scs):
    return 1000.0 / (scs / 15)


def res_end(j, n):
    """End of resource j (0-based) from the start of the first CSI-RS slot, n per slot."""
    return (j // n) * SLOT + (14 - n + j % n + 1) * SYM


def z_pair(ctrl, x, kb):
    """(Z, Z') in symbols of the control carrier, mu being the control carrier's (the smallest)."""
    if ctrl == 120:
        return min(97, x + kb), x
    if ctrl == 30:
        return 33, x
    raise ValueError(ctrl)


def first_csirs(ctrl):
    """Start of the first CSI-RS slot after a trigger at time 0, us."""
    if ctrl == 120:
        return 0.0
    t = (PDCCH_SYM + N_CSIRS[ctrl]) * sym_us(ctrl)
    return math.ceil(t / SLOT - 1e-9) * SLOT


def _next_symbol(t, scs):
    return math.ceil(t / sym_us(scs) - 1e-9) * sym_us(scs)


def _slot(t, scs):
    return math.floor(t / slot_us(scs) + 1e-9)


def _fit(t, scs):
    """A two-symbol transmission at t, moved to the next slot if it would cross a slot boundary."""
    k = _slot(t, scs)
    return t if t + PUSCH_SYM * sym_us(scs) <= (k + 1) * slot_us(scs) + 1e-9 else (k + 1) * slot_us(scs)


def _report(trig, last_end, ctrl, x, kb, zprime_only):
    z, zp = z_pair(ctrl, x, kb)
    t = last_end + zp * sym_us(ctrl)
    if not zprime_only:
        t = max(t, trig + (PDCCH_SYM + z) * sym_us(ctrl))
    return _fit(_next_symbol(t, ctrl), ctrl)


def sweep_reports(n, ctrl, x, kb, zprime_only=False):
    """Start of each of the sweep's uplink transmissions, us from trigger A's slot start."""
    assert ctrl == 30 or n <= 14 - PDCCH_SYM, "an offset-0 set starts after the PDCCH"
    n_sets = math.ceil(N_BEAMS / n)
    assert n_sets <= 2 * SETS_PER_TRIGGER, "two triggers of at most four report settings each"
    s0 = first_csirs(ctrl)
    out = []
    for g, tr in enumerate((0.0, slot_us(ctrl))):
        sets = list(range(g * SETS_PER_TRIGGER, min(n_sets, (g + 1) * SETS_PER_TRIGGER)))
        if not sets:
            continue
        # a set at its slot offset from the trigger, never before the trigger's own first CSI-RS slot
        assert s0 + sets[0] * SLOT >= tr + first_csirs(ctrl) - 1e-9, (n, ctrl)
        last = s0 + res_end(min(N_BEAMS, (sets[-1] + 1) * n) - 1, n)
        t = _report(tr, last, ctrl, x, kb, zprime_only)
        if out and _slot(t, ctrl) <= _slot(out[-1], ctrl):
            t = (_slot(out[-1], ctrl) + 1) * slot_us(ctrl)
        out.append(t)
    return out


def reduced_report(n, ctrl, x, kb, k=4, zprime_only=False):
    """Start of the reduced set's one transmission, us from trigger C's slot start."""
    return _report(0.0, first_csirs(ctrl) + res_end(k - 1, n), ctrl, x, kb, zprime_only)


def times(n, ctrl, x, kb, k=4, zprime_only=False):
    rep = sweep_reports(n, ctrl, x, kb, zprime_only)
    c = reduced_report(n, ctrl, x, kb, k, zprime_only)
    return {"sweep_reports": rep, "d_S": max(rep), "reduced": c, "saving": max(rep) - c}


def threshold(tau, t_max, n, ctrl, x, kb, k=4, zprime_only=False):
    """The lowest lower bound on eta_S at which Proposition 1(a) excludes a candidate of latency
    tau: (t_max - d_L) / (t_max - d_S)."""
    t = times(n, ctrl, x, kb, k, zprime_only)
    return (t_max - (t['reduced'] - SLOT + tau)) / (t_max - t['d_S'])


def capabilities(ctrl):
    """Every (beamReportTiming, beamSwitchTiming) a UE can declare; beamSwitchTiming enters Z at
    mu = 3 only."""
    return [(x, kb) for x in X_MU[ctrl] for kb in (KB_120 if ctrl == 120 else (None,))]


def worst(tau, t_max, n, ctrl, k=4):
    """The declarable timing most favourable to the candidate, the highest threshold:
    (threshold, x, kb)."""
    return max((threshold(tau, t_max, n, ctrl, x, kb, k), x, kb) for x, kb in capabilities(ctrl))


def k0_room(eta, tau, t_max, n, ctrl, x, kb):
    """A camera-only candidate (K = 0) sends no report and infers from the period's start: the
    longest transmission-and-decoding time of the sweep's last report at which Proposition 1(a)
    still excludes it at efficiency eta (negative: excluded at none)."""
    return (tau - SLOT - (1 - eta) * t_max) / eta - times(n, ctrl, x, kb)['d_S']


def unshared_room(eta, tau, t_max, n, ctrl, x, kb, k=4):
    """The longest further delay of the sweep's last report, not shared by the reduced set's, at
    which Proposition 1(a) still excludes a candidate of latency tau at efficiency eta."""
    t = times(n, ctrl, x, kb, k)
    return ((t['reduced'] - SLOT + tau) - (1 - eta) * t_max) / eta - t['d_S']
