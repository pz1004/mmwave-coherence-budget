#!/usr/bin/env python3
"""Rule R30: the aperiodic CSI-RS procedure of Section VI-A under time-division duplexing (TDD), and
the report times of rule R31's SS-block-guided reference.  Declared in protocol.json (analysis
'tdd_ssb') before it ran; make_draft.py imports it.

nrtime.py times the procedure with every symbol of both carriers open to the downlink and the uplink,
as on paired spectrum.  The 120 kHz target carrier is in FR2, whose bands are all TDD (TS 38.101-2
V18.14.0 Table 5.2-1), and so is a 120 kHz control carrier, which shares the target's slot format
(intra-band carrier aggregation assumes one, ibid. 4.2).  A pattern of TS 38.331 V18.11.0 6.3.2
(TDD-UL-DL-Pattern: nrofDownlinkSlots full downlink slots, then nrofDownlinkSymbols downlink symbols
opening the next slot; nrofUplinkSlots full uplink slots closing the period, nrofUplinkSymbols uplink
symbols closing the slot before them; the other symbols flexible, here guard) restricts
  triggers   a PDCCH in the first two symbols of a slot that opens with downlink symbols;
  CSI-RS     m one-symbol resources of one set in one slot's downlink symbols after those two, at
             nrtime.py's positions (a set's resources in the slot's last m symbols, the candidate's
             four at the sweep's first positions), in a partly downlink slot the last m downlink ones;
  reports    a two-symbol transmission within one slot's uplink symbols;
  data       from the start of a slot that opens with downlink symbols.
Two reference patterns of TS 38.101-2 V18.14.0 Annex A at 120 kHz, read 2026-11-06: DDDSU, period
0.625 ms, S = 10D+2G+2U (Table A.3.3.1-1, the downlink reference channels), and 7DS8U, period 2 ms,
S = 12D+2G (Table A.2.3-1, the uplink ones).  A 30 kHz control carrier, FR1, keeps nrtime.py's paired
model; the target's pattern then restricts the CSI-RS and the data alone.

Schedules, for every phase of the pattern at which the first trigger can fall and every declarable UE
timing, each procedure as early as the rules allow:
  sweep      ceil(64/n) sets, n resources each but the last (64 - (ceil(64/n) - 1)n), one set per slot
             (TS 38.214 V18.11.0 5.2.1.5.1), at triggering offsets of at most 31 slots
             (aperiodicTriggeringOffset-r16, TS 38.331 6.3.2) and at most four sets per trigger
             (maxNumberAperiodicCSI-PerBWP-ForBeamReport): trigger A in the first triggering slot,
             trigger B in the next (a 30 kHz control carrier: its next slot, as nrtime.py); the full
             sets in the earliest slots that hold them, the short set in any slot that holds it, any
             split of the sets between the triggers; the earliest last report over these schedules
  reduced    the candidate's K resources under trigger C in the first triggering slot, at the sweep's
             first positions in the earliest slot whose symbols there are downlink; one report
  guided     rule R31's reference: one set of m resources under trigger C in the earliest slot that
             holds it, at its last m downlink symbols; one report
  input      a candidate whose input precedes the trigger (K = 0, SS-block reports): one report, held
             by Z alone
Report times are nrtime.py's (Z after the trigger's PDCCH, Z' after the last resource reported, TS
38.214 5.4; two report transmissions never share a slot), at the first uplink symbols that hold a
two-symbol transmission.

Data.  As in nrtime.py, a report's transmission and decoding, a delay delta >= 0 common to both
procedures, precede the data, which start at the first slot opening with downlink symbols after it,
so t_S <= s_S + delta + W, with W the longest gap between consecutive such slot starts (nrtime.py's
one slot, h, on paired spectrum), and t_L >= s_L + delta + tau.  The bounding inputs d~_S = s_S and
d~_L = s_L - W + tau carry an exclusion to the physical data starts, and the budget is
tau* = s_S - s_L + W.

Gate F: with every symbol of both carriers open to both directions ('paired'), the schedules
reproduce nrtime.times for 12 and 8 resources per slot, both control carriers and every declarable
timing, and W = h.  Every schedule is checked as built (check_schedule).
"""
import itertools
import math

import nrtime as N

SLOT, SYM = N.SLOT, N.SYM
NSYM = 14
MAX_OFFSET = 31                 # aperiodicTriggeringOffset-r16, slots
HORIZON = 160                   # target slots searched: 20 ms
PATTERNS = {
    # name: (period in 120 kHz slots, nrofDownlinkSlots, nrofDownlinkSymbols, nrofUplinkSlots,
    #        nrofUplinkSymbols), TS 38.101-2 V18.14.0 at 120 kHz
    'DDDSU': (5, 3, 10, 1, 2),       # Table A.3.3.1-1: 0.625 ms, S = 10D+2G+2U
    '7DS8U': (16, 7, 12, 8, 0),      # Table A.2.3-1: 2 ms, S = 12D+2G
}


def symbols(name):
    """One period of the pattern: per slot, per symbol, 'D', 'U' or 'G'; 'paired' is one slot of 'B',
    open to both directions."""
    if name == 'paired':
        return [['B'] * NSYM]
    P, dls, dly, uls, uly = PATTERNS[name]
    out = [['G'] * NSYM for _ in range(P)]
    for k in range(dls):
        out[k] = ['D'] * NSYM
    for j in range(dly):
        out[dls][j] = 'D'
    for k in range(P - uls, P):
        out[k] = ['U'] * NSYM
    for j in range(NSYM - uly, NSYM):
        out[P - uls - 1][j] = 'U'
    assert dls + uls + (dly > 0 or uly > 0) <= P
    return out


class Carrier:
    """A 120 kHz carrier with a pattern, its slot 0 at the pattern's slot `phase`."""

    def __init__(self, name, phase=0):
        self.name, self.pat = name, symbols(name)
        self.P, self.phase = len(self.pat), phase % len(self.pat)

    def kind(self, slot, sym):
        return self.pat[(self.phase + slot) % self.P][sym]

    def dl(self, slot, sym):
        return self.kind(slot, sym) in 'DB'

    def ul(self, slot, sym):
        return self.kind(slot, sym) in 'UB'

    def opens_dl(self, slot):
        """A PDCCH fits the slot's first two symbols, and data can start at its start."""
        return all(self.dl(slot, j) for j in range(N.PDCCH_SYM))

    def dl_run(self, slot):
        """The number of downlink symbols from the slot's start."""
        j = 0
        while j < NSYM and self.dl(slot, j):
            j += 1
        return j

    def set_syms(self, slot, m):
        """A set of m resources in the slot: its last m downlink symbols after the PDCCH's two, or
        None.  In a full downlink slot these are nrtime.py's, the slot's last m symbols."""
        r = self.dl_run(slot)
        return list(range(r - m, r)) if r - m >= N.PDCCH_SYM and m > 0 else None

    def wait(self):
        """W: the longest gap between consecutive slot starts that open with downlink symbols, us."""
        starts = [k for k in range(self.P) if self.opens_dl(k)]
        assert starts, self.name
        gaps = [(b - a) % self.P or self.P for a, b in zip(starts, starts[1:] + starts[:1])]
        return max(gaps) * SLOT


def ul_start(t, ctrl, car):
    """The earliest start, at or after t (us), of a two-symbol transmission within one slot's uplink
    symbols of the control carrier: the 120 kHz carrier `car`, or a paired 30 kHz carrier."""
    if ctrl == 30:
        return N._fit(N._next_symbol(t, 30), 30)
    j = math.ceil(t / SYM - 1e-9)
    for j in range(j, j + NSYM * HORIZON):
        s, q = divmod(j, NSYM)
        if q + N.PUSCH_SYM <= NSYM and all(car.ul(s, q + i) for i in range(N.PUSCH_SYM)):
            return j * SYM
    raise AssertionError("no uplink within the horizon")


def ctrl_slot(t, ctrl):
    return math.floor(t / N.slot_us(ctrl) + 1e-9)


def report(trig, last_end, ctrl, car, x, kb, not_before=0.0):
    """nrtime._report under the pattern: no earlier than Z after the trigger's PDCCH and Z' after the
    last resource it reports (last_end None: no resource, Z alone), nor than not_before."""
    z, zp = N.z_pair(ctrl, x, kb)
    t = trig + (N.PDCCH_SYM + z) * N.sym_us(ctrl)
    if last_end is not None:
        t = max(t, last_end + zp * N.sym_us(ctrl))
    return ul_start(max(t, not_before), ctrl, car)


def triggers(ctrl, car):
    """Trigger A's and B's times, us, and the earliest target slot each one's sets may use."""
    if ctrl == 30:
        f = lambda tr: math.ceil((tr + N.first_csirs(30)) / SLOT - 1e-9)
        return [(0.0, f(0.0)), (N.slot_us(30), f(N.slot_us(30)))]
    assert car.opens_dl(0), "the first trigger opens the first triggering slot"
    b = next(k for k in range(1, HORIZON) if car.opens_dl(k))
    return [(0.0, 0), (b * SLOT, b)]


def sweep(n, ctrl, car, x, kb):
    """The sweep's earliest last report over the schedules the rules allow: (s_S, schedule)."""
    n_sets = math.ceil(N.N_BEAMS / n)
    sizes = [n] * (n_sets - 1) + [N.N_BEAMS - (n_sets - 1) * n]
    trig = triggers(ctrl, car)
    first = trig[0][1]
    hold = lambda m: [k for k in range(first, first + HORIZON) if car.set_syms(k, m)]
    full, short = hold(n), hold(sizes[-1])
    best = None
    # the short set in each of the slots that hold it, early enough to matter; the full sets in the
    # earliest remaining slots that hold them
    for ks in short[:2 * n_sets + 2]:
        fs = [k for k in full if k != ks][:n_sets - 1]
        slots = sorted(fs + [ks])
        size_at = {k: (sizes[-1] if k == ks else n) for k in slots}
        ends = {k: k * SLOT + (car.set_syms(k, size_at[k])[-1] + 1) * SYM for k in slots}
        # every split of the sets between A and B, at most four each, B's no earlier than B allows
        for bset in itertools.chain.from_iterable(
                itertools.combinations(slots, r) for r in range(0, N.SETS_PER_TRIGGER + 1)):
            aset = [k for k in slots if k not in bset]
            if not aset or len(aset) > N.SETS_PER_TRIGGER:
                continue
            if any(k < trig[1][1] or k - trig[1][1] > MAX_OFFSET for k in bset):
                continue
            if any(k - trig[0][1] > MAX_OFFSET for k in aset):
                continue
            groups = [(trig[0][0], max(ends[k] for k in aset))]
            if bset:
                groups.append((trig[1][0], max(ends[k] for k in bset)))
            for order in itertools.permutations(range(len(groups))):
                reps, ok = {}, True
                for g in order:
                    after = 0.0
                    if reps:     # never the slot of a report already placed, and after it
                        prev = max(reps.values())
                        after = (ctrl_slot(prev, ctrl) + 1) * N.slot_us(ctrl)
                    reps[g] = report(*groups[g], ctrl, car, x, kb, after)
                s = max(reps.values())
                key = (s, sorted(reps.values()))
                if best is None or key < best[0]:
                    best = (key, dict(slots=slots, short=ks, sizes=size_at, ends=ends,
                                      a=aset, b=list(bset), reports=[reps[g] for g in sorted(reps)]))
    assert best is not None, (car.name, car.phase, n, ctrl)
    return best[0][0], best[1]


def reduced(n, ctrl, car, x, kb, k=4):
    """The candidate's report: its k resources at the sweep's first positions (nrtime.res_end with
    n per slot) in the earliest slot whose symbols there are downlink."""
    pos = [NSYM - n + j for j in range(k)]
    first = triggers(ctrl, car)[0][1]
    slot = next(s for s in range(first, first + HORIZON)
                if pos[0] >= N.PDCCH_SYM and all(car.dl(s, j) for j in pos))
    assert slot - first <= MAX_OFFSET
    return report(0.0, slot * SLOT + (pos[-1] + 1) * SYM, ctrl, car, x, kb), slot


def guided(m, ctrl, car, x, kb):
    """Rule R31's reference: one set of m resources in the earliest slot that holds it."""
    first = triggers(ctrl, car)[0][1]
    slot = next(s for s in range(first, first + HORIZON) if car.set_syms(s, m))
    assert slot - first <= MAX_OFFSET
    return report(0.0, slot * SLOT + (car.set_syms(slot, m)[-1] + 1) * SYM, ctrl, car, x, kb), slot


def input_only(ctrl, car, x, kb):
    """A candidate whose input precedes the trigger: one report, held by Z alone."""
    return report(0.0, None, ctrl, car, x, kb)


def phases(name, ctrl):
    """The pattern slots at which the first trigger can fall: one opening with downlink symbols
    (120 kHz control), or the start of a 30 kHz slot, every fourth 120 kHz slot of the frame."""
    car = Carrier(name)
    if ctrl == 120:
        return [p for p in range(car.P) if Carrier(name, p).opens_dl(0)]
    return sorted({(4 * j) % car.P for j in range(car.P)})


def times(name, phase, n, ctrl, x, kb, k=4):
    car = Carrier(name, phase)
    s_S, sch = sweep(n, ctrl, car, x, kb)
    s_L, red_slot = reduced(n, ctrl, car, x, kb, k)
    W = car.wait()
    out = {"pattern": name, "phase": phase, "n": n, "ctrl": ctrl, "x": x, "kb": kb,
           "s_S": s_S, "s_L": s_L, "W": W, "budget": s_S - s_L + W, "reduced_slot": red_slot,
           "sweep": {key: sch[key] for key in ('slots', 'short', 'a', 'b', 'reports')},
           "input": input_only(ctrl, car, x, kb)}
    for m in (4, 8):
        out[f"guided{m}"], out[f"guided{m}_slot"] = guided(m, ctrl, car, x, kb)
    check_schedule(car, n, ctrl, sch, red_slot, k)
    return out


def check_schedule(car, n, ctrl, sch, red_slot, k):
    """Every resource on downlink symbols after the PDCCH, every report on uplink symbols within one
    slot, at most four sets per trigger, offsets within 31 slots, reports in distinct slots."""
    for s in sch['slots']:
        syms = car.set_syms(s, sch['sizes'][s])
        assert syms and syms[0] >= N.PDCCH_SYM and all(car.dl(s, j) for j in syms)
    assert len(sch['a']) <= N.SETS_PER_TRIGGER and len(sch['b']) <= N.SETS_PER_TRIGGER
    assert len(set(sch['slots'])) == len(sch['slots'])
    reps = sch['reports']
    assert len({ctrl_slot(r, ctrl) for r in reps}) == len(reps)
    if ctrl == 120:
        for r in reps:
            j = round(r / SYM)
            assert abs(j * SYM - r) < 1e-6
            s, q = divmod(j, NSYM)
            assert q + N.PUSCH_SYM <= NSYM and all(car.ul(s, q + i) for i in range(N.PUSCH_SYM))
    assert all(car.dl(red_slot, NSYM - n + j) for j in range(k))


def gate_paired():
    """Gate F: on paired spectrum the schedules reproduce nrtime.times, and W = h."""
    ok = Carrier('paired').wait() == SLOT
    for n in (12, 8):
        for ctrl in (120, 30):
            for x, kb in N.capabilities(ctrl):
                t = times('paired', 0, n, ctrl, x, kb)
                r = N.times(n, ctrl, x, kb)
                ok &= abs(t['s_S'] - r['d_S']) < 1e-9 and abs(t['s_L'] - r['reduced']) < 1e-9
                ok &= abs(t['budget'] - (r['saving'] + SLOT)) < 1e-9
    return bool(ok)


def table(k=4):
    """Every pattern, control carrier, resources per slot, declarable timing and phase."""
    rows = []
    for name in PATTERNS:
        for ctrl, n in ((120, 12), (30, 12), (120, 8)):
            for x, kb in N.capabilities(ctrl):
                for ph in phases(name, ctrl):
                    rows.append(times(name, ph, n, ctrl, x, kb, k))
    return rows


if __name__ == '__main__':
    assert gate_paired(), "gate F: the paired schedules do not reproduce nrtime.py"
    print("gate F holds")
    for r in table():
        if (r['x'], r['kb']) in ((14, 14), (28, None)):
            print(r['pattern'], r['ctrl'], r['n'], r['x'], r['kb'], r['phase'],
                  round(r['s_S'], 1), round(r['s_L'], 1), r['W'], round(r['budget'], 1),
                  round(r['guided4'], 1), round(r['guided8'], 1), round(r['input'], 1))
