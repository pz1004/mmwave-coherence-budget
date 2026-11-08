#!/usr/bin/env python3
"""Rule R36: rule R30's time-division CSI-RS procedure (nrtdd.py) with both carriers at 60 kHz.
Declared in protocol.json (analysis 'tdd_mu') before it ran; make_draft.py imports it.  nrtime.py,
nrtdd.py and nrmu.py are imported unchanged.

nrtdd.py schedules the procedure on 120 kHz carriers.  Here the target and the control carrier share
one subcarrier spacing and one slot format, 60 or 120 kHz, and every rule of nrtdd.py stands:
triggers in a PDCCH in the first two symbols of a slot that opens with downlink symbols; a set's m
one-symbol resources in one slot's last m downlink symbols after the PDCCH's two; at most four sets
per trigger, trigger A in the first triggering slot and B in the next, triggering offsets of at most
31 slots; reports as two-symbol transmissions within one slot's uplink symbols, never two in one
slot; data from the start of a slot that opens with downlink symbols, at most W after a report.
(Z, Z') and the declarable timings are rule R35's (nrmu.py): at mu = 2, (min(44, X2 + KB1), X2),
X2 in {8, 14, 28}, KB1 in {14, 28, 48, 224, 336} symbols.

Patterns, TS 38.101-2 V18.14.0 Annex A, read 2026-11-08: at 60 kHz, DDDSU every 1.25 ms
(Table A.3.3.1-1, the downlink reference channels) and DDDSUUUU every 2 ms (Table A.2.3-1, the uplink
ones), both with S = 4D+6G+4U; at 120 kHz, nrtdd.py's DDDSU and 7DS8U.  The 60 kHz special slot holds
two resources after the PDCCH, so no full or four-resource set fits in it.

Gates.  T: at 120 kHz the schedules reproduce nrtdd.times (s_S, s_L, W, the budget, both guided
references and the SS-input candidate) for both patterns, every phase, 12 and 8 resources per slot and
every declarable timing, to 1e-9.  P: on paired spectrum at 60 kHz they reproduce nrmu.times.  Every
schedule is checked as built (check_schedule).

`python3 nrtddmu.py --gate` runs the gates alone; `python3 nrtddmu.py` also prints R36's values.
"""
import itertools
import math
import sys

import nrmu as M
import nrtdd as TD
import nrtime as N

NSYM = 14
MAX_OFFSET = TD.MAX_OFFSET
HORIZON = TD.HORIZON
PATTERNS = {
    # name: (period in slots, nrofDownlinkSlots, nrofDownlinkSymbols, nrofUplinkSlots,
    #        nrofUplinkSymbols), TS 38.101-2 V18.14.0
    60: {'DDDSU': (5, 3, 4, 1, 4),            # Table A.3.3.1-1: 1.25 ms, S = 4D+6G+4U
         'DDDSUUUU': (8, 3, 4, 4, 4)},        # Table A.2.3-1: 2 ms, S = 4D+6G+4U
    120: dict(TD.PATTERNS),                   # nrtdd.py: DDDSU and 7DS8U
}


def symbols(name, scs):
    """One period of the pattern: per slot, per symbol, 'D', 'U' or 'G'; 'paired' is one slot open
    to both directions."""
    if name == 'paired':
        return [['B'] * NSYM]
    P, dls, dly, uls, uly = PATTERNS[scs][name]
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
    """A carrier of subcarrier spacing scs with a pattern, its slot 0 at the pattern's slot `phase`."""

    def __init__(self, name, scs, phase=0):
        self.name, self.scs, self.pat = name, scs, symbols(name, scs)
        self.P, self.phase = len(self.pat), phase % len(self.pat)
        self.slot, self.sym = N.slot_us(scs), N.sym_us(scs)

    def kind(self, slot, sym):
        return self.pat[(self.phase + slot) % self.P][sym]

    def dl(self, slot, sym):
        return self.kind(slot, sym) in 'DB'

    def ul(self, slot, sym):
        return self.kind(slot, sym) in 'UB'

    def opens_dl(self, slot):
        return all(self.dl(slot, j) for j in range(N.PDCCH_SYM))

    def dl_run(self, slot):
        j = 0
        while j < NSYM and self.dl(slot, j):
            j += 1
        return j

    def set_syms(self, slot, m):
        r = self.dl_run(slot)
        return list(range(r - m, r)) if r - m >= N.PDCCH_SYM and m > 0 else None

    def wait(self):
        """W: the longest gap between consecutive slot starts that open with downlink symbols, us."""
        starts = [k for k in range(self.P) if self.opens_dl(k)]
        assert starts, self.name
        gaps = [(b - a) % self.P or self.P for a, b in zip(starts, starts[1:] + starts[:1])]
        return max(gaps) * self.slot


def ul_start(t, car):
    """The earliest start, at or after t (us), of a two-symbol transmission within one slot's uplink
    symbols of the control carrier, which shares the target's slot format."""
    j = math.ceil(t / car.sym - 1e-9)
    for j in range(j, j + NSYM * HORIZON):
        s, q = divmod(j, NSYM)
        if q + N.PUSCH_SYM <= NSYM and all(car.ul(s, q + i) for i in range(N.PUSCH_SYM)):
            return j * car.sym
    raise AssertionError("no uplink within the horizon")


def ctrl_slot(t, car):
    return math.floor(t / car.slot + 1e-9)


def report(trig, last_end, car, x, kb, not_before=0.0):
    """nrtdd.report on the carrier's spacing: no earlier than Z after the trigger's PDCCH and Z' after
    the last resource it reports (last_end None: Z alone), nor than not_before."""
    z, zp = M.z_pair(car.scs, x, kb)
    t = trig + (N.PDCCH_SYM + z) * car.sym
    if last_end is not None:
        t = max(t, last_end + zp * car.sym)
    return ul_start(max(t, not_before), car)


def triggers(car):
    """Trigger A's and B's times, us, and the earliest target slot each one's sets may use."""
    assert car.opens_dl(0), "the first trigger opens the first triggering slot"
    b = next(k for k in range(1, HORIZON) if car.opens_dl(k))
    return [(0.0, 0), (b * car.slot, b)]


def sweep(n, car, x, kb):
    """nrtdd.sweep on the carrier's spacing: the earliest last report over the schedules the rules
    allow, (s_S, schedule)."""
    n_sets = math.ceil(N.N_BEAMS / n)
    sizes = [n] * (n_sets - 1) + [N.N_BEAMS - (n_sets - 1) * n]
    trig = triggers(car)
    first = trig[0][1]
    hold = lambda m: [k for k in range(first, first + HORIZON) if car.set_syms(k, m)]
    full, short = hold(n), hold(sizes[-1])
    best = None
    for ks in short[:2 * n_sets + 2]:
        fs = [k for k in full if k != ks][:n_sets - 1]
        slots = sorted(fs + [ks])
        size_at = {k: (sizes[-1] if k == ks else n) for k in slots}
        ends = {k: k * car.slot + (car.set_syms(k, size_at[k])[-1] + 1) * car.sym for k in slots}
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
                reps = {}
                for g in order:
                    after = 0.0
                    if reps:
                        prev = max(reps.values())
                        after = (ctrl_slot(prev, car) + 1) * car.slot
                    reps[g] = report(*groups[g], car, x, kb, after)
                s = max(reps.values())
                key = (s, sorted(reps.values()))
                if best is None or key < best[0]:
                    best = (key, dict(slots=slots, short=ks, sizes=size_at, ends=ends,
                                      a=aset, b=list(bset), reports=[reps[g] for g in sorted(reps)]))
    assert best is not None, (car.name, car.scs, car.phase, n)
    return best[0][0], best[1]


def reduced(n, car, x, kb, k=4):
    """The candidate's report: its k resources at the sweep's first positions in the earliest slot
    whose symbols there are downlink."""
    pos = [NSYM - n + j for j in range(k)]
    first = triggers(car)[0][1]
    slot = next(s for s in range(first, first + HORIZON)
                if pos[0] >= N.PDCCH_SYM and all(car.dl(s, j) for j in pos))
    assert slot - first <= MAX_OFFSET
    return report(0.0, slot * car.slot + (pos[-1] + 1) * car.sym, car, x, kb), slot


def guided(m, car, x, kb):
    """Rule R31's reference: one set of m resources in the earliest slot that holds it."""
    first = triggers(car)[0][1]
    slot = next(s for s in range(first, first + HORIZON) if car.set_syms(s, m))
    assert slot - first <= MAX_OFFSET
    return report(0.0, slot * car.slot + (car.set_syms(slot, m)[-1] + 1) * car.sym, car, x, kb), slot


def input_only(car, x, kb):
    return report(0.0, None, car, x, kb)


def phases(name, scs):
    """The pattern slots at which the first trigger can fall: those opening with downlink symbols."""
    car = Carrier(name, scs)
    return [p for p in range(car.P) if Carrier(name, scs, p).opens_dl(0)]


def check_schedule(car, n, sch, red_slot, k):
    for s in sch['slots']:
        syms = car.set_syms(s, sch['sizes'][s])
        assert syms and syms[0] >= N.PDCCH_SYM and all(car.dl(s, j) for j in syms)
    assert len(sch['a']) <= N.SETS_PER_TRIGGER and len(sch['b']) <= N.SETS_PER_TRIGGER
    assert len(set(sch['slots'])) == len(sch['slots'])
    reps = sch['reports']
    assert len({ctrl_slot(r, car) for r in reps}) == len(reps)
    for r in reps:
        j = round(r / car.sym)
        assert abs(j * car.sym - r) < 1e-6
        s, q = divmod(j, NSYM)
        assert q + N.PUSCH_SYM <= NSYM and all(car.ul(s, q + i) for i in range(N.PUSCH_SYM))
    assert all(car.dl(red_slot, NSYM - n + j) for j in range(k))


def times(name, scs, phase, n, x, kb, k=4):
    car = Carrier(name, scs, phase)
    s_S, sch = sweep(n, car, x, kb)
    s_L, red_slot = reduced(n, car, x, kb, k)
    W = car.wait()
    out = {"pattern": name, "scs": scs, "phase": phase, "n": n, "x": x, "kb": kb,
           "s_S": s_S, "s_L": s_L, "W": W, "budget": s_S - s_L + W, "reduced_slot": red_slot,
           "input": input_only(car, x, kb)}
    for m in (4, 8):
        out[f"guided{m}"], out[f"guided{m}_slot"] = guided(m, car, x, kb)
    check_schedule(car, n, sch, red_slot, k)
    return out


def gate_t():
    """Gate T: at 120 kHz the schedules reproduce nrtdd.times."""
    ok = True
    for name in TD.PATTERNS:
        for n in (12, 8):
            for x, kb in N.capabilities(120):
                for ph in TD.phases(name, 120):
                    a, b = times(name, 120, ph, n, x, kb), TD.times(name, ph, n, 120, x, kb)
                    for f in ('s_S', 's_L', 'W', 'budget', 'guided4', 'guided8', 'input'):
                        ok &= abs(a[f] - b[f]) < 1e-9
    return bool(ok)


def gate_p():
    """Gate P: on paired spectrum at 60 kHz the schedules reproduce nrmu.times."""
    ok = True
    for n in (12, 8):
        for x, kb in M.capabilities(60):
            a, b = times('paired', 60, 0, n, x, kb), M.times(n, 60, x, kb)
            ok &= abs(a['s_S'] - b['s_S']) < 1e-9 and abs(a['s_L'] - b['s_L']) < 1e-9
            ok &= abs(a['budget'] - b['budget']) < 1e-9
    return bool(ok)


def table(scs=60, k=4):
    """Every pattern at this spacing, 12 and 8 resources per slot, declarable timing and phase."""
    return [times(name, scs, ph, n, x, kb, k) for name in PATTERNS[scs] for n in (12, 8)
            for x, kb in M.capabilities(scs) for ph in phases(name, scs)]


if __name__ == '__main__':
    assert gate_t(), "gate T: the 120 kHz schedules do not reproduce nrtdd.py"
    assert gate_p(), "gate P: the paired 60 kHz schedules do not reproduce nrmu.py"
    print("gates T and P hold")
    if '--gate' in sys.argv[1:]:
        sys.exit(0)
    rows = table()
    for name in PATTERNS[60]:
        for n in (12, 8):
            rs = [r for r in rows if r['pattern'] == name and r['n'] == n]
            top = max(rs, key=lambda r: r['budget'])
            print(f"60 kHz {name}, {n} per slot: W {rs[0]['W']:.0f} us; budget "
                  f"{min(r['budget'] for r in rs):.1f}-{top['budget']:.1f} us; largest at phase "
                  f"{top['phase']}, X2 = {top['x']}, KB1 = {top['kb']}: s_S {top['s_S']:.1f}, "
                  f"s_L {top['s_L']:.1f}")
