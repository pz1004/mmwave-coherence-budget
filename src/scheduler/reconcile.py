#!/usr/bin/env python3
"""Rule R21, the screen's inputs read over every retained timing.  Declared in protocol.json
(analysis 'reconcile') before it ran; nothing is timed, retrained or re-solved.

R21.  Table I, rule R5's reading, every exclusion margin and every number computed from them read,
for each implementation, every runtime measured: the stored timings and the three repeated runs of
trace_decisive.py (src/latency/trace_decisive/), at the host boundary, the boundary of record, where
the beam index is a CPU integer inside the timed call.  A runtime of the repeated runs enters as the
protocol enters a timing, the largest of its three per-seed 95th percentiles in a run, and of the
three runs the smallest, the value that favors the candidate.  Its lower value at adverse inputs is
margins.py's for a run that kept every call: per seed the distribution-free lower confidence bound
X_(r) of the seed's 95th percentile, r = margins.rank(n, 0.05/3), and of the nine seed-runs the
smallest.  These are nominal, as margins.py's are: successive calls are correlated.

This script first rebuilds rule R5's rows (reading_rules.r5) and margins.json's rows from the
stored timings alone, and stops unless both are reproduced exactly; it then adds the repeated runs'
two runtimes to the three implementations they timed and recomputes every derived field with the
same formulas.  The fastest timed call of the repeated runs is recorded beside them.

Inputs, read and never written: rules.json and rules_rx.json (rule R5), margins.json, results.json
(the grid of T), lat_*.json (each profile's per-probe processing), and the eighteen outputs of
src/latency/trace_decisive/, whose sha256 must be those protocol.json records (analysis
'trace_decisive', files_sha256).

Gates:
  G1_stored_reproduced  without the repeated runs, every R5 row equals rules.json's and
                        rules_rx.json's, every margins row equals margins.json's, and R5's per-model
                        summary equals the stored one;
  G2_verdicts_unchanged with them, every row keeps its exclusion, its budget band, its exclusion at
                        adverse inputs and its infeasibility, and R5's per-model summary is unchanged;
  G3_pins               the outputs read are those protocol.json records, and the script whose
                        sha256 it records (scripts_sha256) wrote them;
  G4_runs_clean         every run started, was not a smoke test, its recheck drift stayed within
                        trace_decisive.py's 10% (not contended), and was the first attempt;
  G5_rank               the charge of every configuration is its largest per-seed 95th percentile,
                        each per-seed 95th percentile is the nearest-rank one of its calls, and the
                        rank at 10,000 calls is 9,453.

usage:  python3 reconcile.py [--check-only]       -> reconcile.json (written once)
"""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = os.path.normpath(os.path.join(HERE, '..', 'latency'))
TD = os.path.join(LAT, 'trace_decisive')
sys.path.insert(0, HERE)
sys.path.insert(0, LAT)
import protocol
import margins as MGN             # rank, nr_end, SLOT, ALPHA; importing it writes nothing
import reading_rules as RR        # R5's constants
from calibrate import nearest_rank_p95

J = lambda *p: json.load(open(os.path.join(*p)))
CASES = {'A': ('resnet18_vision', 'cpu-8t-pcore'), 'B': ('vit_b16_vision', 'gpu-resident'),
         'C': ('resnet18_vision', 'gpu-resident')}
RUNS, BOUNDARY = (1, 2, 3), 'host'
RT = {'uncompiled': 'eager, repeated runs', 'torchscript': 'TorchScript, repeated runs'}
R5_VERDICTS = ('excluded', 'excluded_if_network_only_included', 'at_budget')
MG_VERDICTS = ('excluded_robust', 'infeasible_robust')


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def r5_row(row, charges, bk, grid):
    """reading_rules.r5's fields for one (model, profile) at the given runtimes' charges."""
    tp = RR.T_AIR_US + bk
    ts = (RR.M_BEAMS - RR.K_VISION) * tp
    assert tp == row['t_p_us'] and ts == row['tau_star_us'], (row['component'], row['profile'])
    tmin = min(charges.values())
    ratios = {k: v / ts for k, v in charges.items()}
    rmin = min(ratios.values())
    net_only = row['network_only_us']
    feas = [T for T in grid if T - RR.K_VISION * tp > tmin]
    admit = {f"M{M}": {f"K{K}": {
                 ("measured" if air is None else f"air{air:g}"):
                     tmin < (M - K) * ((RR.T_AIR_US if air is None else air) + bk)
                 for air in (None, *RR.ADMIT_AIR)}
             for K in RR.ADMIT_K if K < M} for M in RR.ADMIT_M}
    out = dict(row)
    out.update({"charges_us": charges, "ratios": ratios, "ratio_min": rmin,
                "excluded": tmin >= ts,
                "excluded_if_network_only_included":
                    min([tmin] + ([net_only] if net_only else [])) >= ts,
                "at_budget": abs(rmin - 1.0) <= RR.AT_BUDGET_BAND,
                "tp_star_us": tmin / (RR.M_BEAMS - RR.K_VISION),
                "tp_star_air_us": tmin / (RR.M_BEAMS - RR.K_VISION) - bk,
                "first_feasible_T_us": feas[0] if feas else None,
                "admission": admit})
    return out


def r5_summary(rows):
    models = {}
    for r in rows:
        models.setdefault(r['component'], []).append(r)
    return {c: {"excluded_profiles": [r['profile'] for r in rs if r['excluded']],
                "every_profile": all(r['excluded'] for r in rs),
                "at_budget_profiles": [r['profile'] for r in rs if r['at_budget']]}
            for c, rs in models.items()}


def mg_row(mrow, r5r, runtimes, MG):
    """margins.py's fields for one row at the given runtimes."""
    K, t_max = MG['K'], MG['t_max_us']
    fastest = min(runtimes, key=lambda k: runtimes[k]['charged_us'])
    tau_lo = min(v['lower_us'] for v in runtimes.values())
    m = lambda dl, eta, ds: dl - (1 - eta) * t_max - eta * ds
    out = dict(mrow)
    out.update({"runtimes": runtimes, "fastest_runtime": fastest,
                "tau_us": runtimes[fastest]['charged_us'], "tau_lo_us": tau_lo,
                "tau_lo_runtime": min(runtimes, key=lambda k: runtimes[k]['lower_us']),
                "tau_lo_kind": min(runtimes.values(), key=lambda v: v['lower_us'])['kind'],
                "dL_us": K * r5r['t_p_us'] + runtimes[fastest]['charged_us'],
                "dL_lo_us": K * MG['t_air_us'] + tau_lo,
                "nr_dL_lo_us": MGN.nr_end(K - 1) - MGN.SLOT + tau_lo})
    out["margin_us"] = m(out['dL_lo_us'], MG['eta_lo']['probe_model'], MG['dS_hi_us'])
    out["excluded_robust"] = out["margin_us"] >= 0
    out["infeasible_robust"] = out['dL_lo_us'] >= t_max
    out["nr_margin_us"] = {d: m(out['nr_dL_lo_us'], e, MG['nr_dS_us'])
                           for d, e in MG['eta_lo']['csirs'].items()}
    return out


def repeated(case, gates):
    """The repeated runs of one case at the host boundary: per runtime, the charge and lower value
    as R21 reads them, with every seed-run's value; and the fastest timed call."""
    out, fastest = {}, []
    for rt, name in RT.items():
        seed_p95, seed_lo, ns, seed_run, seed_id, keys, charges = [], [], [], [], [], [], []
        ranks = set()
        for run in RUNS:
            j = J(TD, f"r{run}_{case}.json")
            z = np.load(os.path.join(TD, f"r{run}_{case}.npz"))
            cfg = next(x for x in j['configurations']
                       if x['runtime'] == rt and x['boundary'] == BOUNDARY)
            p95 = [s['p95_ns'] / 1e3 for s in cfg['per_seed']]
            gates['G5_rank'] &= abs(max(p95) - cfg['charge_ns'] / 1e3) < 1e-9
            charges.append(cfg['charge_ns'] / 1e3)
            for s in cfg['per_seed']:
                key = f"{rt}.{BOUNDARY}.s{s['seed']}"
                a = z[key]
                gates['G5_rank'] &= (a.size == s['n_timed'] and nearest_rank_p95(a) == s['p95_ns']
                                     and int(a.min()) == s['min_ns'])
                r = MGN.rank(a.size, MGN.ALPHA / len(cfg['per_seed']))
                ranks.add((a.size, r))
                seed_p95.append(s['p95_ns'] / 1e3)
                seed_lo.append(float(np.sort(a)[r - 1]) / 1e3)
                ns.append(int(a.size))
                seed_run.append(run)
                seed_id.append(s['seed'])
                keys.append(f"r{run}_{case}.npz:{key}")
                fastest.append(s['min_ns'] / 1e3)
        assert len(ranks) == 1, ranks
        (n, r), = ranks
        gates['G5_rank'] &= (n, r) == (10_000, 9_453)
        i = int(np.argmin(seed_lo))
        out[name] = {"charged_us": min(charges), "lower_us": min(seed_lo),
                     "run_charges_us": charges, "charged_run": RUNS[int(np.argmin(charges))],
                     "seed_p95_us": seed_p95, "seed_lower_us": seed_lo, "n_timed": ns,
                     "seed_run": seed_run, "seed_id": seed_id, "npz_keys": keys, "rank": r,
                     "lower_run": seed_run[i], "lower_seed": seed_id[i],
                     "kind": "order-statistic bound"}
    return out, min(fastest)


def main():
    check = '--check-only' in sys.argv
    pth = os.path.join(HERE, 'reconcile.json')
    assert check or not os.path.exists(pth), "reconcile.json exists: it is not rewritten"
    TDR = protocol.analysis('trace_decisive')
    gates = {"G1_stored_reproduced": True, "G2_verdicts_unchanged": True, "G3_pins": True,
             "G4_runs_clean": True, "G5_rank": True}
    # ---- G3: the outputs read are the stored runs, written by the timing script as it ran -------
    for f, h in TDR['files_sha256'].items():
        gates['G3_pins'] &= sha(os.path.join(HERE, f)) == h
    _pin = TDR['scripts_sha256']['../latency/trace_decisive.py']
    for c in CASES:
        for run in RUNS:
            j = J(TD, f"r{run}_{c}.json")
            gates['G3_pins'] &= j['env']['script_sha256'] == _pin and \
                (j['component'], j['profile']) == CASES[c] and j['run'] == run
            gates['G4_runs_clean'] &= (bool(j['started']) and not j['smoke'] and not j['contended']
                                       and j['attempt'] == 1)
    # ---- G1: the stored rows, rebuilt ------------------------------------------------------------
    R5 = J(HERE, 'rules.json')['R5']
    gates['G1_stored_reproduced'] &= R5 == J(HERE, 'rules_rx.json')['R5']
    MG = J(HERE, 'margins.json')
    grid = J(HERE, 'results.json')['grid_us']
    bk = {p: {r['component']: r['charge_us'] for r in J(HERE, f)['rows']}['probe_update']
          for p, f in RR.LAT_OF.items()}
    MGR = {(r['component'], r['profile']): r for r in MG['rows']}
    rebuilt = [r5_row(r, dict(r['charges_us']), bk[r['profile']], grid) for r in R5['rows']]
    gates['G1_stored_reproduced'] &= rebuilt == R5['rows'] and r5_summary(rebuilt) == R5['by_model']
    rb_mg = [mg_row(MGR[(r['component'], r['profile'])], r,
                    dict(MGR[(r['component'], r['profile'])]['runtimes']), MG) for r in R5['rows']]
    gates['G1_stored_reproduced'] &= (rb_mg == [MGR[(r['component'], r['profile'])]
                                                for r in R5['rows']])
    # ---- the repeated runs join the runtimes they timed -----------------------------------------
    rows, mrows, detail, fastest = [], [], {}, {}
    for r in R5['rows']:
        k = (r['component'], r['profile'])
        mr = MGR[k]
        case = next((c for c, kk in CASES.items() if kk == k), None)
        if case is None:
            rows.append(r)
            mrows.append(mr)
            continue
        rep, fast = repeated(case, gates)
        detail[case] = {"component": k[0], "profile": k[1], "runtimes": rep}
        fastest[case] = fast
        charges = dict(r['charges_us'], **{n: v['charged_us'] for n, v in rep.items()})
        nr = r5_row(r, charges, bk[r['profile']], grid)
        rows.append(nr)
        mrows.append(mg_row(mr, nr, dict(mr['runtimes'], **rep), MG))
    summary = r5_summary(rows)
    # ---- G2: no verdict changes ------------------------------------------------------------------
    for old, new in zip(R5['rows'], rows):
        gates['G2_verdicts_unchanged'] &= all(old[v] == new[v] for v in R5_VERDICTS)
    for old, new in zip([MGR[(r['component'], r['profile'])] for r in R5['rows']], mrows):
        gates['G2_verdicts_unchanged'] &= all(old[v] == new[v] for v in MG_VERDICTS)
    gates['G2_verdicts_unchanged'] &= summary == R5['by_model']
    changed = [[n['component'], n['profile']] for o, n, om, nm in zip(
        R5['rows'], rows, [MGR[(r['component'], r['profile'])] for r in R5['rows']], mrows)
        if min(o['charges_us'].values()) != min(n['charges_us'].values())
        or om['tau_lo_us'] != nm['tau_lo_us']]
    admission_changed = [[n['component'], n['profile'], M, K, a]
                         for o, n in zip(R5['rows'], rows)
                         for M, ks in n['admission'].items() for K, aa in ks.items()
                         for a, v in aa.items() if o['admission'][M][K][a] != v]
    gate = dict(gates, holds=all(gates.values()))
    print(json.dumps(gate))
    for case, d in detail.items():
        for n, v in d['runtimes'].items():
            print(f"{case} {n:28s} charge {v['charged_us']:9.1f} (run {v['charged_run']})  lower "
                  f"{v['lower_us']:9.1f} (run {v['lower_run']}, seed {v['lower_seed']}, r {v['rank']})")
    for n in rows:
        k = (n['component'], n['profile'])
        if list(k) in changed:
            mm = mrows[rows.index(n)]
            print(f"changed {k}: tau {min(n['charges_us'].values()):.1f}  tau_lo {mm['tau_lo_us']:.1f}  "
                  f"margin {mm['margin_us']:.1f}  t_p* {n['tp_star_us']:.2f}")
    print("changed:", changed, " admission changed:", admission_changed)
    print("fastest repeated call (us):", {c: round(v, 1) for c, v in fastest.items()})
    if not gate['holds']:
        raise SystemExit("gate failed: nothing written")
    if check:
        print("gates hold; nothing written")
        return
    out = {"rule": "R21", "script_sha256": sha(os.path.abspath(__file__)),
           "inputs_sha256": {f: sha(os.path.join(HERE, f)) for f in
                             ('rules.json', 'rules_rx.json', 'margins.json', 'results.json')},
           "boundary": BOUNDARY, "runs": list(RUNS), "alpha": MGN.ALPHA,
           "gate": gate, "changed": changed, "admission_changed": admission_changed,
           "fastest_call_repeated_us": fastest, "repeated": detail,
           "r5_rows": rows, "r5_by_model": summary, "margin_rows": mrows}
    json.dump(out, open(pth, 'w'), indent=1)
    print("wrote reconcile.json")


if __name__ == '__main__':
    main()
