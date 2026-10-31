#!/usr/bin/env python3
"""The reading rules R1-R4, applied mechanically to the run artifacts.

The rules were written and recorded in protocol.json before any run with the
ten-seed list, the variants of R2 and R3 or the sweeps. This script applies
them and writes rules.json. make_draft.py reads the framing from rules.json; no framing
decision is taken by hand.

Conventions, as protocol.json states them:
  * "A beats B at a site" means the grid-averaged paired difference A - B has a
    lower bound strictly above zero under EVERY interval the artifact carries:
    the seed-by-block percentile bootstrap and the t interval when present,
    otherwise the prespecified block percentile interval. On top of that,
    its p-value must survive Holm correction across the six evaluation sites.
    "Loses" is the mirror image. When both a bootstrap and a t p-value exist,
    the larger is used.
  * Bounds stored rounded to five decimals are read as stored. A bound that
    rounds to zero does not count, which errs toward "no".
  * The headline controller is the one report.json names (the matched rule's
    choice on the validation site). Every comparator takes its estimation
    suffix.

  R1  Online allocation stays the paper's first contribution only if all
      three hold:
        - on at least 4 of the 6 evaluation sites the controller beats both
          its cost-blind twin and its no-compute twin, on the same site;
        - at least one of those sites is a designated test site;
        - the controller loses to the deployable fixed strategy on neither
          test site.
      Otherwise the paper leads with the accounting, the accuracy-free budget
      and the per-profile exclusion, and reports the controller as a bounded
      case study.
  R2  "How deep to go" is kept only if the staged controller beats both the
      repeated-exit-1 controller and the cheapest-exit controller on at least
      4 of 6 sites. Otherwise the text says "whether and when to invoke".
  R3  If the FC- or CNN-as-compute controller beats the anytime-predictor
      controller under the same restrictions (no reordering, compute only at
      K in {4, 8, 16, 32, 64}, at most once) on at least 4 of 6 sites, the body
      says the controller's results depend on predictor strength and reports
      both. The headline controller never switches.
  R4  An exclusion claim is stated only for the hardware profiles where it
      holds: "delivers zero rate at every tested deadline" (feasibility) or
      "cannot beat the exhaustive sweep at any deadline" (the accuracy-free
      budget, under eta_L <= eta_S).

Three further rules were written before the runs they read.  None of them
changes R1's framing, the gate or the headline selection; they decide wording
only.
  R5  "Cannot beat the exhaustive sweep at any deadline" is stated for a
      (model, profile) only if EVERY measured runtime's charge -- eager at
      either count protocol, TorchScript whole-decision -- is at least that
      profile's tau* = (M-K) t_p, with t_p the profile's own probe charge;
      "every profile" only if that holds on all four.  Every exclusion is
      stated with its break-even probe charge t_p* = tau_min/(M-K), and a
      cheapest ratio within 10% of one (the timing protocol's pass-agreement
      band) is worded "at the budget".
  R6  (a) "tau* holds under channel aging at speed v" is stated only if the
      sweep's held-beam rate dominates every fixed K-probe method's from the
      sweep's commit on (aging.json), on every evaluation site, at v;
      otherwise the failures are named, and the claim is stated up to the
      largest speed at which it holds.  (b) "Block fading is adequate within
      a period at v" only if the beam-aging loss stays below 1% for every
      Delta up to 10 ms on every site; otherwise the largest deadline at
      which it stays below 1% is given per speed.
  R7  "The case study is predictor-limited" is stated only if compute-pays is
      established (as R1 counts it) on at least 4 of 6 sites at some
      oracle-mixing quality q <= 0.25; "latency-awareness depends on predictor
      quality" only if aware-vs-blind is established on at least 4 of 6 sites
      at some q.

usage: python3 reading_rules.py [--out rules.json]
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAT_DIR = os.path.join(HERE, '..', 'latency')
NEED = 4                                   # sites out of six, for R1-R3
HEAVY = ('resnet18_vision', 'vit_b16_vision', 'transformer_small')
PROFILES = ('cpu-1t-pcore', 'cpu-1t-ecore', 'cpu-8t-pcore', 'gpu-resident')
K_VISION = 4


def J(n):
    return json.load(open(os.path.join(HERE, n)))


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank):
            rej[i] = True
        else:
            break
    return rej


def contrast(site, family, key):
    """The best available interval set for one grid-averaged contrast.

    The hierarchical (seed-by-block) aggregate is used when the run produced
    one. Otherwise the prespecified block aggregate is used, with seeds
    averaged first, as the frozen protocol runs it."""
    hier = site.get(f'{family}_aggregate_hier', {}).get(key)
    if hier is not None:
        return {"diff": hier['diff'], "bounds": [hier['ci95'], hier['ci95_t']],
                "p": max(hier['p'], hier['p_t']), "estimator": "seed x block + t"}
    agg = site.get(f'{family}_aggregate', {}).get(key)
    if agg is None:
        return None
    return {"diff": agg['diff'], "bounds": [agg['ci95']], "p": agg['p'],
            "estimator": "block percentile, seeds averaged first (pre-registered)"}


def verdicts(sites, family, key):
    """Per site: +1 beats, -1 loses, 0 neither, None not computed."""
    cs = [contrast(s, family, key) for s in sites]
    if any(c is None for c in cs):
        return [None] * len(sites), cs
    rej = holm([c['p'] for c in cs])
    out = []
    for c, r in zip(cs, rej):
        lo = min(b[0] for b in c['bounds'])
        hi = max(b[1] for b in c['bounds'])
        out.append(+1 if (r and lo > 0) else -1 if (r and hi < 0) else 0)
    return out, cs


def r4():
    atlas = json.load(open(os.path.join(LAT_DIR, 'atlas.json')))['charge_us']
    full = os.path.join(LAT_DIR, 'p_gpu_heavy_full.json')   # full-count GPU re-timing
    if os.path.exists(full):
        for row in json.load(open(full))['rows']:
            atlas[f"{row['component']}|gpu-resident"] = row['charge_us']
    rho = J('rho.json')
    tp, grid = rho['t_p_us'], J('results.json')['grid_us']
    tau_star = (64 - K_VISION) * tp
    rows = []
    for comp in HEAVY:
        for prof in PROFILES:
            tau = atlas.get(f'{comp}|{prof}')
            if tau is None:
                continue
            feas = [T for T in grid if T - K_VISION * tp > tau]
            rows.append({"component": comp, "profile": prof, "tau_us": tau,
                         "feasible_deadlines_us": feas,
                         "zero_rate_at_every_deadline": not feas,
                         "tau_star_us": round(tau_star, 3),
                         "cannot_beat_sweep_anywhere": tau >= tau_star,
                         "margin_over_tau_star": round(tau / tau_star, 3)})
    return rows


M_BEAMS, T_AIR_US = 64, 20.0
LAT_OF = {'cpu-1t-pcore': 'lat_cpu1t_pcore.json', 'cpu-1t-ecore': 'lat_cpu1t_ecore.json',
          'cpu-8t-pcore': 'lat_cpu8t_pcore.json', 'gpu-resident': 'lat_gpu.json'}
O2_OF = {'cpu-1t-pcore': 'o2_cpu1t_pcore.json', 'cpu-1t-ecore': 'o2_cpu1t_ecore.json',
         'gpu-resident': 'o2_gpu.json'}
O_OF = {'cpu-1t-pcore': 'o_cpu1t_pcore.json', 'cpu-1t-ecore': 'o_cpu1t_ecore.json',
        'gpu-resident': 'o_gpu.json'}
AT_BUDGET_BAND = 0.10                      # the timing protocol's pass-agreement band
ADMIT_M, ADMIT_K, ADMIT_AIR = (16, 32, 64, 128, 256), (4, 8, 16), (50.0, 125.0, 250.0)
LOSS_THR = 0.01
Q_LIMITED_MAX = 0.25
PREDQ = (('oq0.125', 0.125), ('oq0.25', 0.25), ('oq0.5', 0.5), ('oq1', 1.0))


def r5():
    """Exclusion per (model, profile) over every measured runtime, and the
    admission map over codebook size, probe count and probe charge."""
    atlas = json.load(open(os.path.join(LAT_DIR, 'atlas.json')))['charge_us']
    full = {r['component']: r['charge_us'] for r in
            json.load(open(os.path.join(LAT_DIR, 'p_gpu_heavy_full.json')))['rows']}
    grid = J('results.json')['grid_us']
    o2 = {p: {r['component']: r['charge_us'] for r in
              json.load(open(os.path.join(LAT_DIR, f)))['rows']} for p, f in O2_OF.items()}
    o1 = {p: {r['component']: r['charge_us'] for r in
              json.load(open(os.path.join(LAT_DIR, f)))['rows']} for p, f in O_OF.items()}
    rows, models = [], {}
    for prof in PROFILES:
        lat = J(LAT_OF[prof])
        bk = {r['component']: r['charge_us'] for r in lat['rows']}['probe_update']
        tp = T_AIR_US + bk
        ts = (M_BEAMS - K_VISION) * tp
        for comp in HEAVY:
            charged = {"eager, Phase-1 counts": atlas.get(f'{comp}|{prof}')}
            if prof == 'gpu-resident' and comp in full:
                charged["eager, full counts"] = full[comp]
            if comp in o2.get(prof, {}):
                charged["TorchScript, whole decision"] = o2[prof][comp]
            charged = {k: v for k, v in charged.items() if v is not None}
            net_only = o1.get(prof, {}).get(comp)
            tmin = min(charged.values())
            ratios = {k: v / ts for k, v in charged.items()}
            rmin = min(ratios.values())
            feas = [T for T in grid if T - K_VISION * tp > tmin]
            admit = {f"M{M}": {f"K{K}": {
                         ("measured" if air is None else f"air{air:g}"):
                             tmin < (M - K) * ((T_AIR_US if air is None else air) + bk)
                         for air in (None, *ADMIT_AIR)}
                     for K in ADMIT_K if K < M} for M in ADMIT_M}
            row = {"component": comp, "profile": prof, "t_p_us": tp, "tau_star_us": ts,
                   "charges_us": charged, "ratios": ratios, "ratio_min": rmin,
                   "network_only_us": net_only,
                   "excluded": tmin >= ts,
                   "excluded_if_network_only_included":
                       min([tmin] + ([net_only] if net_only else [])) >= ts,
                   "at_budget": abs(rmin - 1.0) <= AT_BUDGET_BAND,
                   "tp_star_us": tmin / (M_BEAMS - K_VISION),
                   "tp_star_air_us": tmin / (M_BEAMS - K_VISION) - bk,
                   "first_feasible_T_us": feas[0] if feas else None,
                   "admission": admit,
                   "admission_beyond_64_is_accounting_only": True}
            rows.append(row)
            models.setdefault(comp, []).append(row)
    summary = {c: {"excluded_profiles": [r['profile'] for r in rs if r['excluded']],
                   "every_profile": all(r['excluded'] for r in rs),
                   "at_budget_profiles": [r['profile'] for r in rs if r['at_budget']]}
               for c, rs in models.items()}
    return {"K": K_VISION, "M": M_BEAMS, "rows": rows, "by_model": summary}


def r6(path='aging.json'):
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return {"pending": True}
    A = json.load(open(p))
    speeds = sorted({r['speed_kmh'] for r in A['rows']})
    per = {}
    for v in speeds:
        rows = [r for r in A['rows'] if r['speed_kmh'] == v]
        fails = [{"site": r['site'], "method": m, **d} for r in rows
                 for m, d in r['dominance'].items() if not d['holds']]
        G = rows[0]['beam_aging']['T_us']
        ok_T = [T for i, T in enumerate(G)
                if all(r['beam_aging']['loss_runmax'][i] < LOSS_THR for r in rows)]
        per[f"{v:g}"] = {"dominance_holds": not fails, "failures": fails,
                         "block_fading_adequate":
                             all(r['beam_aging']['loss_max_to_10ms'] < LOSS_THR for r in rows),
                         "largest_delta_below_1pct_us": max(ok_T) if ok_T else None,
                         "n_sites": len(rows)}
    held = [v for v in speeds if v > 0]
    upto = None
    for v in held:
        if not per[f"{v:g}"]["dominance_holds"]:
            break
        upto = v
    return {"pending": False, "speeds_kmh": speeds, "per_speed": per,
            "dominance_holds_up_to_kmh": upto,
            "dominance_holds_at_every_speed": all(per[f"{v:g}"]["dominance_holds"] for v in held)}


def r7(CTL, NOC, blind_key):
    paths = {c: os.path.join(HERE, f'sweep_{c}.json') for c, _ in PREDQ}
    if not all(os.path.exists(p) for p in paths.values()):
        return {"pending": True}
    per = {}
    for c, q in PREDQ:
        sites = json.load(open(paths[c]))['sites']
        assert len(sites) == 6, c
        pays, _ = verdicts(sites, 'compute_pays', f'{CTL}|{NOC}')
        bl, _ = verdicts(sites, 'mechanism', blind_key)
        per[c] = {"q": q, "compute_pays_established": sum(v == 1 for v in pays),
                  "compute_pays_lost": sum(v == -1 for v in pays),
                  "aware_vs_blind_established": sum(v == 1 for v in bl),
                  "aware_vs_blind_lost": sum(v == -1 for v in bl)}
    return {"pending": False, "per_q": per,
            "predictor_limited": any(per[c]['compute_pays_established'] >= NEED
                                     for c, q in PREDQ if q <= Q_LIMITED_MAX),
            "awareness_depends_on_predictor": any(per[c]['aware_vs_blind_established'] >= NEED
                                                  for c, _ in PREDQ)}


def main():
    R, RES, HO = J('report.json'), J('results.json'), J('heldout.json')
    prot = J('protocol.json')
    tests = set(prot['splits']['test'])
    sites = [s for f in (RES, HO) for s in f['sites']]
    names = [s['site'] for s in sites]
    assert len(sites) == 6, names
    CTL, NOC = R['headline_controller'], R['nocompute_controller']
    suf = '-pi' if CTL.endswith('-pi') else ''
    m = lambda base: base + suf

    blind, c_blind = verdicts(sites, 'mechanism', f'{CTL}|{m("sched-costblind")}')
    pays, c_pays = verdicts(sites, 'compute_pays', f'{CTL}|{NOC}')
    dep, c_dep = verdicts(sites, 'deployable', CTL)
    rep1, c_rep1 = verdicts(sites, 'mechanism', f'{CTL}|{m("sched-rep1")}')
    onex, c_onex = verdicts(sites, 'mechanism', f'{CTL}|{m("sched-1exit")}')
    fcc, c_fcc = verdicts(sites, 'mechanism', f'{m("sched-fcc")}|{m("sched-anyr")}')
    cnnc, c_cnnc = verdicts(sites, 'mechanism', f'{m("sched-cnnc")}|{m("sched-anyr")}')

    both = [i for i in range(6) if blind[i] == 1 and pays[i] == 1]
    r1_parts = {
        "sites_beating_blind_and_nocompute": [names[i] for i in both],
        "count_ok": len(both) >= NEED,
        "includes_a_test_site": any(names[i] in tests for i in both),
        "loses_to_deployable_on_a_test_site": [names[i] for i in range(6)
                                               if names[i] in tests and dep[i] == -1],
    }
    r1 = (r1_parts['count_ok'] and r1_parts['includes_a_test_site']
          and not r1_parts['loses_to_deployable_on_a_test_site'])

    def count_rule(*vs):
        if any(v[0] is None for v in vs):
            return None, []
        ok = [names[i] for i in range(6) if all(v[i] == 1 for v in vs)]
        return len(ok) >= NEED, ok

    r2, r2_sites = count_rule(rep1, onex)
    r3_fcc, fcc_sites = count_rule(fcc)
    r3_cnnc, cnnc_sites = count_rule(cnnc)
    r3 = None if r3_fcc is None or r3_cnnc is None else (r3_fcc or r3_cnnc)

    est = sorted({c['estimator'] for c in c_blind + c_pays + c_dep if c})
    out = {
        "headline_controller": CTL, "nocompute_controller": NOC,
        "seeds": RES.get('seeds') or R.get('seeds'),
        "estimators_used": est,
        "need_sites": NEED, "test_sites": sorted(tests),
        "per_site": {n: {"beats_blind": blind[i], "beats_nocompute": pays[i],
                         "vs_deployable": dep[i], "beats_rep1": rep1[i],
                         "beats_1exit": onex[i], "fcc_beats_anyr": fcc[i],
                         "cnnc_beats_anyr": cnnc[i]}
                     for i, n in enumerate(names)},
        "R1": {"holds": r1, **r1_parts,
               "framing": "controller" if r1 else "accounting"},
        "R2": {"holds": r2, "sites": r2_sites,
               "wording": None if r2 is None else
               ("how deep to go" if r2 else "whether and when to invoke")},
        "R3": {"holds": r3, "fcc_sites": fcc_sites, "cnnc_sites": cnnc_sites},
        "R4": r4(),
        # R5-R7: wording rules only
        "R5": r5(),
        "R6": r6(),
        "R7": r7(CTL, NOC, f'{CTL}|{m("sched-costblind")}'),
    }
    dst = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'rules.json'
    json.dump(out, open(os.path.join(HERE, dst), 'w'), indent=1)
    print(f"estimators: {est}")
    print(f"R1 framing = {out['R1']['framing']}  (holds={r1}; "
          f"sites beating blind AND no-compute: {r1_parts['sites_beating_blind_and_nocompute']}; "
          f"test site among them: {r1_parts['includes_a_test_site']}; "
          f"loses to deployable on test: {r1_parts['loses_to_deployable_on_a_test_site']})")
    print(f"R2 = {r2}  sites {r2_sites}")
    print(f"R3 = {r3}  fcc {fcc_sites} cnnc {cnnc_sites}")
    for row in out['R4']:
        print(f"R4 {row['component']:18s} {row['profile']:13s} tau {row['tau_us']:10.1f} us  "
              f"zero-everywhere={row['zero_rate_at_every_deadline']!s:5s} "
              f"cannot-beat-sweep={row['cannot_beat_sweep_anywhere']!s:5s} "
              f"(x{row['margin_over_tau_star']} tau*)")
    for row in out['R5']['rows']:
        print(f"R5 {row['component']:18s} {row['profile']:13s} min x{row['ratio_min']:.4f} tau* "
              f"excluded={row['excluded']!s:5s} at_budget={row['at_budget']!s:5s} "
              f"t_p*={row['tp_star_us']:.1f} us")
    print(f"R6 = {out['R6']}" if out['R6'].get('pending') else
          f"R6 dominance up to {out['R6']['dominance_holds_up_to_kmh']} km/h; per speed "
          + str({v: (d['dominance_holds'], d['block_fading_adequate'],
                     d['largest_delta_below_1pct_us']) for v, d in out['R6']['per_speed'].items()}))
    print(f"R7 = {out['R7']}" if out['R7'].get('pending') else
          f"R7 predictor_limited={out['R7']['predictor_limited']} awareness_depends="
          f"{out['R7']['awareness_depends_on_predictor']} "
          + str({c: (d['compute_pays_established'], d['aware_vs_blind_established'])
                 for c, d in out['R7']['per_q'].items()}))
    return out


if __name__ == '__main__':
    main()
