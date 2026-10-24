#!/usr/bin/env python3
"""Turn the raw runs into the phase's claims.

Selection rules, all resolved on the VALIDATION city and never on a test city:
  * for each controller ACTION SET, the estimation variant (off-policy only, or
    one round of policy iteration);
  * for each controller FAMILY (may spend compute / may not), the single
    representative used in the headline comparison.  Taking a maximum over
    variants on the test city instead would favour the larger family.

Two comparisons are pre-specified and get Holm-corrected bootstrap p-values
across the twelve deadlines: does the compute action pay, and does the
controller beat the fixed strategy a practitioner could actually have deployed.
The gate's own comparison, against the fixed strategy that turned out best on
the test city itself, is post-hoc by construction and is reported as such.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'budget'))
import theory

ACTION_SETS = {"sched": ("sched", "sched-pi"),
               "sched-noreorder": ("sched-noreorder", "sched-noreorder-pi"),
               "sched-nocompute": ("sched-nocompute", "sched-nocompute-pi"),
               "osla": ("osla", "osla-pi"),
               "sched-costblind": ("sched-costblind", "sched-costblind-pi"),
               "sched-1exit": ("sched-1exit", "sched-1exit-pi"),
               "blinkered": ("blinkered", "blinkered-pi")}
FAMILIES = {"with_compute": ("sched", "sched-pi", "sched-noreorder", "sched-noreorder-pi"),
            "without_compute": ("sched-nocompute", "sched-nocompute-pi")}
FIXED_PREFIXES = ('topk', 'sweep', 'hier', 'fc', 'cnn', 'ee')
ALPHA = 0.05
# `--suffix _3seed` reads validation_3seed.json / results_3seed.json and writes
# report_3seed.json: the pre-registered three-seed analysis on the same code.
SUFFIX = sys.argv[sys.argv.index('--suffix') + 1] if '--suffix' in sys.argv else ''


def curve(site, nm):
    return np.array([site['methods'][nm]['per_T'][str(T)]['mean'] for T in site['grid']])


def holm(ps, alpha=ALPHA):
    """Holm-Bonferroni: returns the rejection mask in the original order."""
    m = len(ps)
    order = np.argsort(ps)
    rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if ps[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej


def select(val):
    s = val['sites'][0]
    out = {}
    for name, cands in list(ACTION_SETS.items()) + list(FAMILIES.items()):
        sc = {c: float(curve(s, c).mean()) for c in cands}
        out[name] = {"chosen": max(sc, key=sc.get), "city": s['site'],
                     "validation_mean_goodput": {k: round(v, 5) for k, v in sc.items()}}

    # The compute-pays comparison must not confound the ACTION SET with the
    # ESTIMATION PROCEDURE.  Taking the argmax over each family independently
    # can pair an off-policy controller against an on-policy baseline, and then
    # the difference measures both changes at once.  So: choose the estimation
    # suffix on validation by which one makes the BASELINE strongest -- the
    # conservative direction, since the baseline is what the compute action has
    # to beat -- and then take the best compute-enabled controller using that
    # same suffix.
    pi = lambda c: c.endswith('-pi')
    best = {}
    for suf in (False, True):
        wo = [c for c in FAMILIES['without_compute'] if pi(c) == suf]
        wi = [c for c in FAMILIES['with_compute'] if pi(c) == suf]
        if wo and wi:
            best[suf] = (max(float(curve(s, c).mean()) for c in wo),
                         max(wi, key=lambda c: float(curve(s, c).mean())),
                         max(wo, key=lambda c: float(curve(s, c).mean())))
    suf = max(best, key=lambda k: best[k][0])
    out['with_compute'] = {"chosen": best[suf][1], "city": s['site'],
                           "matched_estimation_rule": (
                               "estimation suffix chosen by the strongest "
                               "baseline, then both arms take that suffix"),
                           "validation_mean_goodput":
                               out['with_compute']['validation_mean_goodput']}
    out['without_compute'] = {"chosen": best[suf][2], "city": s['site'],
                              "validation_mean_goodput":
                                  out['without_compute']['validation_mean_goodput']}
    return out


def crossings(site, a, b):
    g = np.array(site['grid'], float)
    d = curve(site, a) - curve(site, b)
    out = []
    for i in range(len(g) - 1):
        if d[i] == 0 or np.sign(d[i]) == np.sign(d[i + 1]) or d[i + 1] == 0:
            continue
        w = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]))
        out.append(round(float(np.exp(np.log(g[i]) + w * (np.log(g[i + 1]) - np.log(g[i])))), 1))
    return out


def summarise(rows, key='diff'):
    ps = [r['p'] for r in rows]
    rej = holm(ps)
    wins = [r['T_us'] for r, k in zip(rows, rej) if k and r[key] > 0]
    losses = [r['T_us'] for r, k in zip(rows, rej) if k and r[key] < 0]
    return {"n_wins_holm": len(wins), "wins_holm": wins,
            "n_losses_holm": len(losses), "losses_holm": losses,
            "n_wins_uncorrected": sum(1 for r in rows if r['ci95'][0] > 0),
            "n_losses_uncorrected": sum(1 for r in rows if r['ci95'][1] < 0),
            "rows": rows}


def main():
    val = json.load(open(os.path.join(HERE, f'validation{SUFFIX}.json')))
    res = json.load(open(os.path.join(HERE, f'results{SUFFIX}.json')))
    sel = select(val)
    gate_ctl = sel['sched']['chosen']                 # the DECLARED action set
    best_ctl = sel['with_compute']['chosen']          # best controller, picked on val
    noc_ctl = sel['without_compute']['chosen']
    rep = {"protocol_sha256": res['protocol_sha256'], "selection": sel,
           "grid_us": res['grid_us'], "gate_controller": gate_ctl,
           "headline_controller": best_ctl, "nocompute_controller": noc_ctl,
           "sites": []}

    for s in res['sites']:
        grid = s['grid']
        fixed = [m for m in s['methods'] if m.startswith(FIXED_PREFIXES) and m != 'oracle']
        env = np.stack([curve(s, m) for m in fixed])
        best_fixed = [fixed[i] for i in env.argmax(0)]
        gr = {r['T_us']: r for r in s['gate_rows'][gate_ctl]}
        dep = {r['T_us']: r for r in s['deployable_rows'][best_ctl]}
        pays_key = f"{best_ctl}|{noc_ctl}"
        pay = {r['T_us']: r for r in s['compute_pays_rows'][pays_key]}
        rows = []
        for j, T in enumerate(grid):
            rows.append({
                "T_us": T, "best_fixed": best_fixed[j],
                "best_fixed_mean": round(float(env[:, j].max()), 5),
                "sched": s['methods'][best_ctl]['per_T'][str(T)]['mean'],
                "sched_gate": s['methods'][gate_ctl]['per_T'][str(T)]['mean'],
                "sched_nocompute": s['methods'][noc_ctl]['per_T'][str(T)]['mean'],
                "sched_noreorder": s['methods'][sel['sched-noreorder']['chosen']
                                                ]['per_T'][str(T)]['mean'],
                "osla": s['methods'][sel['osla']['chosen']]['per_T'][str(T)]['mean'],
                "blinkered": s['methods'][sel['blinkered']['chosen']]
                             ['per_T'][str(T)]['mean'],
                "zhang": s['methods']['zhang-AHTBS+AHIBS']['per_T'][str(T)]['mean'],
                "bound_stopping": s['methods']['bound-stopping']['per_T'][str(T)]['mean'],
                "deployable": dep[T]['baseline'], "deployable_mean": dep[T]['baseline_mean'],
                "paired_diff": gr[T]['paired_diff'], "ci95": gr[T]['ci95'],
                "rel_to_oracle": gr[T]['rel_to_oracle']})
        allgate = {}
        for v, rws in s['gate_rows'].items():
            w = [r for r in rws if r['ci95'][0] > 0]
            lo = [r for r in rws if r['ci95'][1] < 0]
            allgate[v] = {"wins": [r['T_us'] for r in w], "n_wins": len(w),
                          "n_losses": len(lo),
                          "worst_rel_oracle": round(min(r['rel_to_oracle'] for r in rws), 5),
                          "mean_goodput": round(float(np.mean(
                              [r['sched_mean'] for r in rws])), 5)}
        pubs = ['sweep64', 'hier6', 'topk8', 'topk16', 'fc16', 'cnn16',
                'zhang-AHTBS+AHIBS']
        inv = [[a, b] for i, a in enumerate(pubs) for b in pubs[i + 1:]
               if (curve(s, a)[0] - curve(s, b)[0]) *
                  (curve(s, a)[-1] - curve(s, b)[-1]) < 0]
        # The prediction must not be fitted to the curves it predicts.  eta and
        # cost come from the VALIDATION city; the crossing they are compared
        # against is measured on this TEST city.  Reading both from `s` would make
        # the agreement follow from the rate model rather than test it.
        vmeth = val['sites'][0]['methods']
        sw = theory.Strategy('sweep64', 0, vmeth['sweep64']['cost_us'],
                             vmeth['sweep64']['eta'])
        tstar = []
        for m in ('topk8', 'topk16', 'fc16', 'cnn16', 'hier6', 'ee1k8'):
            L = theory.Strategy(m, 0, vmeth[m]['cost_us'], vmeth[m]['eta'])
            pred = theory.breakpoint(L, sw)
            meas = crossings(s, 'sweep64', m)
            tstar.append({"method": m, "cost_us": vmeth[m]['cost_us'],
                          "eta": vmeth[m]['eta'],
                          "eta_source": val['sites'][0]['site'],
                          "eta_test_site": s['methods'][m]['eta'],
                          "predicted_T_us": round(float(pred), 1) if pred else None,
                          "measured_T_us": meas,
                          "ratio": (round(meas[0] / pred, 3)
                                    if meas and pred and pred > 0 else None)})
        # same discipline as the crossover above: the menu's coordinates come
        # from the validation city, the best-per-deadline it is checked against
        # is measured here
        menu = [theory.Strategy(m, 0, vmeth[m]['cost_us'], vmeth[m]['eta'])
                for m in fixed if m in vmeth]
        hull = theory.envelope(menu)
        pred_best = [next((st.name for lo, hi, st in hull if lo <= T < hi), None)
                     for T in grid]
        agree = sum(1 for a, b in zip(pred_best, best_fixed) if a == b)
        rep['sites'].append({
            "site": s['site'], "n": s['n'], "n_blocks": s['n_blocks'],
            "oracle_se": s['oracle_se'],
            "strata_frac": s['strata_frac'], "strata_counts": s['strata_counts'],
            "rows": rows, "gate_all_variants": allgate,
            "gate": {"controller": gate_ctl,
                     "wins": [r['T_us'] for r in rows if r['ci95'][0] > 0],
                     "n_wins": sum(1 for r in rows if r['ci95'][0] > 0),
                     "n_losses": sum(1 for r in rows if r['ci95'][1] < 0),
                     "worst_loss_rel_oracle": round(min(r['rel_to_oracle']
                                                        for r in rows), 5),
                     "guard_ok": bool(min(r['rel_to_oracle'] for r in rows) > -0.01),
                     "post_hoc_baseline": True},
            "compute_pays": {"with": best_ctl, "without": noc_ctl,
                             "matched_estimation": (best_ctl.endswith('-pi') ==
                                                    noc_ctl.endswith('-pi')),
                             "mean_with": allgate[best_ctl]['mean_goodput'],
                             "mean_without": allgate[noc_ctl]['mean_goodput'],
                             "mean_gain_rel": round(
                                 allgate[best_ctl]['mean_goodput'] /
                                 allgate[noc_ctl]['mean_goodput'] - 1, 5),
                             **summarise([pay[T] for T in grid])},
            "vs_deployable": summarise([dep[T] for T in grid]),
            # sign flip between the grid's two ENDS, on a 7-method menu; named for
            # both so it cannot be confused with accounting.json's counts (10-method
            # menu, and the sign-change-anywhere count beside them).
            "rank_inversions_endpoint_7menu": inv,
            "order_at_250us": sorted(pubs, key=lambda m: -curve(s, m)[0]),
            "order_at_10ms": sorted(pubs, key=lambda m: -curve(s, m)[-1]),
            "t_star": tstar,
            "envelope": {"hull": [[round(lo, 1), (round(hi, 1) if np.isfinite(hi) else None),
                                   st.name, round(st.eta, 4), round(st.cost, 2)]
                                  for lo, hi, st in hull],
                         "predicted_best": pred_best, "measured_best": best_fixed,
                         "agreement": f"{agree}/{len(grid)}"},
            "allocation": s['allocation'], "diagnostics": s['diagnostics'],
            "osla_loss_rel": [round(1 - r['osla'] / r['sched'], 4) for r in rows],
            "blinkered_loss_rel": [round(1 - r['blinkered'] / r['sched'], 4)
                                   for r in rows],
            # Hay et al.'s ordering Q_myopic <= Q_blinkered must hold on the
            # selection surrogates; if it is violated on the EVALUATED policies
            # that is worth knowing, so record it rather than assume it
            "blinkered_beats_myopic": [bool(r['blinkered'] >= r['osla'])
                                       for r in rows]})

    # deployment cost of the controller itself
    # the solve this report's tables came from ('_rr' and '_rr_3seed' read
    # scheduler_rr.json; '' and '_3seed' read scheduler.json)
    sc = json.load(open(os.path.join(HERE, f'scheduler{SUFFIX.replace("_3seed", "")}.json')))
    rep['controller_cost'] = {
        "table_bytes_per_deadline": sc['seeds'][0]['table_bytes'],
        "deadlines": len(res['grid_us']),
        "total_table_bytes": sc['seeds'][0]['table_bytes'] * len(res['grid_us']),
        # the kappa-augmented table's consult, timed on it, when the tables are that kind
        "consult_us": sc.get('c_lookup_aug_us') or sc['c_lookup_us'],
        "note": ("one table per deadline; a deployment interpolating over a coarser "
                 "grid would hold fewer. The consult is charged on every decision.")}
    rep['multiplicity'] = {
        "gate_cis_computed": len(res['grid_us']) * len(res['sites']) *
                             len(res['sites'][0]['gate_rows']),
        "note": ("the gate table reports uncorrected 95% CIs over every variant and "
                 "deadline; the two PRE-SPECIFIED comparisons (compute pays, versus "
                 "the deployable fixed baseline) are Holm-corrected across the "
                 "twelve deadlines and use a single controller chosen on validation")}
    json.dump(rep, open(os.path.join(HERE, f'report{SUFFIX}.json'), 'w'), indent=1)
    print(f"gate controller      {gate_ctl}\nheadline controller  {best_ctl}"
          f"\nno-compute control   {noc_ctl}\n")
    for s in rep['sites']:
        cp, dp = s['compute_pays'], s['vs_deployable']
        print(f"== {s['site']}")
        print(f"   gate (post-hoc baseline): {s['gate']['n_wins']} wins, "
              f"{s['gate']['n_losses']} losses, worst {s['gate']['worst_loss_rel_oracle']*100:+.2f}%"
              f" of oracle -> guard {'OK' if s['gate']['guard_ok'] else 'FAIL'}")
        print(f"   compute pays ({cp['with']} vs {cp['without']}, estimation "
              f"{'MATCHED' if cp['matched_estimation'] else 'UNMATCHED -- confounded'}): "
              f"{cp['mean_gain_rel']*100:+.1f}% mean; Holm-significant at "
              f"{cp['n_wins_holm']}/{len(s['rows'])} deadlines, against at {cp['n_losses_holm']}")
        print(f"   vs deployable fixed: Holm-significant wins {dp['n_wins_holm']}, "
              f"losses {dp['n_losses_holm']} (uncorrected {dp['n_wins_uncorrected']}/"
              f"{dp['n_losses_uncorrected']})")
        print(f"   envelope {s['envelope']['agreement']}, "
              f"{len(s['rank_inversions_endpoint_7menu'])} endpoint rank inversions (7-method menu)")


if __name__ == '__main__':
    main()
