#!/usr/bin/env python3
"""Post-hoc explanation of where the compute action stops paying.

Not a pre-specified claim: it is offered as an arithmetic check on an effect the
experiment found.

STATE.  K codewords have been probed, nothing has been computed, so the elapsed
time is K*t_p.  The two admissible continuations are: commit now to the best
MEASURED beam, worth eta_base = eta(topk_K); or pay the cumulative charge tau_e
of exit e and commit to the predicted beam, worth eta_e = (1+g)*eta_base.  So g
is a RELATIVE gain whose denominator is the measurement-only strategy at the
same K, and tau_e is the whole charge from a cold start, not an increment.

That cold-start charge is tau_fresh[e] -- ONE invocation from the feature vector
through blocks 1..e to head e.  It is NOT tau_restart[e], which is what the
controller pays to climb to exit e through e separate invocations, and it is not
the resumable chain tau_cumulative[e], which evaluates every head along the way.
A compute-and-commit strategy runs the network once.

BOUNDARY.  Computing pays when

    (T - K*t_p - tau_e)(1+g) > (T - K*t_p)      <=>   T > K*t_p + tau_e (1+g)/g.

Both the probe term and the exit's OWN charge matter: using tau_1 on every row
regardless of which exit the row selects, and dropping K*t_p entirely, would put
the boundary at 0.28 ms where it belongs near 0.66 ms.  Both quantities are
measured on the validation city, so the crossing is a prediction, not a fit.
"""
import json, os
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
PR = json.load(open(os.path.join(D, 'predictor_profile.json')))['methods']
lat = json.load(open(os.path.join(D, 'lat_cpu1t_pcore.json')))
TAU = lat['tau_fresh_us']
TAU1 = TAU['1']
CH = {r['component']: r['charge_us'] for r in lat['rows']}
T_P = 20.0 + CH['probe_update']
GRID = json.load(open(os.path.join(D, 'protocol.json')))['coherence_grid_us']

print(f"tau_1 = {TAU1:.3f} us, measured quiet "
      f"(recheck drift {lat['contention']['worst_recheck_drift']:.1%})\n")
print(f"{'K':>3} {'best exit':>10} {'tau_e us':>9} {'eta gain over topk':>19} "
      f"{'break-even T (ms)':>18}")
rows = []
for k in (2, 4, 8, 16, 32):
    base = PR[f'topk{k}']['eta']
    ex = [(PR[f'ee{e}k{k}']['eta'], e) for e in (1, 2, 3) if f'ee{e}k{k}' in PR]
    eta, e = max(ex)
    gain = eta / base - 1
    tau_e = TAU[str(e)]                      # the charge of the exit THIS row picks
    t_be = (k * T_P + tau_e * (1 + gain) / gain) / 1000 if gain > 0 else None
    rows.append((k, e, gain, t_be, tau_e))
    print(f"{k:3d} {('exit '+str(e)):>10} {tau_e:9.3f} {gain*100:+18.2f}% "
          f"{(f'{t_be:.2f}' if t_be else 'never'):>18}")
good = [r for r in rows if r[3]]
if good:
    lo = min(r[3] for r in good)
    print(f"\nThe cheapest break-even is {lo:.2f} ms.  Below roughly that deadline the "
          f"exit charge\nexceeds what the prediction is worth, and a FIXED "
          f"compute-and-commit strategy\nshould lose.  This is not a prediction of "
          f"where the adaptive controller's sign\nchange falls, and it does not land "
          f"there: see analyze.py for the deadlines at\nwhich the controller's losses "
          f"are Holm-significant.")
    print("\nThis is the same mechanism as Phase 2's operating window -- a lower edge "
          "set by\nwhat the compute costs, not by what it is worth -- but the formula "
          "there is for a\nFIXED strategy, while the controller may decline to "
          "compute.  That asymmetry\npredicts the controller loses LESS than a fixed "
          "compute-using strategy below the\nedge; it is not needed to explain the "
          "location of the edge itself.")
json.dump({"tau1_us": TAU1, "t_probe_us": round(T_P, 3),
           "boundary": "T > K*t_p + tau_e (1+g)/g",
           "gain_denominator": "eta of topk_K on the validation city",
           "rows": [{"K": k, "exit": e, "tau_e_us": te, "eta_gain": round(g, 5),
                     "probe_cost_us": round(k * T_P, 3),
                     "break_even_ms": (round(t, 3) if t else None)}
                    for k, e, g, t, te in rows]},
          open(os.path.join(D, 'window.json'), 'w'), indent=1)
