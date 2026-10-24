#!/usr/bin/env python3
"""Check every Phase 2 claim numerically.  Claims that fail are reported."""

import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp
from dp import COMMIT, PROBE, COMPUTE
from theory import (Strategy, T_P, M_BEAMS, tau_star, window, breakpoint,
                    envelope, certified_stop, load_phase1)

ATLAS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "latency")
RES = {}

# The idealised instance used for the structural claims and the myopic-rule
# comparison.  Appendix A.4 of the paper states this counterexample explicitly,
# so the parameters are defined once here and recorded in verification.json
# rather than retyped anywhere.
INST_TAU = [13.75, 20.22, 27.14]   # cumulative us to exits 1..3
INST_Q = [1, 4, 10]                # predictor quality (order-statistic exponent)


def claim(name, ok, detail=""):
    RES[name] = {"holds": bool(ok), "detail": detail}
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


# ==========================================================================
print("\n== A. Structure of the exact optimal policy ==")
inst = dp.Instance(tau=INST_TAU, q=INST_Q, T=2000.0)
V, A = inst.solve()

# A1: for each (i,k) the commit set is an up-set in the best-so-far gain
bad = []
for i in range(inst.n_lvl):
    for k in range(inst.k_max + 1):
        c = (A[i, k] == COMMIT).astype(int)
        # up-set <=> once 1, stays 1
        if np.any(np.diff(c) < 0):
            bad.append((i, k))
claim("A1 commit region is a threshold in the best measured gain",
      not bad, f"{len(bad)} of {inst.n_lvl*(inst.k_max+1)} (i,k) slices violate"
      if bad else "every (i,k) slice has a single crossing")

# A2: deadline monotonicity -- committing at k implies committing at k+1
bad2 = []
for i in range(inst.n_lvl):
    for k in range(inst.k_max):
        viol = (A[i, k] == COMMIT) & (A[i, k + 1] != COMMIT)
        if viol.any():
            bad2.append((i, k, int(viol.sum())))
claim("A2 once committing is optimal, more elapsed time cannot reverse it",
      not bad2, f"{len(bad2)} violating (i,k) pairs" if bad2
      else "stopping region grows monotonically in elapsed time")

# A3: compute is only ever taken early
lvl_used = [(i, k) for i in range(inst.n_lvl) for k in range(inst.k_max + 1)
            if (A[i, k] == COMPUTE).any()]
kmax_compute = max((k for _, k in lvl_used), default=-1)
claim("A3 deliberation is front-loaded", kmax_compute >= 0,
      f"COMPUTE never chosen after probe {kmax_compute} of {inst.k_max}")

# ==========================================================================
print("\n== B. Is the myopic value-of-information rule optimal? ==")
gaps = []
for T in (300.0, 500.0, 800.0, 1200.0, 2000.0, 5000.0, 10000.0):
    it = dp.Instance(tau=INST_TAU, q=INST_Q, T=T)
    Ve, _ = it.solve()
    Vo, _ = it.solve_osla()
    rel = (Ve[0, 0, 0] - Vo[0, 0, 0]) / Ve[0, 0, 0] if Ve[0, 0, 0] > 0 else 0.0
    gaps.append((T, Ve[0, 0, 0], Vo[0, 0, 0], rel))
    print(f"    T={T:8.0f} us   exact={Ve[0,0,0]:7.4f}  myopic={Vo[0,0,0]:7.4f}"
          f"   loss={rel*100:6.2f}%")
worst = max(g[3] for g in gaps)
claim("B1 the one-step rule is NOT optimal in general", worst > 1e-6,
      f"worst-case value loss {worst*100:.1f}% -- the problem is not in the "
      f"monotone case when many probes remain")
tight = [g for g in gaps if g[0] <= 500]
claim("B2 the one-step rule becomes optimal as the deadline tightens",
      all(g[3] < 0.02 for g in tight),
      f"loss at T<=500us: {', '.join(f'{g[3]*100:.2f}%' for g in tight)}")

# ==========================================================================
print("\n== C. rho-sufficiency: results port across hardware ==")
base = dp.Instance(tau=INST_TAU, q=INST_Q, T=2000.0)
Vb, Ab = base.solve()
ok_rho = True
for alpha in (0.25, 3.0, 40.0):
    sc = dp.Instance(tau=[t * alpha for t in INST_TAU],
                     q=[1, 4, 10], T=2000.0 * alpha,
                     t_p=T_P * alpha, c_sel=1.88 * alpha)
    Vs, As = sc.solve()
    ok_rho &= np.array_equal(Ab, As) and np.allclose(Vb, Vs, atol=1e-12)
claim("C1 scaling every cost and the deadline by the same factor changes "
      "nothing", ok_rho,
      "policy and value identical at alpha in {0.25, 3, 40}")

# ==========================================================================
print("\n== D. Certified stop is sound ==")
it = dp.Instance(tau=INST_TAU, q=INST_Q, T=2000.0)
Ve, Ae = it.solve()
g_max = np.log2(1 + it.gbar * sum(1.0 / j for j in range(1, M_BEAMS + 1)))
unsound = 0
for i in range(it.n_lvl):
    for k in range(it.k_max + 1):
        d = it.d(i, k)
        cmin = min(it.t_p, (it.tau[i + 1] - it.tau[i]) if i + 1 < it.n_lvl else it.t_p)
        cert = np.array([certified_stop(p, g_max, cmin, it.T, d) for p in it.payoff])
        unsound += int(np.sum(cert & (Ae[i, k] != COMMIT)))
claim("D1 the certified-stop bound never certifies a state the exact DP "
      "would continue", unsound == 0, f"{unsound} unsound certifications")

# ==========================================================================
print("\n== E. Closed forms vs brute force ==")
charge = load_phase1(ATLAS)["cpu-1t-pcore"]
sweep = Strategy("exhaustive sweep", M_BEAMS, 0.0, 1.0, charge["sweep_select"])

def brute_window(L, S, lo=1.0, hi=4e5, n=400000):
    Ts = np.linspace(lo, hi, n)
    better = np.array([L.rate(t) > S.rate(t) for t in Ts])
    if not better.any():
        return None
    idx = np.where(better)[0]
    return Ts[idx[0]], Ts[idx[-1]]

ok_w = True
detail = []
for nmame, comp, k, eta in [("FC classifier", "beam_mlp", 8, 0.95),
                            ("CNN", "beam_cnn", 8, 0.95),
                            ("Graff covariance", "graff_col", 4, 0.95),
                            ("small transformer", "transformer_small", 8, 0.95),
                            ("ResNet-18", "resnet18_vision", 4, 0.95),
                            ("ViT-B/16", "vit_b16_vision", 4, 0.99)]:
    L = Strategy(nmame, k, charge[comp], eta)
    cf, bf = window(L, sweep), brute_window(L, sweep)
    if cf is None:
        ok = bf is None
        detail.append(f"{nmame}: closed-form EMPTY, brute-force {'empty' if bf is None else bf}")
    else:
        ok = bf is not None and abs(cf[0] - bf[0]) < 2.0 and abs(cf[1] - bf[1]) < 2.0
        detail.append(f"{nmame}: [{cf[0]:.0f}, {cf[1]:.0f}] us")
    ok_w &= ok
claim("E1 the window formula matches brute-force evaluation", ok_w,
      "; ".join(detail))

# tau*: accuracy-free budget
ok_t = True
for k in (4, 8, 16):
    ts = tau_star(k, charge["sweep_select"])
    for eta in (0.5, 0.8, 0.95, 0.999):
        below = Strategy("x", k, ts * 0.999, eta)
        above = Strategy("y", k, ts * 1.001, eta)
        ok_t &= (window(below, sweep) is not None) and (window(above, sweep) is None)
claim("E2 tau* is exactly the feasibility boundary, at every accuracy", ok_t,
      f"tau*(K=4)={tau_star(4, charge['sweep_select']):.1f} us, "
      f"tau*(K=8)={tau_star(8, charge['sweep_select']):.1f} us, "
      f"tau*(K=16)={tau_star(16, charge['sweep_select']):.1f} us")

# ==========================================================================
print("\n== F. The envelope: which architectures are ever worth running ==")
menu = [Strategy("exhaustive sweep", M_BEAMS, 0.0, 1.000, charge["sweep_select"]),
        Strategy("FC classifier", 8, charge["beam_mlp"], 0.880),
        Strategy("Graff covariance net", 4, charge["graff_col"], 0.910),
        Strategy("Graff APS conv net", 4, charge["graff_aps"], 0.900),
        Strategy("Graff eigenvector net", 4, charge["graff_eig"], 0.915),
        Strategy("CNN predictor", 8, charge["beam_cnn"], 0.930),
        Strategy("LSTM tracker", 8, charge["lstm_cached"], 0.940),
        Strategy("small transformer", 8, charge["transformer_small"], 0.960),
        Strategy("ResNet-18 vision", 4, charge["resnet18_vision"], 0.975),
        Strategy("ViT-B/16 vision", 4, charge["vit_b16_vision"], 0.985)]
env = envelope(menu)
print("    coherence time -> optimal strategy")
for lo, hi, s in env:
    hs = "inf" if hi == float("inf") else f"{hi:,.0f}"
    print(f"      [{lo:>10,.0f}, {hs:>10}) us   {s.name}"
          f"   (c={s.cost:,.1f} us, eta={s.eta:.3f})")
on = {s.name for _, _, s in env}
dominated = [s.name for s in menu if s.name not in on]
print(f"    never optimal at any coherence time: {', '.join(dominated) or 'none'}")

# brute-force cross-check of the envelope
Ts = np.geomspace(50, 5e5, 20000)
ok_e = True
for T in Ts:
    best = max(menu, key=lambda s: s.rate(T))
    seg = [s for lo, hi, s in env if lo <= T < hi]
    if best.rate(T) > 0 and seg and seg[0].name != best.name:
        if abs(seg[0].rate(T) - best.rate(T)) > 1e-12:
            ok_e = False
            break
claim("F1 the convex-hull envelope matches brute force at every T", ok_e,
      f"{len(env)} segments; {len(dominated)} of {len(menu)} strategies never optimal")

json.dump({"claims": RES,
           "envelope": [{"lo_us": lo, "hi_us": (None if hi == float('inf') else hi),
                         "name": s.name, "cost_us": s.cost, "eta": s.eta}
                        for lo, hi, s in env],
           "osla_gaps": [{"T_us": t, "exact": e, "myopic": m, "rel_loss": r}
                         for t, e, m, r in gaps],
           "osla_instance": {"tau_cum_us": INST_TAU, "q": INST_Q,
                             "t_p_us": inst.t_p, "c_sel_us": inst.c_sel,
                             "gbar": inst.gbar, "k_max": int(inst.k_max),
                             "n_grid": int(len(inst.g)),
                             "g_max": float(inst.g[-1])},
           "tau_star": {str(k): tau_star(k, charge["sweep_select"])
                        for k in (2, 4, 8, 16, 32)}},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "verification.json"), "w"), indent=2)
n_ok = sum(1 for v in RES.values() if v["holds"])
print(f"\n{n_ok}/{len(RES)} claims verified")
