#!/usr/bin/env python3
"""Phase 2 closed forms for compute-aware beam acquisition.

THE MODEL
---------
One coherence interval of length T.  A codebook of M beams.  An agent spends
time on three things and only three: probing beams (t_p each), running its own
predictor (tau), and transmitting.  Committing at elapsed time d with a beam of
SNR gamma earns the net rate

    R = max(0, 1 - d/T) * log2(1 + gamma).

Write G = E[log2(1+gamma)] for the spectral efficiency a strategy achieves and
eta = G / G_max for its efficiency relative to an oracle that always picks the
true best beam.  A strategy is then just a pair (c, eta): what it costs in time
and what fraction of the oracle rate it delivers.  Its net rate is

    R(T) = (1 - c/T) * eta * G_max          for T > c,   0 otherwise,

so on the useful range T > c, ranking strategies by R is the same as ranking
them by

    g(T) = eta * (T - c)                    -- A LINE IN T.

Every result in this file is a statement about the upper envelope of a family
of lines.  That is the whole theory, and it is why the conclusions are clean.
"""

import json
import os
from dataclasses import dataclass

T_P = 20.0          # us per probe: 10 us pilot + 10 us feedback (G1)
M_BEAMS = 64


@dataclass(frozen=True)
class Strategy:
    name: str
    k_probe: int        # beams measured
    tau: float          # us of per-decision computation
    eta: float          # spectral efficiency relative to the oracle beam
    c_sel: float = 0.0  # selection/emit cost not already inside tau

    @property
    def cost(self) -> float:
        """Total deterministic overhead d before payload can start."""
        return self.k_probe * T_P + self.tau + self.c_sel

    def rate(self, T: float) -> float:
        """Net rate in units of G_max."""
        return max(0.0, 1.0 - self.cost / T) * self.eta

    def line(self, T: float) -> float:
        """eta * (T - c): order-equivalent to rate() wherever rate > 0."""
        return self.eta * (T - self.cost)


# --------------------------------------------------------------------------
# Result 1.  The compute budget, free of any accuracy assumption.
# --------------------------------------------------------------------------

def tau_star(k_probe: int, c_sel_sweep: float, m: int = M_BEAMS) -> float:
    """Largest per-decision compute a K-probe method may spend and still be
    worth using at SOME coherence time, whatever its accuracy.

    A method beats the sweep at some T if and only if it finishes sooner than
    the sweep (Prop. 1).  d_L < d_S is  K t_p + tau < M t_p + c_sel, hence

        tau < tau* = (M - K) t_p + c_sel.

    Note what is absent: eta.  A perfect predictor obeys the same bound.  The
    budget is exactly the probe time the method saves -- no more.
    """
    return (m - k_probe) * T_P + c_sel_sweep


# --------------------------------------------------------------------------
# Result 2.  The operating window.
# --------------------------------------------------------------------------

def window(learned: Strategy, sweep: Strategy):
    """Interval of coherence times on which `learned` strictly beats `sweep`.

    Prop. 2.  With d_L < d_S and eta < 1 the set {T : R_L(T) > R_S(T)} is the
    open interval

        ( d_L ,  (eta_S d_S - eta_L d_L) / (eta_S - eta_L) ).

    Below the lower edge the learned method cannot finish; above the upper edge
    the sweep's full gain wins because overhead has stopped mattering.  If
    d_L >= d_S the set is EMPTY at every T -- no accuracy rescues a method that
    is slower than measuring everything.
    """
    dl, ds = learned.cost, sweep.cost
    if dl >= ds:
        return None
    if learned.eta >= sweep.eta:
        return (dl, float("inf"))
    hi = (sweep.eta * ds - learned.eta * dl) / (sweep.eta - learned.eta)
    return (dl, hi) if hi > dl else None


def breakpoint(a: Strategy, b: Strategy):
    """Coherence time at which the optimum switches from `a` to `b`.

    Two lines eta(T - c) cross at T = (eta_b c_b - eta_a c_a)/(eta_b - eta_a).
    With eta_b > eta_a, b (slower, better) is preferred above this T.
    """
    if a.eta == b.eta:
        return None
    return (b.eta * b.cost - a.eta * a.cost) / (b.eta - a.eta)


# --------------------------------------------------------------------------
# Result 3.  Which strategies are ever worth running.
# --------------------------------------------------------------------------

def envelope(strats):
    """Upper envelope of the lines g_i(T) = eta_i (T - c_i), T > c_i.

    Prop. 3.  Strategy i is optimal for some coherence time iff its line is on
    the upper envelope, i.e. iff the point (slope, intercept) = (eta_i,
    -eta_i c_i) is a vertex of the upper convex hull of those points.  Returns
    the envelope as [(T_lo, T_hi, strategy)] in increasing T, which is also the
    order of increasing eta.

    The practical reading: a menu of candidate architectures reduces to the
    hull vertices.  Everything inside the hull is dominated at every deadline
    and never needs to be trained, tuned or reported.
    """
    pts = sorted(strats, key=lambda s: (s.eta, -s.cost))
    # Keep, for each eta, only the cheapest; a slower method with the same
    # accuracy is dominated everywhere.
    best = {}
    for s in pts:
        if s.eta not in best or s.cost < best[s.eta].cost:
            best[s.eta] = s
    cand = sorted(best.values(), key=lambda s: s.eta)

    hull = []
    for s in cand:
        while hull:
            if s.cost <= hull[-1].cost:
                hull.pop()              # cheaper AND better: dominates
                continue
            if len(hull) >= 2:
                if breakpoint(hull[-2], hull[-1]) >= breakpoint(hull[-1], s):
                    hull.pop()          # never on top: skipped over
                    continue
            break
        hull.append(s)

    out, lo = [], 0.0
    for i, s in enumerate(hull):
        lo_i = max(lo, s.cost)
        hi_i = breakpoint(s, hull[i + 1]) if i + 1 < len(hull) else float("inf")
        if hi_i > lo_i:
            out.append((lo_i, hi_i, s))
            lo = hi_i
    return out


# --------------------------------------------------------------------------
# Result 4.  Adaptive scheduling: the one-step rule and when it is optimal.
# --------------------------------------------------------------------------

def osla_continue(delta_G, G_now, cost, T, d):
    """One-step-lookahead test for spending `cost` more time.

    Committing now earns (T-d) G_now.  Taking the action and then committing
    earns (T-d-cost)(G_now + delta_G).  Continuing wins iff

        (T-d-cost)(G_now + delta_G) > (T-d) G_now
        <=>   delta_G / (G_now + delta_G)  >  cost / (T - d)
        <=>   lambda                       >  rho_remaining

    Left: the RELATIVE value of the information the action buys.  Right: the
    action's cost as a fraction of the time still left.  Both dimensionless.
    """
    if T - d <= 0:
        return False, 0.0, float("inf")
    lam = delta_G / (G_now + delta_G) if (G_now + delta_G) > 0 else 0.0
    rho = cost / (T - d)
    return lam > rho, lam, rho


def certified_stop(G_now, G_max, cost_min, T, d):
    """Sufficient condition for committing now being exactly optimal.

    Any continuation costs at least cost_min and cannot do better than the
    oracle, so it is bounded by (T - d - cost_min) G_max.  Hence commit now is
    optimal whenever

        (T - d) G_now >= (T - d - cost_min) G_max.

    Unlike the one-step rule this needs no assumption at all; it is a bound on
    the exact value function.
    """
    u = T - d
    return u <= 0 or u * G_now >= (u - cost_min) * G_max


# --------------------------------------------------------------------------

def load_phase1(atlas_dir):
    """Measured conservative per-decision charges from the Phase 1 atlas.

    The optimised column must trace the WHOLE decision, ending in a beam index.
    Earlier revisions traced only the network, which silently dropped the
    selection step and understated every compiled charge; those files (o_*) are
    ignored here in favour of the full-decision ones (o2_*).
    """
    prof, opt = {}, {}
    for f in os.listdir(atlas_dir):
        p = os.path.join(atlas_dir, f)
        if f.startswith("p_") and f.endswith(".json"):
            j = json.load(open(p))
            prof[j["env"]["profile"]] = {r["component"]: r["charge_us"]
                                         for r in j["rows"]}
        elif f.startswith("o2_") and f.endswith(".json"):
            j = json.load(open(p))
            opt[j["env"]["profile"]] = {r["component"]: r["charge_us"]
                                        for r in j["rows"]}
    alias = {"s1_q_nets": "s1_q"}
    cons = {}
    for pn, d in prof.items():
        o = {alias.get(k, k): v for k, v in opt.get(pn, {}).items()}
        cons[pn] = {k: (min(v, o[k]) if k in o else v) for k, v in d.items()}
    return cons
