#!/usr/bin/env python3
"""Exact finite-horizon DP for the probe / compute / commit schedule.

The closed forms in theory.py are statements about fixed strategies.  This
file solves the ADAPTIVE problem exactly, by backward induction, so the
structural claims can be checked rather than asserted.

Instance (an idealisation, stated so it can be argued with)
-----------------------------------------------------------
* M beams.  Per-beam SNR gamma ~ Exp(mean Gbar), i.i.d.  For a Rayleigh
  channel and a unitary DFT codebook the beam responses are themselves i.i.d.
  complex Gaussian, so exponential beam SNRs are the natural law.
* A probe costs t_p and reveals its beam's gamma exactly.  Pilot noise is
  omitted here: it blunts the estimates without touching the time accounting,
  which is what these claims are about.
* The predictor has n+1 exit levels.  Reaching level i has cumulative cost
  tau_i and makes each subsequent probe as good as the best of q_i random
  beams -- q_0 = 1 is probing in random order.  Draws are treated as i.i.d.
  (sampling with replacement), exact in the large-codebook limit.
* Committing at elapsed d with best measured gain m earns
  max(0, 1 - d/T) * log2(1 + m).

State (i, k, m): compute level, probes taken, best gain so far.  Elapsed time
d(i,k) = k t_p + tau_i + c_sel is DETERMINISTIC given (i,k) -- that is what
makes exact backward induction possible, and it is exactly the frozen-profile
convention G3 specifies.
"""

import numpy as np

COMMIT, PROBE, COMPUTE = 0, 1, 2


class Instance:
    def __init__(self, tau, q, T, t_p=20.0, c_sel=1.88, gbar=10.0,
                 m_max=160.0, n_grid=1200, k_max=64):
        self.tau = np.asarray(tau, float)      # cumulative us per exit level
        self.q = np.asarray(q, float)          # predictor quality per level
        self.T, self.t_p, self.c_sel = T, t_p, c_sel
        self.gbar, self.k_max = gbar, k_max
        self.n_lvl = len(tau)
        # grid over the best-so-far gain
        self.g = np.linspace(0.0, m_max, n_grid)
        self.payoff = np.log2(1.0 + self.g)
        # P(draw lands in cell j) at each level: F_i(x) = (1 - e^{-x/gbar})^{q_i}
        edges = np.concatenate([[0.0], 0.5 * (self.g[1:] + self.g[:-1]),
                                [np.inf]])
        self.pmf = np.empty((self.n_lvl, n_grid))
        for i, qi in enumerate(self.q):
            F = np.where(np.isfinite(edges), (1 - np.exp(-edges / gbar)) ** qi, 1.0)
            self.pmf[i] = np.diff(F)
        self.cdf = np.cumsum(self.pmf, axis=1)

    def d(self, i, k):
        return k * self.t_p + self.tau[i] + self.c_sel

    # ---------------------------------------------------------------- exact
    def solve(self):
        """Backward induction.  Returns V[i,k,:] and the argmax action."""
        n, K, N = self.n_lvl, self.k_max, len(self.g)
        V = np.zeros((n, K + 1, N))
        A = np.full((n, K + 1, N), COMMIT, np.int8)
        for i in range(n - 1, -1, -1):
            for k in range(K, -1, -1):
                u = max(0.0, 1.0 - self.d(i, k) / self.T)
                v_commit = u * self.payoff
                best, act = v_commit.copy(), np.full(N, COMMIT, np.int8)

                if k < K and self.d(i, k + 1) < self.T:
                    nxt = V[i, k + 1]
                    # E[V(max(m, gamma))] = V(m) F(m) + sum_{g>m} p(g) V(g)
                    tail = np.concatenate([np.cumsum(
                        (self.pmf[i] * nxt)[::-1])[::-1][1:], [0.0]])
                    v_probe = nxt * self.cdf[i] + tail
                    take = v_probe > best
                    best, act = np.where(take, v_probe, best), np.where(take, PROBE, act)

                if i + 1 < n and self.d(i + 1, k) < self.T:
                    v_comp = V[i + 1, k]
                    take = v_comp > best
                    best, act = np.where(take, v_comp, best), np.where(take, COMPUTE, act)

                V[i, k], A[i, k] = best, act
        return V, A

    # ------------------------------------------------- one-step lookahead
    def solve_osla(self):
        """Value of the one-step-lookahead policy, evaluated exactly.

        OSLA commits unless some single action, followed by immediate commit,
        beats committing now.  Its value is then computed by the same backward
        recursion, but with the policy FIXED to that rule -- so this is the
        true value of the myopic policy, not a second approximation.
        """
        n, K, N = self.n_lvl, self.k_max, len(self.g)
        V = np.zeros((n, K + 1, N))
        A = np.full((n, K + 1, N), COMMIT, np.int8)
        for i in range(n - 1, -1, -1):
            for k in range(K, -1, -1):
                u = max(0.0, 1.0 - self.d(i, k) / self.T)
                v_commit = u * self.payoff
                # one-step values
                cand = [(v_commit, COMMIT)]
                if k < K and self.d(i, k + 1) < self.T:
                    u2 = max(0.0, 1.0 - self.d(i, k + 1) / self.T)
                    tail = np.concatenate([np.cumsum(
                        (self.pmf[i] * self.payoff)[::-1])[::-1][1:], [0.0]])
                    cand.append((u2 * (self.payoff * self.cdf[i] + tail), PROBE))
                if i + 1 < n and self.d(i + 1, k) < self.T:
                    u2 = max(0.0, 1.0 - self.d(i + 1, k) / self.T)
                    cand.append((u2 * self.payoff, COMPUTE))
                stack = np.stack([c[0] for c in cand])
                pick = np.argmax(stack, axis=0)
                act = np.array([cand[p][1] for p in pick], np.int8)

                # now evaluate that fixed rule against the true continuation
                val = v_commit.copy()
                if k < K and self.d(i, k + 1) < self.T:
                    nxt = V[i, k + 1]
                    tail = np.concatenate([np.cumsum(
                        (self.pmf[i] * nxt)[::-1])[::-1][1:], [0.0]])
                    vp = nxt * self.cdf[i] + tail
                    val = np.where(act == PROBE, vp, val)
                if i + 1 < n and self.d(i + 1, k) < self.T:
                    val = np.where(act == COMPUTE, V[i + 1, k], val)
                V[i, k], A[i, k] = val, act
        return V, A
