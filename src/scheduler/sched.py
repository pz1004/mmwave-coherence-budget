#!/usr/bin/env python3
"""The deadline-aware scheduler: empirical DP on the FIT city only.

The Phase 2 DP assumed i.i.d. exponential beam SNRs and a predictor worth
"best of q_i random beams".  Real beams are neither, so here the DP's
primitives are ESTIMATED from the fit city instead of assumed:

  state    (i, k, b, fresh)  exit level, probes taken, best-measured-SNR
                             bucket, and whether the predictor's output is
                             still unconsumed by a later probe
  actions  probe (t_p) / compute deeper (tau_{i+1}-tau_i) / commit to the best
           measured beam / commit to the predictor's beam (fresh states only)
  payoff   u(i,k) * g(i,k,b) with u = max(0, 1 - d(i,k)/T)

Exploration is a uniformly random policy over the available actions, 8
replicates per seed, so every reachable state is visited rather than only the
states some particular policy likes.  The Markov assumption on (i,k,b,fresh)
is an approximation IN THE CONTROLLER -- the evaluation simulates the true
process, including the identity of every probed beam.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs, feats, sim
from sim import (DEV, K_MAX, N_EXIT, NB, NC, COMMIT_M, COMMIT_P, PROBE, COMPUTE)

FIT = 'city_0_newyork_28'
import protocol
SEEDS = protocol.seed_arg()   # the ten seeds in force; --seeds overrides it
REPLICATES = 24
ON_REPS = 10
EPS = 0.15
MIN_N = 20
NEG = -1e30


def explore(site, model, seed, tau, table=None, eps=0.15, reps=None,
            reorder=True, max_exit=N_EXIT, spec=None, nets=None, rank_exact=False):
    """Rollouts, returning the estimated DP primitives.

    table=None      uniform-random policy -- broad coverage, off-policy
    table=A         epsilon-greedy on A -- accurate where THAT policy goes

    Both are needed.  A model fitted only off-policy is systematically wrong on
    the states the policy actually reaches (it over-values the compute
    action, which is the whole object of this phase); a model fitted only
    on-policy has no data anywhere else and cannot be improved.  primitives()
    prefers the on-policy estimate wherever it has enough samples and backs off
    to the off-policy one otherwise.

    spec (sim.Spec) replaces the anytime predictor's level+1 head with another
    compute action -- a fixed exit, or a per-K FC/CNN network legal only at the
    probe counts it exists for.  spec=None is the headline controller, unchanged.

    rank_exact: `table` is kappa-major (solve_exact), so the on-policy passes
    track each user's probe count at its first inference call and consult that block; the
    random draws are the same calls in the same order either way."""
    kset = None if spec is None or spec.ks is None else \
        torch.tensor(spec.ks, device=DEV, dtype=torch.long)
    n = site.n
    SH = (N_EXIT + 1, K_MAX + 1, NB, NC)
    cnt = torch.zeros(SH, dtype=torch.float64, device=DEV)
    trans = torch.zeros((*SH, NB, NC), dtype=torch.float32, device=DEV)
    sm = torch.zeros(SH, dtype=torch.float64, device=DEV)
    sp = torch.zeros(SH, dtype=torch.float64, device=DEV)
    cp = torch.zeros(SH, dtype=torch.float64, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(seed * 7919 + 13)
    sprd = sim.spread_score()
    one = torch.ones(n, dtype=torch.float64, device=DEV)
    one32 = torch.ones(n, dtype=torch.float32, device=DEV)

    tab = None if table is None else torch.tensor(table, device=DEV, dtype=torch.long)
    for rep in range(REPLICATES if reps is None else reps):
        y = sim.measure(site.pn, gen)
        snr = feats.snr_db_t(y)
        probed = torch.zeros((n, 64), dtype=torch.bool, device=DEV)
        score = sprd.expand(n, -1).clone()
        lvl = torch.zeros(n, dtype=torch.long, device=DEV)
        k = torch.zeros(n, dtype=torch.long, device=DEV)
        rec = torch.zeros(n, dtype=torch.long, device=DEV)
        fresh = torch.zeros(n, dtype=torch.long, device=DEV)
        bidx = torch.zeros(n, dtype=torch.long, device=DEV)
        bsnr = torch.full((n,), -1e9, device=DEV, dtype=torch.float64)
        msum = torch.zeros(n, device=DEV, dtype=torch.float64)
        alive = torch.ones(n, dtype=torch.bool, device=DEV)
        kap = torch.zeros(n, dtype=torch.long, device=DEV)      # probes at the first call
        for step in range(K_MAX + N_EXIT):
            any_p = k > 0
            b = sim.bucket(bsnr, any_p)
            c = sim.level_class(msum / k.clamp(min=1), any_p)
            se_m = site.sn.gather(1, bidx[:, None])[:, 0].double()
            idx = (lvl[alive], k[alive], b[alive], c[alive])
            cnt.index_put_(idx, one[:int(alive.sum())], accumulate=True)
            sm.index_put_(idx, se_m[alive], accumulate=True)

            can_probe = alive & (k < K_MAX)
            can_comp = alive & (lvl < max_exit)
            if kset is not None:
                can_comp = can_comp & torch.isin(k, kset)
            r = torch.rand(n, generator=gen, device=DEV)
            if tab is None:
                do_comp = can_comp & ((r < 0.5) | ~can_probe)
                do_probe = can_probe & ~do_comp
            else:
                act = tab[kap, lvl, k, b, c, fresh] if rank_exact else tab[lvl, k, b, c, fresh]
                explore_now = torch.rand(n, generator=gen, device=DEV) < eps
                do_comp = can_comp & torch.where(explore_now, (r < 0.5) | ~can_probe,
                                                 act == COMPUTE)
                do_probe = can_probe & ~do_comp & torch.where(
                    explore_now, torch.ones_like(alive), act == PROBE)
                alive = alive & (do_comp | do_probe)      # a commit ends the run
            if do_comp.any():
                f, _ = feats.build(y, probed)
                for e in range(N_EXIT):
                    sel = do_comp & (lvl == e)
                    if not sel.any():
                        continue
                    if spec is None:
                        lg = sim.exit_logits(model, f[sel], e + 1)
                    else:
                        lg = sim.compute_logits(spec, model, nets, y, probed, f, sel, e, k)
                    if reorder:
                        score[sel] = lg
                    rc = lg.argmax(1)
                    rec[sel] = rc
                    se_p = site.sn[sel].gather(1, rc[:, None])[:, 0].double()
                    j = (torch.full_like(rc, e + 1), k[sel], b[sel], c[sel])
                    sp.index_put_(j, se_p, accumulate=True)
                    cp.index_put_(j, one[:int(sel.sum())], accumulate=True)
                kap = torch.where(do_comp & (lvl == 0), k, kap)
                lvl = lvl + do_comp.long()
                fresh = torch.where(do_comp, torch.ones_like(fresh), fresh)
            if do_probe.any():
                nxt = torch.where(probed, torch.full_like(score, NEG), score).argmax(1)
                probed[do_probe, nxt[do_probe]] = True
                ns = snr.gather(1, nxt[:, None])[:, 0]
                better = do_probe & (ns > bsnr)
                bidx = torch.where(better, nxt, bidx)
                bsnr = torch.where(better, ns, bsnr)
                msum = msum + torch.where(do_probe, ns, torch.zeros_like(ns))
                k2 = k + do_probe.long()
                nb2 = sim.bucket(bsnr, k2 > 0)
                nc2 = sim.level_class(msum / k2.clamp(min=1), k2 > 0)
                jj = (lvl[do_probe], k[do_probe], b[do_probe], c[do_probe],
                      nb2[do_probe], nc2[do_probe])
                trans.index_put_(jj, one32[:int(do_probe.sum())], accumulate=True)
                k = k2
                fresh = torch.where(do_probe, torch.zeros_like(fresh), fresh)
            alive = alive & (can_probe | can_comp)
            if not alive.any():
                break
    return cnt, trans, sm, sp, cp


def primitives(stats, off=None):
    """Means with a declared backoff: the on-policy estimate wherever it has at
    least MIN_N samples, else the off-policy (uniform-exploration) estimate,
    else the analytic bucket-centre value log2(1 + gamma_b)."""
    cnt, trans, sm, sp, cp = stats
    cc = sim.bucket_centre_snr()
    fb = torch.tensor(np.log2(1 + 10 ** (cc / 10)), device=DEV, dtype=torch.float64)
    fb = fb[None, None, :, None].expand_as(sm)
    eye = torch.eye(NB * NC, dtype=torch.float32, device=DEV).reshape(NB, NC, NB, NC)

    def mean(num, den, base):
        return torch.where(den < MIN_N, base, num / den.clamp(min=1))

    def rows(tr, base):
        tot = tr.sum((-2, -1), keepdim=True)
        return torch.where(tot < MIN_N, base, tr / tot.clamp(min=1)), tot

    if off is None:
        g_m = mean(sm, cnt, fb)
        g_p = mean(sp, cp, g_m)
        P, tot = rows(trans, eye[None, None])
    else:
        o_cnt, o_trans, o_sm, o_sp, o_cp = off
        g_m0 = mean(o_sm, o_cnt, fb)
        g_p0 = mean(o_sp, o_cp, g_m0)
        P0, _ = rows(o_trans, eye[None, None])
        g_m = mean(sm, cnt, g_m0)
        g_p = mean(sp, cp, g_p0)
        P, tot = rows(trans, P0)
    return (g_m, g_p, P,
            {"thin_gm": int((cnt < MIN_N).sum()), "thin_gp": int((cp < MIN_N).sum()),
             "thin_trans": int((tot < MIN_N).sum()),
             "cells": int(cnt.numel()), "visits": float(cnt.sum()),
             "reachable_cells": int((cnt > 0).sum()),
             "visit_frac_in_thin_gm":
                 round(float(cnt[cnt < MIN_N].sum() / cnt.sum().clamp(min=1)), 6),
             "visit_frac_in_thin_trans":
                 round(float(cnt[(tot < MIN_N)[..., 0, 0]].sum() / cnt.sum().clamp(min=1)), 6)})


def _probe_value(P_ik, V_next):
    """E[V(b', c') | b, c] for one probe: the joint transition applied to the
    whole next-state value surface.  V_next is (NB, NC)."""
    return (P_ik.reshape(NB * NC, NB * NC).double()
            @ V_next.reshape(-1)).reshape(NB, NC)


def solve(g_m, g_p, P, tau, T, t_probe, c_lookup, allow_compute=True,
          max_exit=N_EXIT, compute_ks=None, c_rank=0.0):
    """Exact backward induction over (i, k, b, c, fresh).

    allow_compute=False restricts the action set to probe/commit, which is
    the ablation that separates adaptive probe stopping from the compute
    dimension the phase exists to test.

    max_exit caps how deep the compute action may go.  max_exit=1 is the
    cheapest-exit-only controller: it may still decide WHETHER and WHEN to
    compute, but not HOW FAR, which separates selective invocation from depth
    allocation."""
    V = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC, 2), dtype=torch.float64, device=DEV)
    A = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC, 2), dtype=torch.int8, device=DEV)
    t_cm = torch.tensor(COMMIT_M, dtype=torch.int8, device=DEV)
    t_cp = torch.tensor(COMMIT_P, dtype=torch.int8, device=DEV)
    t_pr = torch.tensor(PROBE, dtype=torch.int8, device=DEV)
    t_co = torch.tensor(COMPUTE, dtype=torch.int8, device=DEV)
    for i in range(N_EXIT, -1, -1):
        for k in range(K_MAX, -1, -1):
            d = sim.elapsed_us(k, i, tau, t_probe, c_lookup, c_rank)
            u = max(0.0, 1.0 - d / T)
            vm = u * g_m[i, k]
            vpr = None
            if k < K_MAX and sim.elapsed_us(k + 1, i, tau, t_probe, c_lookup, c_rank) < T:
                vpr = _probe_value(P[i, k], V[i, k + 1, :, :, 0])
            for fresh in (1, 0):
                v = vm.clone()
                a = torch.full_like(v, COMMIT_M, dtype=torch.int8)
                if fresh and i >= 1:
                    vp = u * g_p[i, k]
                    t = vp > v
                    v, a = torch.where(t, vp, v), torch.where(t, t_cp, a)
                if vpr is not None:
                    t = vpr > v
                    v, a = torch.where(t, vpr, v), torch.where(t, t_pr, a)
                if (allow_compute and i < max_exit
                        and (compute_ks is None or k in compute_ks)
                        and sim.elapsed_us(k, i + 1, tau, t_probe, c_lookup, c_rank) < T):
                    vc = V[i + 1, k, :, :, 1]
                    t = vc > v
                    v, a = torch.where(t, vc, v), torch.where(t, t_co, a)
                V[i, k, :, :, fresh], A[i, k, :, :, fresh] = v, a
    return V, A


def solve_blinkered(g_m, g_p, P, tau, T, t_probe, c_lookup, c_rank=0.0):
    """Hay et al.'s blinkered construction, with the computation classes
    partitioned by ACTION TYPE rather than by target action.

    The myopic rule assumes at most one further action before committing, which
    is exactly why it stops too early.  A blinkered policy allows arbitrarily
    many but looks in only one direction at a time: it values PROBE by the best
    pure probe chain (never computing in between) and COMPUTE by the best pure
    compute chain (never probing in between), and takes the best of those
    against committing now.  The myopic option set is the chain-length-one
    subset of this one, so on the selection surrogates
    Q_myopic <= Q_blinkered <= Q_optimal.

    Hay et al. state this for metalevel problems with INDEPENDENT ACTIONS,
    which this problem does not have -- probing and computing both inform the
    same beam choice.  That condition is needed for their decomposition
    theorem, not for the definition, and the ordering above follows from set
    inclusion alone.  As in solve_osla the surrogates only CHOOSE the action;
    the true continuation is backed up, so this is policy evaluation of a
    blinkered controller under the same model the exact DP uses."""
    V = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC, 2), dtype=torch.float64, device=DEV)
    A = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC, 2), dtype=torch.int8, device=DEV)
    # Pure-chain values.  Neither carries a `fresh` axis, and that is not an
    # omission: a probe clears fresh, so a probe chain always terminates on the
    # measured beam; a compute sets it, so a compute chain may always terminate
    # on either.  Both fill inside the existing sweep -- Wp reads k+1 (previous
    # inner step), Wc reads i+1 (previous outer step).
    Wp = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC), dtype=torch.float64, device=DEV)
    Wc = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC), dtype=torch.float64, device=DEV)
    for i in range(N_EXIT, -1, -1):
        for k in range(K_MAX, -1, -1):
            d = sim.elapsed_us(k, i, tau, t_probe, c_lookup, c_rank)
            u = max(0.0, 1.0 - d / T)
            vm = u * g_m[i, k]
            ok_pr = (k < K_MAX
                     and sim.elapsed_us(k + 1, i, tau, t_probe, c_lookup, c_rank) < T)
            ok_co = (i < N_EXIT
                     and sim.elapsed_us(k, i + 1, tau, t_probe, c_lookup, c_rank) < T)
            # value of probing at least once and then staying in the probe chain
            chain_p = _probe_value(P[i, k], Wp[i, k + 1]) if ok_pr else None
            Wp[i, k] = torch.maximum(vm, chain_p) if ok_pr else vm
            # at level i having just computed, both commits are available
            here = u * torch.maximum(g_m[i, k], g_p[i, k]) if i >= 1 else vm
            Wc[i, k] = torch.maximum(here, Wc[i + 1, k]) if ok_co else here
            true_p = (_probe_value(P[i, k], V[i, k + 1, :, :, 0])
                      if ok_pr else None)
            for fresh in (1, 0):
                cand = [(vm, COMMIT_M)]
                if fresh and i >= 1:
                    cand.append((u * g_p[i, k], COMMIT_P))
                if ok_pr:
                    cand.append((chain_p, PROBE))
                if ok_co:
                    cand.append((Wc[i + 1, k], COMPUTE))
                pick = torch.stack([c[0] for c in cand]).argmax(0)
                lut = torch.tensor([c[1] for c in cand], dtype=torch.int8, device=DEV)
                a = lut[pick]
                v = vm.clone()
                if fresh and i >= 1:
                    v = torch.where(a == COMMIT_P, u * g_p[i, k], v)
                if true_p is not None:
                    v = torch.where(a == PROBE, true_p, v)
                if ok_co:
                    v = torch.where(a == COMPUTE, V[i + 1, k, :, :, 1], v)
                V[i, k, :, :, fresh], A[i, k, :, :, fresh] = v, a
    return V, A


def solve_osla(g_m, g_p, P, tau, T, t_probe, c_lookup, c_rank=0.0):
    """The myopic value-of-computation rule metareasoning prescribes: take an
    action only if that action followed by an immediate commit beats committing
    now.  Evaluated against the TRUE continuation, as Phase 2 does."""
    V = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC, 2), dtype=torch.float64, device=DEV)
    A = torch.zeros((N_EXIT + 1, K_MAX + 1, NB, NC, 2), dtype=torch.int8, device=DEV)
    for i in range(N_EXIT, -1, -1):
        for k in range(K_MAX, -1, -1):
            d = sim.elapsed_us(k, i, tau, t_probe, c_lookup, c_rank)
            u = max(0.0, 1.0 - d / T)
            vm = u * g_m[i, k]
            d2 = (sim.elapsed_us(k + 1, i, tau, t_probe, c_lookup, c_rank)
                  if k < K_MAX else float('inf'))
            d3 = (sim.elapsed_us(k, i + 1, tau, t_probe, c_lookup, c_rank)
                  if i < N_EXIT else float('inf'))
            u2, u3 = max(0.0, 1 - d2 / T), max(0.0, 1 - d3 / T)
            myopic = (u2 * _probe_value(P[i, k], g_m[i, k + 1]) if d2 < T else None)
            true_p = (_probe_value(P[i, k], V[i, k + 1, :, :, 0]) if d2 < T else None)
            for fresh in (1, 0):
                cand = [(vm, COMMIT_M)]
                if fresh and i >= 1:
                    cand.append((u * g_p[i, k], COMMIT_P))
                if myopic is not None:
                    cand.append((myopic, PROBE))
                if d3 < T:
                    cand.append((u3 * torch.maximum(g_m[i + 1, k], g_p[i + 1, k]), COMPUTE))
                pick = torch.stack([c[0] for c in cand]).argmax(0)
                lut = torch.tensor([c[1] for c in cand], dtype=torch.int8, device=DEV)
                a = lut[pick]
                v = vm.clone()
                if fresh and i >= 1:
                    v = torch.where(a == COMMIT_P, u * g_p[i, k], v)
                if true_p is not None:
                    v = torch.where(a == PROBE, true_p, v)
                if d3 < T:
                    v = torch.where(a == COMPUTE, V[i + 1, k, :, :, 1], v)
                V[i, k, :, :, fresh], A[i, k, :, :, fresh] = v, a
    return V, A


# ------------------------------------------------- the exact ranking count -----
# After its first inference call the controller ranks each next codeword by a masked argmax
# over the exit's scores; before it, it probes in the fixed order and ranks nothing.  The
# ranking operations a decision sequence performs are therefore k - kappa, with kappa the
# probe count at the first call, whereas sim.elapsed_us's 1[i >= 1] k c_rank charges the kappa
# probes before the call too.  The solvers below carry kappa in the state for i >= 1, all 65
# values at once, so each pass is still 4 x 65 steps and exact for the estimated model.  V and
# A are indexed [i, kappa, k, b, c, fresh]; the deployed table is returned kappa-major,
# [kappa, i, k, b, c, fresh]: 65 blocks of the original shape, the controller consulting
# block 0 until its first call and block kappa after it.  At i = 0 every block holds the same
# actions, and blocks with kappa > k, which no sequence reaches, hold COMMIT_M.
K1 = K_MAX + 1


def _kappas():
    return torch.arange(K1, dtype=torch.float64, device=DEV)


def _probe_values(P_ik, Vn):
    """_probe_value for every kappa at once; Vn is (K1, NB, NC)."""
    return (P_ik.reshape(NB * NC, NB * NC).double()
            @ Vn.reshape(K1, NB * NC).T).T.reshape(K1, NB, NC)


def _exact_d(k, i, tau, t_probe, c_lookup, c_rank, kap):
    """(K1,) elapsed times at (i, k), one per first-call probe count."""
    return sim.elapsed_exact_us(k, i, tau, t_probe, c_lookup, c_rank, kap)


def _first_call_d(k, tau, t_probe, c_lookup, c_rank):
    """Elapsed time right after the first call, at k probes: kappa = k, nothing ranked."""
    return float(sim.elapsed_exact_us(k, 1, tau, t_probe, c_lookup, c_rank, k))


def _kappa_major(A):
    unreach = (torch.arange(K1, device=DEV)[:, None] > torch.arange(K1, device=DEV)[None, :])
    A[1:, unreach] = COMMIT_M
    return A.permute(1, 0, 2, 3, 4, 5).contiguous()


def solve_exact(g_m, g_p, P, tau, T, t_probe, c_lookup, c_rank, allow_compute=True,
                max_exit=N_EXIT, compute_ks=None):
    """solve() with the ranking time counted exactly.  Same action set, tie order and
    model; the compute action from i = 0 at k probes lands in (1, kappa = k, k)."""
    kap = _kappas()
    V = torch.zeros((N_EXIT + 1, K1, K1, NB, NC, 2), dtype=torch.float64, device=DEV)
    A = torch.zeros((N_EXIT + 1, K1, K1, NB, NC, 2), dtype=torch.int8, device=DEV)
    t_cp = torch.tensor(COMMIT_P, dtype=torch.int8, device=DEV)
    t_pr = torch.tensor(PROBE, dtype=torch.int8, device=DEV)
    t_co = torch.tensor(COMPUTE, dtype=torch.int8, device=DEV)
    for i in range(N_EXIT, -1, -1):
        for k in range(K_MAX, -1, -1):
            d = _exact_d(k, i, tau, t_probe, c_lookup, c_rank, kap)
            u = torch.clamp(1.0 - d / T, min=0.0)
            vm = u[:, None, None] * g_m[i, k][None]
            vpr = okp = None
            if k < K_MAX:
                okp = (_exact_d(k + 1, i, tau, t_probe, c_lookup, c_rank, kap) < T)[:, None, None]
                vpr = _probe_values(P[i, k], V[i, :, k + 1, :, :, 0])
            vc = okc = None
            if allow_compute and i < max_exit and (compute_ks is None or k in compute_ks):
                if i == 0:
                    if _first_call_d(k, tau, t_probe, c_lookup, c_rank) < T:
                        vc = V[1, k, k, :, :, 1][None].expand(K1, NB, NC)
                else:
                    okc = (_exact_d(k, i + 1, tau, t_probe, c_lookup, c_rank, kap) < T)[:, None, None]
                    vc = V[i + 1, :, k, :, :, 1]
            for fresh in (1, 0):
                v = vm.clone()
                a = torch.full_like(v, COMMIT_M, dtype=torch.int8)
                if fresh and i >= 1:
                    vp = u[:, None, None] * g_p[i, k][None]
                    t = vp > v
                    v, a = torch.where(t, vp, v), torch.where(t, t_cp, a)
                if vpr is not None:
                    t = (vpr > v) & okp
                    v, a = torch.where(t, vpr, v), torch.where(t, t_pr, a)
                if vc is not None:
                    t = vc > v if okc is None else (vc > v) & okc
                    v, a = torch.where(t, vc, v), torch.where(t, t_co, a)
                V[i, :, k, :, :, fresh], A[i, :, k, :, :, fresh] = v, a
    return V, _kappa_major(A)


def solve_osla_exact(g_m, g_p, P, tau, T, t_probe, c_lookup, c_rank):
    """solve_osla() with the ranking time counted exactly.  An action the period
    cannot fit for a given kappa enters the candidate list at -inf, so it is never chosen,
    and the remaining candidates keep solve_osla()'s order and its first-maximum ties."""
    kap = _kappas()
    NEGINF = torch.tensor(float('-inf'), dtype=torch.float64, device=DEV)
    V = torch.zeros((N_EXIT + 1, K1, K1, NB, NC, 2), dtype=torch.float64, device=DEV)
    A = torch.zeros((N_EXIT + 1, K1, K1, NB, NC, 2), dtype=torch.int8, device=DEV)
    for i in range(N_EXIT, -1, -1):
        for k in range(K_MAX, -1, -1):
            d = _exact_d(k, i, tau, t_probe, c_lookup, c_rank, kap)
            u = torch.clamp(1.0 - d / T, min=0.0)
            vm = u[:, None, None] * g_m[i, k][None]
            myopic = true_p = None
            if k < K_MAX:
                d2 = _exact_d(k + 1, i, tau, t_probe, c_lookup, c_rank, kap)
                u2 = torch.clamp(1 - d2 / T, min=0.0)
                ok2 = (d2 < T)[:, None, None]
                myopic = torch.where(ok2, u2[:, None, None]
                                     * _probe_value(P[i, k], g_m[i, k + 1])[None], NEGINF)
                true_p = _probe_values(P[i, k], V[i, :, k + 1, :, :, 0])
            comp = vnext = None
            if i < N_EXIT:
                if i == 0:
                    d3 = torch.full((K1,), _first_call_d(k, tau, t_probe, c_lookup, c_rank),
                                    dtype=torch.float64, device=DEV)
                    vnext = V[1, k, k, :, :, 1][None].expand(K1, NB, NC)
                else:
                    d3 = _exact_d(k, i + 1, tau, t_probe, c_lookup, c_rank, kap)
                    vnext = V[i + 1, :, k, :, :, 1]
                u3 = torch.clamp(1 - d3 / T, min=0.0)
                comp = torch.where((d3 < T)[:, None, None], u3[:, None, None]
                                   * torch.maximum(g_m[i + 1, k], g_p[i + 1, k])[None], NEGINF)
            for fresh in (1, 0):
                cand = [(vm, COMMIT_M)]
                if fresh and i >= 1:
                    cand.append((u[:, None, None] * g_p[i, k][None], COMMIT_P))
                if myopic is not None:
                    cand.append((myopic, PROBE))
                if comp is not None:
                    cand.append((comp, COMPUTE))
                pick = torch.stack([c[0] for c in cand]).argmax(0)
                lut = torch.tensor([c[1] for c in cand], dtype=torch.int8, device=DEV)
                a = lut[pick]
                v = vm.clone()
                if fresh and i >= 1:
                    v = torch.where(a == COMMIT_P, u[:, None, None] * g_p[i, k][None], v)
                if true_p is not None:
                    v = torch.where(a == PROBE, true_p, v)
                if comp is not None:
                    v = torch.where(a == COMPUTE, vnext, v)
                V[i, :, k, :, :, fresh], A[i, :, k, :, :, fresh] = v, a
    return V, _kappa_major(A)


def solve_blinkered_exact(g_m, g_p, P, tau, T, t_probe, c_lookup, c_rank):
    """solve_blinkered() with the ranking time counted exactly; the pure-chain values
    carry kappa too, and infeasible candidates enter at -inf as in solve_osla_exact."""
    kap = _kappas()
    NEGINF = torch.tensor(float('-inf'), dtype=torch.float64, device=DEV)
    V = torch.zeros((N_EXIT + 1, K1, K1, NB, NC, 2), dtype=torch.float64, device=DEV)
    A = torch.zeros((N_EXIT + 1, K1, K1, NB, NC, 2), dtype=torch.int8, device=DEV)
    Wp = torch.zeros((N_EXIT + 1, K1, K1, NB, NC), dtype=torch.float64, device=DEV)
    Wc = torch.zeros((N_EXIT + 1, K1, K1, NB, NC), dtype=torch.float64, device=DEV)
    for i in range(N_EXIT, -1, -1):
        for k in range(K_MAX, -1, -1):
            d = _exact_d(k, i, tau, t_probe, c_lookup, c_rank, kap)
            u = torch.clamp(1.0 - d / T, min=0.0)
            vm = u[:, None, None] * g_m[i, k][None]
            chain_p = true_p = ok_pr = None
            if k < K_MAX:
                ok_pr = (_exact_d(k + 1, i, tau, t_probe, c_lookup, c_rank, kap) < T)[:, None, None]
                chain_p = _probe_values(P[i, k], Wp[i, :, k + 1])
                true_p = _probe_values(P[i, k], V[i, :, k + 1, :, :, 0])
                Wp[i, :, k] = torch.where(ok_pr, torch.maximum(vm, chain_p), vm)
            else:
                Wp[i, :, k] = vm
            here = (u[:, None, None] * torch.maximum(g_m[i, k], g_p[i, k])[None]
                    if i >= 1 else vm)
            wc_next = vnext = ok_co = None
            if i < N_EXIT:
                if i == 0:
                    ok_co = _first_call_d(k, tau, t_probe, c_lookup, c_rank) < T
                    ok_co = torch.full((K1, 1, 1), ok_co, dtype=torch.bool, device=DEV)
                    wc_next = Wc[1, k, k][None].expand(K1, NB, NC)
                    vnext = V[1, k, k, :, :, 1][None].expand(K1, NB, NC)
                else:
                    ok_co = (_exact_d(k, i + 1, tau, t_probe, c_lookup, c_rank, kap) < T)[:, None, None]
                    wc_next = Wc[i + 1, :, k]
                    vnext = V[i + 1, :, k, :, :, 1]
                Wc[i, :, k] = torch.where(ok_co, torch.maximum(here, wc_next), here)
            else:
                Wc[i, :, k] = here
            for fresh in (1, 0):
                cand = [(vm, COMMIT_M)]
                if fresh and i >= 1:
                    cand.append((u[:, None, None] * g_p[i, k][None], COMMIT_P))
                if chain_p is not None:
                    cand.append((torch.where(ok_pr, chain_p, NEGINF), PROBE))
                if wc_next is not None:
                    cand.append((torch.where(ok_co, wc_next, NEGINF), COMPUTE))
                pick = torch.stack([c[0] for c in cand]).argmax(0)
                lut = torch.tensor([c[1] for c in cand], dtype=torch.int8, device=DEV)
                a = lut[pick]
                v = vm.clone()
                if fresh and i >= 1:
                    v = torch.where(a == COMMIT_P, u[:, None, None] * g_p[i, k][None], v)
                if true_p is not None:
                    v = torch.where(a == PROBE, true_p, v)
                if vnext is not None:
                    v = torch.where(a == COMPUTE, vnext, v)
                V[i, :, k, :, :, fresh], A[i, :, k, :, :, fresh] = v, a
    return V, _kappa_major(A)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=None)   # read by protocol.seed_arg
    ap.add_argument('--lat', default='lat_cpu1t_pcore.json', help="latency file")
    ap.add_argument('--tau-scale', type=float, default=1.0,
                    help="rho: scale the inference charge the planner is given (sweeps)")
    ap.add_argument('--tp-air-us', type=float, default=None,
                    help="probe air time, us (sweeps); default obs.T_P_US")
    ap.add_argument('--variants', default=None,
                    help="comma-separated subset of the variant keys to solve")
    ap.add_argument('--out-suffix', default='',
                    help="write tables_<seed><suffix>.npz and scheduler<suffix>.json")
    ap.add_argument('--oracle-mix', type=float, default=None,
                    help="predictor quality q: the oracle beam ranks first for a "
                         "fraction q of receivers; default: the trained predictor")
    ap.add_argument('--rank-charge', action='store_true',
                    help="charge the masked argmax that picks each next codeword "
                         "after an inference call (the sweep_select timing) to the "
                         "variants that re-rank; default: not charged")
    ap.add_argument('--rank-exact', action='store_true',
                    help="charge the ranking only on the probes after the first inference "
                         "call (kappa in the state, kappa-major tables); implies --rank-charge")
    ap.add_argument('--c-lookup-aug', choices=('file', 'original'), default='file',
                    help="the augmented table's lookup charge: 'file' reads lat_lookup_aug.json, "
                         "'original' uses the original table's (the pilot that is timed)")
    cli = ap.parse_args()
    if cli.rank_exact:                  # the exact count charges the same c_rank
        cli.rank_charge = True
    t0 = time.time()
    lat = json.load(open(os.path.join(HERE, cli.lat)))
    # The restart charge, not the cumulative one: every compute action is a
    # fresh pass to its exit (see sim.elapsed_us and latency.py).
    tc = lat['tau_restart_us']
    TAU = np.array([0.0, tc['1'], tc['2'], tc['3']])
    if cli.tau_scale != 1.0:
        TAU = TAU * cli.tau_scale
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    C_LOOK = sim.lookup_charge(lat)       # real-table lookup on the primary profile
    T_PROBE = (obs.T_P_US if cli.tp_air_us is None else cli.tp_air_us) + ch['probe_update']
    # c_rank, not scaled by rho -- like the lookup and the per-probe processing it is
    # controller logic, not inference
    CR = ch['sweep_select'] if cli.rank_charge else 0.0
    # the variants that re-rank use the kappa-augmented table, whose consult is timed on
    # it; the others keep C_LOOK, the charge for the table of the original shape
    C_LOOK_AUG = None
    if cli.rank_exact:
        C_LOOK_AUG = C_LOOK if cli.c_lookup_aug == 'original' else sim.lookup_charge_aug(lat)
    AUG = ('dp', 'osla', 'blind', 'one', 'blnk', 'rep1', 'deep1') if cli.rank_exact else ()
    GRID = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    site = sim.Site(FIT)
    print(f"fit {FIT} n={site.n}  tau={TAU.round(3)}  probe={T_PROBE:.3f} "
          f"lookup={C_LOOK} us", flush=True)
    rec = {"tau_us": TAU.tolist(), "c_lookup_us": C_LOOK, "t_probe_us": T_PROBE,
           "fit": FIT, "k_max": K_MAX, "replicates": REPLICATES, "min_cell": MIN_N,
           "policy_iterations": 1, "on_policy_reps": ON_REPS, "epsilon": EPS,
           "seeds": []}
    if (cli.lat, cli.tau_scale, cli.tp_air_us, cli.variants) != \
            ('lat_cpu1t_pcore.json', 1.0, None, None):
        rec["charging"] = {"lat_file": cli.lat, "tau_scale": cli.tau_scale,
                           "tp_air_us": cli.tp_air_us, "variants": cli.variants}
    if cli.oracle_mix is not None:      # absent otherwise, so no schema moves
        rec["oracle_mix"] = cli.oracle_mix
        rec.setdefault("charging", {})["oracle_mix"] = cli.oracle_mix
    if cli.rank_charge:                 # absent otherwise, so no schema moves
        rec["c_rank_us"] = CR
        rec.setdefault("charging", {})["c_rank_us"] = CR
    if cli.rank_exact:                  # absent otherwise, so no schema moves
        rec["c_rank_mode"] = "exact"
        rec["c_lookup_aug_us"] = C_LOOK_AUG
        rec["c_lookup_aug_source"] = cli.c_lookup_aug
        rec["table_layout"] = {"augmented": "kappa-major [kappa, i, k, b, c, fresh], int8, "
                                            f"shape {[K_MAX + 1, N_EXIT + 1, K_MAX + 1, NB, NC, 2]}",
                               "other": f"[i, k, b, c, fresh], shape {[N_EXIT + 1, K_MAX + 1, NB, NC, 2]}"}
        rec.setdefault("charging", {}).update(c_rank_mode="exact", c_lookup_aug_us=C_LOOK_AUG,
                                              c_lookup_aug_source=cli.c_lookup_aug)
    TAU0 = np.zeros_like(TAU)
    # (action set, whether COMPUTE also re-ranks the remaining probes,
    #  deepest exit the policy may reach)
    VARIANTS = {
        # c_rank=CR for every variant that re-ranks after an inference call;
        # dpn and noc keep 0 (dpn never re-ranks, noc never infers)
        'dp':   (lambda pr, T: solve(*pr, TAU, float(T), T_PROBE, C_LOOK, c_rank=CR),
                 True, N_EXIT),
        'dpn':  (lambda pr, T: solve(*pr, TAU, float(T), T_PROBE, C_LOOK),
                 False, N_EXIT),
        'noc':  (lambda pr, T: solve(*pr, TAU, float(T), T_PROBE, C_LOOK,
                                     allow_compute=False), True, N_EXIT),
        'osla': (lambda pr, T: solve_osla(*pr, TAU, float(T), T_PROBE, C_LOOK,
                                          c_rank=CR), True, N_EXIT),
        # Cost-blind planner.  Same action set, same estimated model, same
        # predictor -- but it PLANS with tau zeroed (TAU0) while still being
        # charged the honest lookup cost C_LOOK, and is then EVALUATED under the
        # honest charges.  Zeroing C_LOOK too would confound the contrast with
        # the controller's own overhead, so it is not zeroed.  This is the
        # comparison that separates "accounting for computation when choosing
        # actions" from "having a predictor available at all".
        'blind': (lambda pr, T: solve(*pr, TAU0, float(T), T_PROBE, C_LOOK,
                                      c_rank=CR), True, N_EXIT),
        # Cheapest-exit-only.  May still choose whether and when to compute,
        # but not how far: separates selective invocation from depth allocation.
        'one':  (lambda pr, T: solve(*pr, TAU, float(T), T_PROBE, C_LOOK,
                                     max_exit=1, c_rank=CR), True, 1),
        # Blinkered: unbounded lookahead within one action type, never
        # interleaving.  Strictly stronger than the myopic rule and the
        # comparator that makes the negative result mean something.
        'blnk': (lambda pr, T: solve_blinkered(*pr, TAU, float(T), T_PROBE,
                                               C_LOOK, c_rank=CR), True, N_EXIT)}
    if cli.rank_exact:
        # the same variants, solved with the ranking counted from the first call on and
        # charged the augmented table's lookup
        CLA = C_LOOK_AUG
        VARIANTS.update({
            'dp': (lambda pr, T: solve_exact(*pr, TAU, float(T), T_PROBE, CLA, CR), True, N_EXIT),
            'osla': (lambda pr, T: solve_osla_exact(*pr, TAU, float(T), T_PROBE, CLA, CR),
                     True, N_EXIT),
            'blind': (lambda pr, T: solve_exact(*pr, TAU0, float(T), T_PROBE, CLA, CR),
                      True, N_EXIT),
            'one': (lambda pr, T: solve_exact(*pr, TAU, float(T), T_PROBE, CLA, CR, max_exit=1),
                    True, 1),
            'blnk': (lambda pr, T: solve_blinkered_exact(*pr, TAU, float(T), T_PROBE, CLA, CR),
                     True, N_EXIT)})
    # Variants whose compute action is a different predictor or exit rule.
    # Each gets its OWN off-policy rollouts, because its
    # predictions and its probe order differ from the headline controller's; the
    # other variants' rollouts, tables and random streams are untouched
    # (explore() seeds a fresh generator per call).
    NEWV = sim.variant_specs(lat, cli.tau_scale)
    if cli.variants:
        keep = cli.variants.split(',')
        known = set(VARIANTS) | set(NEWV)
        assert set(keep) <= known, f"unknown variants {set(keep) - known}"
        VARIANTS = {k: v for k, v in VARIANTS.items() if k in keep}
        NEWV = {k: v for k, v in NEWV.items() if k in keep}
    rec["variant_tau_us"] = {k: v[1].tolist() for k, v in NEWV.items()}
    if cli.rank_exact:                  # which tables are kappa-major
        rec["augmented_variants"] = sorted(k for k in list(VARIANTS) + list(NEWV) if k in AUG)
    for seed in SEEDS:
        model = sim.load_predictor(seed)
        # the predictor-quality condition's anytime predictor, read by both
        # rollout passes; None (the default) leaves every call as it was.
        MS = sim.with_mix(None, sim.oracle_mix(site, FIT, seed, cli.oracle_mix))
        off = {r: explore(site, model, seed, TAU, reorder=r, spec=MS) for r in (True, False)}
        g0 = {r: primitives(v) for r, v in off.items()}
        tabs = {}
        covn = {"cnt_off": off[True][0].float().cpu().numpy(),
                "tot_off": off[True][1].sum((-2, -1)).float().cpu().numpy()}
        for T in GRID:
            for key, (fn, ro, mx) in VARIANTS.items():
                _, A0 = fn(g0[ro][:3], T)
                tabs[f'{key}off_{T}'] = A0.cpu().numpy()
                on = explore(site, model, seed, TAU, table=A0.cpu().numpy(),
                             eps=EPS, reps=ON_REPS, reorder=ro, max_exit=mx, spec=MS,
                             rank_exact=key in AUG)
                g1 = primitives(on, off[ro])
                _, A1 = fn(g1[:3], T)
                tabs[f'{key}pi_{T}'] = A1.cpu().numpy()
                if key == 'dp':         # counts for the coverage of deployed decisions
                    covn[f'cnt_on_{T}'] = on[0].float().cpu().numpy()
                    covn[f'tot_on_{T}'] = on[1].sum((-2, -1)).float().cpu().numpy()
        if NEWV:
            nets = sim.load_nets(seed)
            for key, (sp, tv, ro, mx) in NEWV.items():
                sp = sim.with_mix(sp, None if MS is None else MS.mix)   # as evaluate.py
                offv = explore(site, model, seed, tv, reorder=ro, max_exit=mx,
                               spec=sp, nets=nets)
                g0v = primitives(offv)
                for T in GRID:
                    fn = (lambda pr, T, tv=tv, mx=mx, ks=sp.ks, cr=(CR if ro else 0.0):
                          solve(*pr, tv, float(T), T_PROBE, C_LOOK, max_exit=mx,
                                compute_ks=ks, c_rank=cr))
                    if key in AUG:              # rep1 and deep1 re-rank
                        fn = (lambda pr, T, tv=tv, mx=mx, ks=sp.ks:
                              solve_exact(*pr, tv, float(T), T_PROBE, C_LOOK_AUG, CR,
                                          max_exit=mx, compute_ks=ks))
                    _, A0 = fn(g0v[:3], T)
                    tabs[f'{key}off_{T}'] = A0.cpu().numpy()
                    on = explore(site, model, seed, tv, table=A0.cpu().numpy(), eps=EPS,
                                 reps=ON_REPS, reorder=ro, max_exit=mx, spec=sp, nets=nets,
                                 rank_exact=key in AUG)
                    _, A1 = fn(primitives(on, offv)[:3], T)
                    tabs[f'{key}pi_{T}'] = A1.cpu().numpy()
        if 'dp' in VARIANTS:            # untracked (.gitignore); read by evaluate.py
            np.savez_compressed(os.path.join(HERE, f'coverage_{seed}{cli.out_suffix}.npz'),
                                **covn)
        np.savez_compressed(os.path.join(HERE, f'tables_{seed}{cli.out_suffix}.npz'),
                            g_m=g0[True][0].cpu().numpy(),
                            g_p=g0[True][1].cpu().numpy(), **tabs)
        d = g0[True][3]
        rec['seeds'].append({"seed": seed, **d,
                             "table_bytes": int(next(iter(tabs.values())).nbytes)})
        print(f"  seed {seed}: {d['visits']:.0f} off-policy visits over "
              f"{d['reachable_cells']}/{d['cells']} cells; visits in a thin cell "
              f"g_m {d['visit_frac_in_thin_gm']:.5f} trans "
              f"{d['visit_frac_in_thin_trans']:.5f}; refined {len(GRID)} x "
              f"{len(VARIANTS)} policies", flush=True)
    rec['seconds'] = round(time.time() - t0, 1)
    json.dump(rec, open(os.path.join(HERE, f'scheduler{cli.out_suffix}.json'), 'w'), indent=1)
    print(f"tables built in {rec['seconds']}s")
