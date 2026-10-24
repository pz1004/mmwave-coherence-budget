#!/usr/bin/env python3
"""Channel aging within one re-acquisition period, for the fixed menu.

The question is what the accounting looks like when the channel is not
block-faded within the period.  This replaces the block-fading assumption by a
time-indexed observation law on the same ray-traced paths, for every fixed
method (topK and the sweep, hierarchical search, the per-K FC and CNN networks,
the one-shot anytime exits).  The controller and the published comparator are
not run here.

MODEL.  Within one period (at most 10 ms) the receiver moves under 0.35 m even
at 120 km/h, so the geometry -- every path's angles, delay and power -- is held
fixed and only each path's phase rotates, at its Doppler shift
    nu_l = (v / lambda) * u_l . h,    u_l = (sin th cos ph, sin th sin ph, cos th)
from the path's ARRIVAL direction (aoa_el is the polar angle, aoa_az the azimuth;
u_l points from the receiver toward where the path comes from, so moving toward
it raises the frequency), h a horizontal heading drawn uniformly per receiver
from a per-site generator and shared by every speed and seed, lambda at the
carrier for every sub-band.  Not modelled: departure-angle drift, the Doppler
spread across sub-bands, paths appearing or vanishing, blockage.
    g_{u,c,j}(t) = sum_l a_ul exp(-j 2 pi tau_ul df_j) A_ulc exp(j 2 pi nu_ul t)
with a_ul the path gain and A_ulc the path's response on codeword c -- exactly
build.py's channel at t = 0.  A probe of codeword c taken at time t sees the
band-averaged power P_c(t); the beam held after commitment earns S_c(t).

ANCHORING.  The stored datasets were built from float32 ray-tracing arrays, so a
recomputation does not reproduce them bit for bit.  The aged quantities are
therefore anchored to the stored ones,
    pbar_c(t) = pbar_c * P_c(t) / P_c(0),    se_c(t) = max(0, se_c + S_c(t) - S_c(0)),
and at v = 0 they ARE the stored ones: the v = 0 run must reproduce every
committed beam and every per-deadline mean of results.json and heldout.json.

TIMING.  Probe i of the frozen spread order is taken at i*t_p, and the
hierarchical search probes codeword (l, b) at (2(l-1) + b mod 2)*t_p.  The noise
is evaluate.py's common random numbers -- the same draws, with the noncentrality
taken at the probe time.  The payload earns the held beam's rate integrated over
[d, T]; the trapezoid grid is t_p/4 with every charge, probe time and deadline
placed exactly.  A site mean is linear in the per-receiver integrals, so only the
receiver-mean rate curve of each (method, seed) is accumulated.

OUTPUTS (aging.json; reading rule R6 reads them):
  dominance   min over t in [d_sweep, 10 ms] of s_sweep(t) - s_L(t), per method
              -- Proposition 1 holds verbatim under any evolution if the
              sweep's held-beam rate dominates pointwise from its own commit on
  beam_aging  1 - E[S_b0(t)] / E[max_b S_b(t)] (ratio of means, as eta is),
              with b0 the rate-optimal beam at t = 0; P(argmax = b0); the
              per-receiver mean loss alongside
  accounting  the static and aged versions of what charging inference changes:
              inversions, reversals, winners, feasibility flips, T* shifts

usage: python3 aging.py [--sites ...] [--speeds 0 3 30 60 120] [--out aging.json]
       python3 aging.py --physics-check SITE | --doppler-check SITE |
                        --quadrature-check SITE
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
import scipy.io as sio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'budget'))
import obs, g1, sim, methods, evaluate, protocol

DEV = sim.DEV
SITES_DIR = os.path.join(HERE, '..', 'sites')
EVAL = ['city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28',
        'city_5_philadelphia_28', 'city_6_miami_28', 'city_7_sandiego_28']
SPEEDS_KMH = (0.0, 3.0, 30.0, 60.0, 120.0)
LAMBDA = g1.C_LIGHT / g1.F_C
T_MAX_US = 10000.0
DOM_TOL = 1e-6             # dominance holds if min margin >= -DOM_TOL (bit/s/Hz)
LOSS_THR = 0.01            # beam-aging loss threshold of R6(b)
# accounting.PUBS without the published comparator, which is not run here
PUBS = ['sweep64', 'hier6', 'topk8', 'topk16', 'topk32', 'fc16', 'fc32', 'cnn16', 'cnn32']
TSTAR_METHODS = ('topk8', 'topk16', 'topk32', 'fc16', 'fc32', 'cnn16', 'cnn32', 'hier6')
FIELDS = ('power', 'phase', 'delay', 'aod_el', 'aod_az', 'aoa_el', 'aoa_az', 'inter', 'rx_pos')
L6 = obs.LAYER6_OFFSET


# ------------------------------------------------------------------ data ----
def load_paths(name):
    """Per-path arrays of the eligible receivers, joined through `elig` and
    checked against the qualification digest and the stored dataset."""
    qual = {s['scenario']: s for s in
            json.load(open(os.path.join(SITES_DIR, 'qualification.json')))['sites']}[name]
    tag = f"t{qual['chosen_tx_set_id']:03d}_tx{qual['chosen_tx_index']:03d}_r000"
    raw = {f: sio.loadmat(os.path.join(SITES_DIR, 'scenarios', name, f'{f}_{tag}.mat'))[f]
           for f in FIELDS}
    z = np.load(os.path.join(HERE, 'data', f'{name}.npz'))
    elig = z['elig']
    want = [t for t in qual['transmitters'] if t['tx_index'] == qual['chosen_tx_index']][0]
    assert hashlib.sha256(elig.astype(np.int64).tobytes()).hexdigest()[:16] == \
        want['eligible_sha256'], f"{name}: eligible set differs from qualification"
    P = {f: raw[f][elig] for f in FIELDS}
    assert np.array_equal(P['rx_pos'], z['rx_pos']), f"{name}: rx_pos join failed"
    assert np.array_equal(P['inter'][:, 0] == 0, z['los']), f"{name}: los join failed"
    ok = np.isfinite(P['power'])
    for f in ('aoa_el', 'aoa_az'):
        assert np.array_equal(np.isfinite(P[f]), ok), f"{name}: {f} mask != power mask"
    return P


def headings(name, n):
    rng = np.random.default_rng(evaluate.stable_seed('aging-heading', name))
    return rng.uniform(0.0, 2.0 * np.pi, n)


def doppler_hz(P, rows, psi_h, v_mps):
    """(m, 25) per-path Doppler shift; padded paths get 0."""
    th = np.deg2rad(P['aoa_el'][rows].astype(np.float64))
    ph = np.deg2rad(P['aoa_az'][rows].astype(np.float64))
    nu = (v_mps / LAMBDA) * np.sin(th) * np.cos(ph - psi_h[rows, None])
    return np.where(np.isfinite(nu), nu, 0.0)


_F = None


def coeffs(P, rows):
    """X[u, c*K_F + j, l] = a_ul exp(-j 2 pi tau_ul df_j) A_ulc, complex128."""
    global _F
    if _F is None:
        _F = torch.tensor(obs.hierarchical_codebook(), dtype=torch.complex128, device=DEV)
    a = g1.path_gains(P['power'][rows].astype(np.float64), P['phase'][rows].astype(np.float64))
    tau = P['delay'][rows].astype(np.float64)
    tau = np.where(np.isfinite(tau), tau, 0.0)
    psi = g1.spatial_freq(P['aod_el'][rows].astype(np.float64),
                          P['aod_az'][rows].astype(np.float64))
    psi = np.where(np.isfinite(psi), psi, 0.0)
    t = lambda x: torch.tensor(x, device=DEV)
    ant = torch.arange(g1.N_ANT, device=DEV, dtype=torch.float64)
    A = torch.exp(1j * t(psi)[:, :, None] * ant) @ _F                     # (m, 25, 126)
    df = t(obs.subband_offsets())
    ph = torch.exp(-2j * np.pi * t(tau)[:, :, None] * df)                   # (m, 25, 16)
    X = t(a)[:, :, None, None] * A[:, :, :, None] * ph[:, :, None, :]       # (m,25,126,16)
    return X.permute(0, 2, 3, 1).reshape(X.shape[0], obs.N_CODE * obs.K_F, X.shape[1])


def power_rate(X, nu, times, dtype=torch.complex64):
    """Band-averaged power P (m, 126, nt) and rate S (m, 126, nt) at `times` (us)."""
    ph = 2.0 * np.pi * torch.tensor(nu, device=DEV)[:, :, None] \
        * torch.tensor(times, device=DEV, dtype=torch.float64)[None, None, :] * 1e-6
    G = torch.bmm(X.to(dtype), torch.exp(1j * ph).to(dtype))              # (m, 2016, nt)
    p = (G.real ** 2 + G.imag ** 2).reshape(X.shape[0], obs.N_CODE, obs.K_F, -1)
    S = torch.log2(1.0 + obs.P_TX * p / obs.SIGMA_D2).mean(2)
    return p.mean(2), S


# ----------------------------------------------------------- time grid ------
def build_grid(tp, breaks, step):
    """Uniform `step` grid on [0, T_MAX] plus every breakpoint, exactly."""
    pts = set(np.round(np.arange(0.0, T_MAX_US + step / 2, step), 9).tolist())
    pts |= {float(b) for b in breaks if 0.0 <= b <= T_MAX_US}
    g = np.array(sorted(pts))
    return g, {float(x): i for i, x in enumerate(g)}


def cumtrap(t, f):
    """Cumulative trapezoid of f (..., nt) on the grid t, float64."""
    inc = 0.5 * (f[..., 1:] + f[..., :-1]) * np.diff(t)
    return np.concatenate([np.zeros(f.shape[:-1] + (1,)), np.cumsum(inc, -1)], -1)


def rate_at(t, CI, idx_d, d, T):
    """Site-mean R(T) = (1/T) int_d^T s(t) dt from the cumulative integral."""
    return 0.0 if T <= d else (CI[..., idx_T(t, T)] - CI[..., idx_d]) / T


def idx_T(t, T):
    i = int(np.searchsorted(t, T))
    assert abs(t[i] - T) < 1e-6, (T, t[i])
    return i


# ------------------------------------------------------------- one site -----
def menu(site, yA, yH, ee, nets, q):
    """The fixed menu, as evaluate.py runs it: name -> (d us (float), c (n,))."""
    ch, ONESHOT, tp = q['ch'], q['ONESHOT'], q['T_PROBE']
    runs = {'oracle': (0.0, site.oracle_beam + L6)}
    put = lambda nm, r: runs.__setitem__(nm, (float(r[0][0]), r[1]))
    for k in (1, 2, 4, 8, 16, 32, 64):
        put('sweep64' if k == 64 else f'topk{k}', methods.topk(site, yA, k, tp))
    for L in (3, 4, 5, 6):
        put(f'hier{L}', methods.hierarchical(site, yH, L, tp))
    for k in (4, 8, 16, 32, 64):
        put(f'fc{k}', methods.fixed_net(site, yA, k, 'fc', nets[f'fc{k}'], ch[f'fc_k{k}'], tp))
        put(f'cnn{k}', methods.fixed_net(site, yA, k, 'cnn', nets[f'cnn{k}'], ch['cnn'], tp))
    for e in (1, 2, 3):
        for k in (2, 4, 8, 16, 32):
            put(f'ee{e}k{k}', methods.fixed_net(site, yA, k, f'ee{e}', ee, ONESHOT[e], tp))
    return runs


def noisy(pbar, z, rr):
    """sim.measure with pre-drawn noise: bit-identical for the same draws."""
    lam = 2.0 * obs.K_F * obs.P_TX * pbar / obs.SIGMA_P2
    x = (z + torch.sqrt(lam)) ** 2
    return torch.clamp(obs.SIGMA_P2 / (2 * obs.K_F) * (x + rr) - obs.SIGMA_P2,
                       min=1e-3 * obs.SIGMA_P2)


def probe_times(tp):
    tA = np.zeros(obs.N_CODE)
    tA[L6 + obs.spread_order()] = np.arange(64) * tp           # layer-6, spread order
    tH = np.zeros(obs.N_CODE)
    for l in range(1, obs.N_LAYER + 1):
        b = np.arange(2 ** l)
        tH[obs.flat_index(l, b)] = (2 * (l - 1) + b % 2) * tp
    return tA, tH


def run_site(name, speeds, seeds, lat, step=None, chunk=32, verbose=True):
    site = sim.Site(name)
    q, qf = evaluate.charges(lat), evaluate.charges(lat, free_compute=True)
    tp = q['T_PROBE']
    P = load_paths(name)
    psi_h = headings(name, site.n)
    tA, tH = probe_times(tp)
    ptimes = np.unique(np.concatenate([tA[L6:], tH]))
    se0 = site.se.double()                                     # (n, 126) stored
    strat = np.full(site.n, -1)
    for i, (lo, hi) in enumerate(evaluate.STRATA):
        strat[(site.strat_snr >= lo) & (site.strat_snr < hi)] = i
    nets_by_seed = {s: evaluate.load_nets(s) for s in seeds}
    gens = {}
    for s in seeds:          # evaluate.py's draws, reduced as sim.measure reduces them
        gen = torch.Generator(device=DEV).manual_seed(evaluate.stable_seed(name, s))
        zz = torch.randn(site.pbar.shape, generator=gen, device=DEV, dtype=torch.float64)
        r = torch.randn((*site.pbar.shape, 2 * obs.K_F - 1), generator=gen, device=DEV,
                        dtype=torch.float64)
        gens[s] = (zz, (r * r).sum(-1))
        del r
    out = []
    for v_kmh in speeds:
        t0 = time.time()
        v = v_kmh / 3.6
        # ---- pass 1: anchored powers at the probe times -> the menu's choices
        pA, pH = site.pbar.clone(), site.pbar.clone()
        if v > 0:
            it = {float(x): i for i, x in enumerate(ptimes)}
            iA = torch.tensor([it[float(x)] for x in tA[L6:]], device=DEV)
            iH = torch.tensor([it[float(x)] for x in tH], device=DEV)
            for s0 in range(0, site.n, chunk):
                rows = slice(s0, min(s0 + chunk, site.n))
                Pt, _ = power_rate(coeffs(P, rows), doppler_hz(P, rows, psi_h, v), ptimes)
                Pt = Pt.double()
                ratio = torch.where(Pt[..., :1] > 0, Pt / Pt[..., :1], torch.ones_like(Pt))
                m = Pt.shape[0]
                ar = torch.arange(m, device=DEV)[:, None]
                pA[rows, L6:] = site.pbar[rows, L6:] * ratio[ar, torch.arange(L6, obs.N_CODE,
                                                                       device=DEV), iA]
                pH[rows] = site.pbar[rows] * ratio[ar, torch.arange(obs.N_CODE, device=DEV), iH]
        choice = {}
        for s in seeds:
            ee, nets = nets_by_seed[s]
            zz, rr = gens[s]
            choice[s] = menu(site, noisy(pA, zz, rr), noisy(pH, zz, rr), ee, nets, q)
        names = list(choice[seeds[0]])
        dj = {nm: choice[seeds[0]][nm][0] for nm in names}
        free = menu(site, noisy(pA, *gens[seeds[0]]), noisy(pH, *gens[seeds[0]]),
                    *nets_by_seed[seeds[0]], qf)
        df_ = {nm: free[nm][0] for nm in names}
        breaks = list(dj.values()) + list(df_.values()) + list(ptimes) + \
            [64 * tp] + list(evaluate_grid())
        t, ix = build_grid(tp, breaks, tp / 4 if step is None else step)
        # ---- pass 2: the held beams' rate curves, and the beam-aging curves
        C = torch.stack([torch.stack([choice[s][nm][1] for nm in names]) for s in seeds])
        acc = torch.zeros((len(seeds), len(names), len(t)), dtype=torch.float64, device=DEV)
        ba = torch.zeros((4, len(t)), dtype=torch.float64, device=DEV)   # b0, max, same, rel
        bas = torch.zeros((4, 2, len(t)), dtype=torch.float64, device=DEV)  # strata: b0, max
        strat_t = torch.tensor(strat, device=DEV)
        for s0 in range(0, site.n, chunk):
            rows = slice(s0, min(s0 + chunk, site.n))
            m = rows.stop - rows.start
            if v > 0:
                _, S = power_rate(coeffs(P, rows), doppler_hz(P, rows, psi_h, v), t)
                S = S.double()
                S = torch.clamp(se0[rows][:, :, None] + S - S[..., :1], min=0.0)
            else:
                S = se0[rows][:, :, None].expand(m, obs.N_CODE, len(t))
            Cr = C[:, :, rows]                                              # (S, M, m)
            g = S[torch.arange(m, device=DEV), Cr]                          # (S, M, m, nt)
            acc += g.sum(2)
            S6 = S[:, L6:, :]
            mx = S6.max(1)
            b0 = site.oracle_beam[rows]
            sb0 = S6[torch.arange(m, device=DEV), b0]
            ba[0] += sb0.sum(0); ba[1] += mx.values.sum(0)
            ba[2] += (mx.indices == b0[:, None]).double().sum(0)
            ba[3] += torch.where(mx.values > 0, 1 - sb0 / mx.values,
                                 torch.zeros_like(sb0)).sum(0)
            for i in range(4):
                k_ = strat_t[rows] == i
                if k_.any():
                    bas[i, 0] += sb0[k_].sum(0); bas[i, 1] += mx.values[k_].sum(0)
        curves = (acc / site.n).cpu().numpy()                               # (S, M, nt)
        out.append(summarize(name, v_kmh, t, ix, names, dj, df_, curves,
                             ba.cpu().numpy() / site.n, bas.cpu().numpy(),
                             np.bincount(strat, minlength=4)))
        if v == 0:
            out[-1]['_choice'] = choice           # read by regression(), never written
        if verbose:
            print(f"  {name} {v_kmh:5.1f} km/h: {time.time() - t0:6.1f}s  "
                  f"dominance {'holds' if out[-1]['dominance_holds'] else 'FAILS'}  "
                  f"loss@10ms {out[-1]['beam_aging']['loss'][-1]:.4f}", flush=True)
    return out


def evaluate_grid():
    return json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']


# ------------------------------------------------------------- summaries ----
def _curve12(t, CI, ix_d, d):
    return [float(rate_at(t, CI, ix_d, d, T)) for T in evaluate_grid()]


def summarize(name, v_kmh, t, ix, names, dj, df_, curves, ba, bas, ns):
    G = evaluate_grid()
    CI = cumtrap(t, curves)                                   # (S, M, nt)
    sbar = curves.mean(0)                                     # (M, nt) seed-mean beam rate
    meth = {}
    for mi, nm in enumerate(names):
        J = np.mean([_curve12(t, CI[s, mi], ix[dj[nm]], dj[nm]) for s in range(len(CI))], 0)
        Fr = np.mean([_curve12(t, CI[s, mi], ix[df_[nm]], df_[nm]) for s in range(len(CI))], 0)
        meth[nm] = {"cost_joint_us": dj[nm], "cost_free_us": df_[nm],
                    "joint": J.tolist(), "free": Fr.tolist(),
                    "beam_rate": [float(sbar[mi, idx_T(t, T)]) for T in G]}
    # dominance from the sweep's own commit on
    i0 = idx_T(t, dj['sweep64'])
    ss = sbar[names.index('sweep64')]
    dom = {}
    for mi, nm in enumerate(names):
        if nm in ('sweep64', 'oracle'):
            continue
        marg = ss[i0:] - sbar[mi, i0:]
        j = int(np.argmin(marg))
        dom[nm] = {"min_margin": float(marg[j]), "t_us": float(t[i0 + j]),
                   "holds": bool(marg[j] >= -DOM_TOL)}
    # T*: smallest T >= d_sweep beyond which the sweep is never behind
    Rgrid = {}
    for acct, dd in (('joint', dj), ('free', df_)):
        Rm = {}
        for nm in ['sweep64', *TSTAR_METHODS]:
            mi = names.index(nm)
            ci = CI[:, mi].mean(0)
            Rm[nm] = np.where(t > dd[nm], (ci - ci[ix[dd[nm]]]) / np.maximum(t, 1e-12), 0.0)
        Rgrid[acct] = Rm
    ts = []
    for nm in TSTAR_METHODS:
        row = {"method": nm}
        for acct, dd in (('joint', dj), ('free', df_)):
            Rs, Rl = Rgrid[acct]['sweep64'], Rgrid[acct][nm]
            ok = (t >= dd['sweep64']) & (Rs >= Rl)
            bad = np.flatnonzero(~ok & (t >= dd['sweep64']))
            first = (int(bad[-1]) + 1) if bad.size else int(np.searchsorted(t, dd['sweep64']))
            row[f"T_star_{acct}_us"] = float(t[first]) if first < len(t) else None
        a, b = row["T_star_free_us"], row["T_star_joint_us"]
        row["shift_pct"] = (b / a - 1) * 100 if a and b else None
        ts.append(row)
    lossr = 1 - ba[0] / np.maximum(ba[1], 1e-300)
    runmax = np.maximum.accumulate(lossr)
    keep = np.linspace(0, T_MAX_US, 201)
    kidx = [int(np.argmin(np.abs(t - x))) for x in keep]
    rec = {"site": name, "speed_kmh": v_kmh,
           "dominance": dom, "dominance_holds": all(x['holds'] for x in dom.values()),
           "beam_aging": {"T_us": G,
                          "loss": [float(lossr[idx_T(t, T)]) for T in G],
                          "loss_runmax": [float(runmax[idx_T(t, T)]) for T in G],
                          "loss_per_rx": [float(ba[3][idx_T(t, T)]) for T in G],
                          "p_same": [float(ba[2][idx_T(t, T)]) for T in G],
                          "loss_max_to_10ms": float(runmax[-1]),
                          "loss_strata": [[float(1 - bas[i, 0, idx_T(t, T)] /
                                                 max(bas[i, 1, idx_T(t, T)], 1e-300))
                                           if ns[i] else None for T in G] for i in range(4)],
                          "strata_counts": ns.tolist()},
           "curve": {"t_us": [float(t[i]) for i in kidx],
                     "loss": [float(lossr[i]) for i in kidx],
                     "p_same": [float(ba[2][i]) for i in kidx]},
           "methods": meth, "t_star": ts,
           "accounting": accounting_of(meth)}
    return rec


def accounting_of(meth):
    """accounting.py's comparisons on these curves (PUBS without Zhang)."""
    G = evaluate_grid()
    cur = lambda nm, a: np.array(meth[nm][a])
    def anywhere(a):
        out = set()
        for i, x in enumerate(PUBS):
            for y in PUBS[i + 1:]:
                d = cur(x, a) - cur(y, a)
                if (d > 1e-12).any() and (d < -1e-12).any():
                    out.add((x, y))
        return out
    def ends(a):
        return {(x, y) for i, x in enumerate(PUBS) for y in PUBS[i + 1:]
                if (lambda d: d[0] * d[-1] < 0)(cur(x, a) - cur(y, a))}
    af, aj, pf, pj = anywhere('free'), anywhere('joint'), ends('free'), ends('joint')
    rev = [{"pair": [x, y], "T_us": T} for i, x in enumerate(PUBS) for y in PUBS[i + 1:]
           for k, T in enumerate(G)
           if (cur(x, 'free') - cur(y, 'free'))[k] * (cur(x, 'joint') - cur(y, 'joint'))[k] < 0]
    wf = [max(PUBS, key=lambda m: cur(m, 'free')[k]) for k in range(len(G))]
    wj = [max(PUBS, key=lambda m: cur(m, 'joint')[k]) for k in range(len(G))]
    flips = [{"method": m, "T_us": T} for m in PUBS for k, T in enumerate(G)
             if cur(m, 'free')[k] > 0 and cur(m, 'joint')[k] == 0]
    return {"pubs": PUBS,
            "inversions_anywhere_created_by_compute": sorted(map(list, aj - af)),
            "inversions_anywhere_removed_by_compute": sorted(map(list, af - aj)),
            "inversions_endpoint_created_by_compute": sorted(map(list, pj - pf)),
            "n_reversal_cells": len(rev),
            "winner_changes": [{"T_us": T, "free": a, "joint": b}
                               for T, a, b in zip(G, wf, wj) if a != b],
            "feasibility_flips": flips}


# ------------------------------------------------------------ the checks ----
def regression(name, rows0, choice0, seeds, lat):
    """The v = 0 run against the committed artifacts, and against the static
    methods run in this process on evaluate.py's own measurement."""
    site = sim.Site(name)
    q = evaluate.charges(lat)
    ref = {}
    for f in ('results.json', 'heldout.json', 'validation.json'):
        for s in json.load(open(os.path.join(HERE, f)))['sites']:
            ref[s['site']] = s
    rf = {s['site']: s for s in
          json.load(open(os.path.join(HERE, 'results_freecompute.json')))['sites']}
    G = evaluate_grid()
    worst, worst_f, nd = 0.0, (0.0 if name in rf else None), 0
    for nm, m in rows0['methods'].items():
        r = ref[name]['methods'][nm]
        worst = max(worst, max(abs(a - r['per_T'][str(T)]['mean']) for a, T in zip(m['joint'], G)))
        if nm != 'oracle':
            nd = max(nd, abs(m['cost_joint_us'] - r['cost_us']))
        if name in rf:
            rr = rf[name]['methods'][nm]
            worst_f = max(worst_f, max(abs(a - rr['per_T'][str(T)]['mean'])
                                       for a, T in zip(m['free'], G)))
    same_c = True
    for s in seeds:
        gen = torch.Generator(device=DEV).manual_seed(evaluate.stable_seed(name, s))
        y = sim.measure(site.pbar, gen)
        ee, nets = evaluate.load_nets(s)
        st = menu(site, y, y, ee, nets, q)
        same_c &= all(torch.equal(st[nm][1], choice0[s][nm][1]) for nm in st)
    return {"site": name, "max_abs_joint": worst, "max_abs_free": worst_f,
            "max_abs_cost_us": nd, "committed_beams_identical": bool(same_c),
            "holds": bool(worst <= 1e-5 and (worst_f is None or worst_f <= 1e-5)
                          and nd <= 1e-3 and same_c)}


def physics_check(name, n=64, seed=0):
    """V6: this module's path x codeword evolution against g1.narrowband_at with
    each path's phase advanced by 360 nu t degrees, in float64."""
    P = load_paths(name)
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(P['power']), n, replace=False))
    sub = {f: P[f][rows] for f in P}
    psi_h = rng.uniform(0, 2 * np.pi, n)
    worst = 0.0
    for v_kmh, t_us in ((30.0, 731.0), (120.0, 9377.0), (3.0, 4000.0)):
        nu = doppler_hz(sub, slice(None), psi_h, v_kmh / 3.6)
        Pm, _ = power_rate(coeffs(sub, slice(None)), nu, np.array([t_us]), torch.complex128)
        ph2 = sub['phase'].astype(np.float64) + 360.0 * nu * t_us * 1e-6
        F = obs.hierarchical_codebook()
        acc = 0.0
        for df in obs.subband_offsets():
            h = g1.narrowband_at(sub['power'].astype(np.float64), ph2,
                                 sub['aod_el'].astype(np.float64),
                                 sub['aod_az'].astype(np.float64),
                                 sub['delay'].astype(np.float64), float(df))
            acc = acc + np.abs(h @ F) ** 2
        ref = acc / obs.K_F
        got = Pm[..., 0].cpu().numpy()
        worst = max(worst, float(np.max(np.abs(got - ref) / np.maximum(ref.max(1, keepdims=True),
                                                                         1e-300))))
    return {"site": name, "n_rx": n, "max_rel_err": worst, "holds": worst <= 1e-10}


def doppler_check(name, v_kmh=30.0, dt_us=1.0):
    """V9: the line-of-sight path's Doppler against the finite-difference rate
    of change of the receiver-transmitter distance (tx_pos, rx_pos -- no
    angles), and |nu| <= v / lambda for every path."""
    P = load_paths(name)
    qual = {s['scenario']: s for s in
            json.load(open(os.path.join(SITES_DIR, 'qualification.json')))['sites']}[name]
    tag = f"t{qual['chosen_tx_set_id']:03d}_tx{qual['chosen_tx_index']:03d}_r000"
    tx = sio.loadmat(os.path.join(SITES_DIR, 'scenarios', name, f'tx_pos_{tag}.mat'))['tx_pos']
    tx = tx[0].astype(np.float64)
    n = len(P['power'])
    psi_h = headings(name, n)
    v = v_kmh / 3.6
    nu = doppler_hz(P, slice(None), psi_h, v)
    los = P['inter'][:, 0] == 0
    rx = P['rx_pos'].astype(np.float64)
    hv = np.stack([np.cos(psi_h), np.sin(psi_h), np.zeros(n)], 1)
    d0 = np.linalg.norm(tx - rx, axis=1)
    d1 = np.linalg.norm(tx - (rx + hv * v * dt_us * 1e-6), axis=1)
    nu_geo = -(d1 - d0) / (dt_us * 1e-6) / LAMBDA
    err = np.abs(nu[los, 0] - nu_geo[los]) / (v / LAMBDA)
    return {"site": name, "speed_kmh": v_kmh, "n_los": int(los.sum()),
            "max_rel_err_los": float(err.max()),
            "max_abs_nu_over_vmax": float(np.abs(nu).max() / (v / LAMBDA)),
            "holds": bool(err.max() <= 1e-4 and np.abs(nu).max() <= v / LAMBDA * (1 + 1e-12))}


def quadrature_check(name, seeds, lat, v_kmh=120.0):
    """V10: the declared grid (t_p/4) against t_p/8 at the fastest speed."""
    q = evaluate.charges(lat)
    a = run_site(name, [v_kmh], seeds, lat, verbose=False)[0]
    b = run_site(name, [v_kmh], seeds, lat, step=q['T_PROBE'] / 8, verbose=False)[0]
    worst = max(abs(x - y) for nm in a['methods'] for acct in ('joint', 'free')
                for x, y in zip(a['methods'][nm][acct], b['methods'][nm][acct]))
    return {"site": name, "speed_kmh": v_kmh, "seeds": list(seeds),
            "max_abs_diff_bps_hz": worst, "holds": worst <= 1e-4}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', nargs='+', default=EVAL)
    ap.add_argument('--speeds', nargs='+', type=float, default=list(SPEEDS_KMH))
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--out', default='aging.json')
    ap.add_argument('--physics-check', default=None)
    ap.add_argument('--doppler-check', default=None)
    ap.add_argument('--quadrature-check', default=None)
    ap.add_argument('--chunk', type=int, default=32)
    a = ap.parse_args()
    lat = json.load(open(os.path.join(HERE, evaluate.LAT_FILE)))
    seeds = tuple(a.seeds) if a.seeds else protocol.seeds()
    if a.physics_check:
        print(json.dumps(physics_check(a.physics_check), indent=1)); sys.exit(0)
    if a.doppler_check:
        print(json.dumps(doppler_check(a.doppler_check), indent=1)); sys.exit(0)
    if a.quadrature_check:
        print(json.dumps(quadrature_check(a.quadrature_check, seeds[:2], lat), indent=1))
        sys.exit(0)
    t0 = time.time()
    speeds = sorted(set([0.0] + a.speeds))          # the v = 0 regression always runs
    res = {"model": "per-path Doppler phase rotation over fixed geometry",
           "speeds_kmh": speeds, "seeds": list(seeds), "sites": a.sites,
           "lambda_m": LAMBDA, "grid_step": "t_p/4 plus every breakpoint",
           "dominance_tol": DOM_TOL, "loss_threshold": LOSS_THR, "pubs": PUBS,
           "regression": [], "rows": []}
    for name in a.sites:
        rows = run_site(name, speeds, seeds, lat, chunk=a.chunk)
        r0 = [r for r in rows if r['speed_kmh'] == 0.0][0]
        reg = regression(name, r0, r0.pop('_choice'), seeds, lat)
        res['regression'].append(reg)
        assert reg['holds'], f"v = 0 regression failed: {reg}"
        res['rows'] += rows
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(HERE, a.out), 'w'), indent=1)
    print(f"wrote {a.out} in {res['seconds']}s")
