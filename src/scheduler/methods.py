"""Every method under test.  Each returns (d_us, c) -- elapsed time before the
payload starts, and the flat codeword index committed to -- for every receiver.

Time accounting is uniform: a probe costs T_PROBE = 20 us + the receiver-side
bookkeeping charge; a predictor costs its measured G3 charge including its own
argmax; the scheduler additionally pays one table consult per decision; and
selection among already-measured beams is free because the running best is
maintained by the per-probe bookkeeping already charged.
"""
from __future__ import annotations
import numpy as np
import torch

import obs, feats, sim
from sim import DEV, K_MAX, N_EXIT, COMMIT_M, COMMIT_P, PROBE, COMPUTE

L6 = obs.LAYER6_OFFSET
NEG = -1e30


def _spread(n):
    return torch.tensor(obs.spread_order().copy(), device=DEV)[None, :].expand(n, -1)


# ----------------------------------------------------------- zero-compute ---
def topk(site, y, k, t_probe):
    """Probe the first k of the frozen spread order; commit to the best."""
    idx = torch.tensor(obs.spread_order()[:k].copy(), device=DEV)
    b = idx[y[:, L6:][:, idx].argmax(1)]
    return torch.full((site.n,), k * t_probe, device=DEV), L6 + b


def hierarchical(site, y, layers, t_probe, c_dec=0.0):
    """Binary hierarchical search: two probes per layer, refine the winner.
    This is exactly Zhang et al.'s fixed-length HIBS baseline."""
    n = site.n
    b = y[:, 0:2].argmax(1)
    for l in range(2, layers + 1):
        ch = (2 ** l - 2) + 2 * b
        w = torch.stack([y.gather(1, ch[:, None])[:, 0],
                         y.gather(1, (ch + 1)[:, None])[:, 0]], 1).argmax(1)
        b = 2 * b + w
    d = torch.full((n,), 2 * layers * t_probe + layers * c_dec, device=DEV)
    return d, (2 ** layers - 2) + b


# ------------------------------------------------------ fixed-compute nets ---
def fixed_net(site, y, k, kind, model, tau, t_probe, mix=None):
    """Probe k beams in the frozen order, run the network once, commit to its
    top-1.  The literature's convention: predict and use, without verifying.
    mix (sim.oracle_mix) applies to the anytime predictor only."""
    n = site.n
    idx = torch.tensor(obs.spread_order()[:k].copy(), device=DEV)
    mask = torch.zeros((n, 64), dtype=torch.bool, device=DEV)
    mask[:, idx] = True
    yn = y[:, L6:]
    s = torch.clamp(feats.snr_db_t(yn), feats.CLIP_LO, feats.CLIP_HI)
    with torch.no_grad():
        if kind == 'fc':
            rec = model(s[:, idx].float() / 20.0).argmax(1)
        elif kind == 'cnn':
            x = torch.stack([torch.where(mask, s, torch.full_like(s, feats.UNPROBED)) / 20.0,
                             mask.to(s.dtype)], 1).float()
            rec = model(x).argmax(1)
        else:                                    # 'ee<e>' anytime predictor
            e = int(kind[2:])
            f, _ = feats.build(yn, mask)
            rec = sim.apply_mix(sim.exit_logits(model, f, e), mix).argmax(1)
    return torch.full((n,), k * t_probe + tau, device=DEV), L6 + rec


# ------------------------------------------------- Zhang et al. (2020) -------
def _chi2_ppf(q, df):
    """Wilson-Hilferty: accurate to <0.3% for df >= 30, which is the range here
    (df = 32 r).  Used only to set the Neyman-Pearson threshold."""
    from math import sqrt
    z = 0.8416212335729143 if abs(q - 0.8) < 1e-12 else None
    assert z is not None, q
    return df * (1 - 2 / (9 * df) + z * sqrt(2 / (9 * df))) ** 3


def zhang(site, y, T, t_probe, c_dec, adaptive_layer, adaptive_slots,
          gen=None, r_max=4, p_fa=0.2, max_layer=6):
    """AHIBS (adaptive layer termination) and AHTBS (adaptive training length).

    Adaptations, both documented: the paper assumes a single dominant path with
    ideal sector-antenna gains, so its predicted gain from narrowing one layer
    is exactly 2x -- kept here as the criterion's own model, while the channel
    it is applied to is DeepMIMO multipath; and it charges feedback separately
    at T_f = 10-100 us, where G1 folds 10 us of feedback into the 20 us probe.
    """
    n = site.n
    thr = [obs.SIGMA_P2 * (_chi2_ppf(0.8, 2 * obs.K_F * r) / (2 * obs.K_F * r) - 1)
           for r in range(1, r_max + 1)]

    def probe(flat):
        """Measure codeword `flat` (n,), returning (value, slots_used)."""
        v = y.gather(1, flat[:, None])[:, 0]
        if not adaptive_slots:
            return v, torch.ones(n, device=DEV)
        acc, slots, done = v.clone(), torch.ones(n, device=DEV), v > thr[0]
        for r in range(2, r_max + 1):
            extra = sim.measure(site.pbar.gather(1, flat[:, None])[:, 0], gen)
            acc = torch.where(done, acc, (acc * (r - 1) + extra) / r)
            slots = torch.where(done, slots, torch.full_like(slots, r))
            done = done | (acc > thr[r - 1])
        return acc, slots

    b = torch.zeros(n, dtype=torch.long, device=DEV)
    v0, s0 = probe(b)                       # layer 1 flat indices are 0 and 1
    v1, s1 = probe(b + 1)
    b = (v1 > v0).long()
    gam = torch.maximum(v0, v1) * obs.P_TX / obs.SIGMA_D2
    slots = s0 + s1
    layer = torch.ones(n, dtype=torch.long, device=DEV)
    ndec = torch.ones(n, device=DEV)
    live = torch.ones(n, dtype=torch.bool, device=DEV)

    for l in range(2, max_layer + 1):
        d_now = slots * t_probe + ndec * c_dec
        if adaptive_layer:
            d_nxt = d_now + 2 * t_probe + c_dec
            u_now = torch.clamp(1 - d_now / T, min=0.0)
            u_nxt = torch.clamp(1 - d_nxt / T, min=0.0)
            go = live & (u_nxt * torch.log2(1 + 2 * gam) > u_now * torch.log2(1 + gam))
        else:
            go = live.clone()
        if not go.any():
            break
        ch = (2 ** l - 2) + 2 * b
        va, sa = probe(ch)
        vb, sb = probe(ch + 1)
        w = (vb > va).long()
        nb = 2 * b + w
        nv = torch.maximum(va, vb)
        b = torch.where(go, nb, b)
        gam = torch.where(go, nv * obs.P_TX / obs.SIGMA_D2, gam)
        slots = torch.where(go, slots + sa + sb, slots)
        ndec = torch.where(go, ndec + 1, ndec)
        layer = torch.where(go, torch.full_like(layer, l), layer)
        live = go
    d = slots * t_probe + ndec * c_dec
    return d, (2 ** layer - 2) + b, slots, layer


# ---------------------------------------------- Zhang et al., deadline-guarded -----
# Neither form of the adaptive baseline stops a descent at the end of the period, while the
# controller allows only actions that fit within it.  The guarded forms
# below keep each form's detection and hierarchy rules and add the controller's rule: before
# every look, the elapsed time plus that look, the check itself and the layer decision that may
# follow must stay below T, so no descent overruns the period.  A check
# costs c_guard, the timed threshold comparison (np_test), whether it passes or not.  A look
# the period cannot fit ends the descent: no later look is cheaper, so nothing after it fits.
# A test the guard cuts short keeps the looks it has; a layer whose two children both have a
# first look is complete, and the beam is the winner of the deepest complete layer (none if
# layer 1 is incomplete, which then delivers nothing).
def zhang_guarded(site, y, T, t_probe, c_dec, gen, c_guard, r_max=4, max_layer=6):
    """zhang(..., adaptive_layer=True, adaptive_slots=True) with the deadline guard.

    The elapsed time is kept as zhang() keeps it, float32 looks x t_probe + decisions x c_dec,
    plus checks x c_guard, so with c_guard = 0 and a period no descent can overrun, this
    reproduces zhang() exactly.  sim.measure is drawn for every user at every look slot, in
    zhang()'s order; the guard only removes looks and its checks only add time, so the users
    descending a layer are a subset of zhang()'s and the draws a prefix of zhang()'s.  With
    `gen` restored to the state zhang() started from, both see the same repeats.

    Returns (d, codeword, looks, layer, info), info = (share of users the guard stopped or cut
    short, share left without a beam)."""
    n = site.n
    thr = [obs.SIGMA_P2 * (_chi2_ppf(0.8, 2 * obs.K_F * r) / (2 * obs.K_F * r) - 1)
           for r in range(1, r_max + 1)]
    looks = torch.zeros(n, device=DEV)
    ndec = torch.zeros(n, device=DEV)
    nchk = torch.zeros(n, device=DEV)
    hit = torch.zeros(n, dtype=torch.bool, device=DEV)
    el = lambda: looks * t_probe + ndec * c_dec + nchk * c_guard     # elapsed, zhang()'s form

    def probe(flat, act):
        """Measure codeword `flat` for the users in `act`: (value, looks, had a first look)."""
        nonlocal looks, nchk, hit
        v = y.gather(1, flat[:, None])[:, 0]
        fit = act & (el() + c_guard + t_probe + c_dec < T)
        nchk = nchk + act.float()
        looks = looks + fit.float()
        acc, sl = v.clone(), fit.float()
        done = ~fit | (v > thr[0])
        hit = hit | (act & ~fit)
        for r in range(2, r_max + 1):
            extra = sim.measure(site.pbar.gather(1, flat[:, None])[:, 0], gen)
            want = act & ~done
            ok = want & (el() + c_guard + t_probe + c_dec < T)
            nchk = nchk + want.float()
            looks = looks + ok.float()
            acc = torch.where(ok, (acc * (r - 1) + extra) / r, acc)
            sl = torch.where(ok, torch.full_like(sl, r), sl)
            hit = hit | (want & ~ok)
            done = done | (want & ~ok) | (acc > thr[r - 1])
        return acc, sl, fit

    b = torch.zeros(n, dtype=torch.long, device=DEV)
    every = torch.ones(n, dtype=torch.bool, device=DEV)
    v0, s0, f0 = probe(b, every)                 # layer 1 flat indices are 0 and 1
    v1, s1, f1 = probe(b + 1, f0)
    live = f0 & f1                               # layer 1 complete
    b = torch.where(live, (v1 > v0).long(), b)
    gam = torch.maximum(v0, v1) * obs.P_TX / obs.SIGMA_D2
    ndec = ndec + live.float()
    layer = live.long()
    for l in range(2, max_layer + 1):
        d_now = el()
        d_nxt = d_now + 2 * t_probe + c_dec
        u_now = torch.clamp(1 - d_now / T, min=0.0)
        u_nxt = torch.clamp(1 - d_nxt / T, min=0.0)
        go = live & (u_nxt * torch.log2(1 + 2 * gam) > u_now * torch.log2(1 + gam))
        if not go.any():
            break
        ch = (2 ** l - 2) + 2 * b
        va, sa, fa = probe(ch, go)
        vb, sb, fb = probe(ch + 1, go & fa)
        comp = go & fa & fb                      # layer l complete
        w = (vb > va).long()
        b = torch.where(comp, 2 * b + w, b)
        gam = torch.where(comp, torch.maximum(va, vb) * obs.P_TX / obs.SIGMA_D2, gam)
        ndec = ndec + comp.float()
        layer = torch.where(comp, torch.full_like(layer, l), layer)
        live = comp
    nobeam = layer == 0
    d = torch.where(nobeam, torch.full_like(looks, float(T)), el())
    c = (2 ** layer.clamp(min=1) - 2) + b
    return d, c, looks, layer, (float(hit.float().mean()), float(nobeam.float().mean()))


# ------------------------------------- Zhang et al. (2020), as published -----
# zhang() above is a simplified reimplementation: a one-sided test on the running mean
# of the clipped reports,
# at most four looks, and a look-ahead layer rule that predicts the next layer as
# twice the measured SNR.  The paper's own algorithm (Algorithms 1-2, eqs. 13-17
# and 24; AHTBS and AHIBS coincide for single-antenna users, differing only in
# their transmit/receive scan order) is implemented below beside it:
#   * a sequential detection test at the receiver on the running energy sum,
#     H0 noise only, H1 a single Rayleigh path of known variance sigma_a^2 in the
#     codeword's sector (gain C_k = 2^(k-1), Zhang's -3 dB rule), with Wald's
#     thresholds A = (1-Pm)/Pa and B = Pm/(1-Pa), at most NP2_NMAX looks;
#   * the child with the larger average energy wins (Algorithm 2), whatever the
#     test decided;
#   * each codeword costs its looks' pilots plus ONE feedback (Zhang's T_E + T_f);
#   * the layer rule looks back: after training layer k it compares the rate
#     expected from the known statistics, eq. (24), with layer k-1's and stops at
#     the first k where it does not rise (or k = 6), outputting layer k.
# The modelling choices the paper leaves open are those protocol.json records (analysis
# 'rerank'); sigma_a^2 is each receiver's own mean channel power, which favours this baseline.
NP2_PA, NP2_PM, NP2_NMAX = 0.2, 0.05, 3
NP2_LNA = float(np.log((1 - NP2_PM) / NP2_PA))       # ln 4.75
NP2_LNB = float(np.log(NP2_PM / (1 - NP2_PA)))       # ln 0.0625
NP2_PILOT_US = 10.0                                   # g1: a probe is 10 us of pilot
NP2_FB_US = obs.T_P_US - NP2_PILOT_US                 # + 10 us of feedback
assert NP2_FB_US == 10.0, obs.T_P_US


def _ln_1f1_exact(b, z):
    """ln 1F1(1; b; z) for z >= 0 and a scalar b (numpy): the power series below
    z = b, the incomplete-gamma form above it, 1F1(1;b;z) = (b-1) e^z z^(1-b)
    gamma(b-1, z)."""
    from scipy.special import gammaln, gammainc
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    lo = z < b
    if lo.any():
        zz = z[lo]
        term, tot = np.ones_like(zz), np.ones_like(zz)
        for j in range(400):
            term = term * zz / (b + j)
            tot = tot + term
            if float(np.max(term / tot)) < 1e-17:
                break
        out[lo] = np.log(tot)
    hi = ~lo
    if hi.any():
        zz = z[hi]
        out[hi] = (np.log(b - 1) + gammaln(b - 1) + (1 - b) * np.log(zz) + zz
                   + np.log(gammainc(b - 1, zz)))
    return out


_F_TAB = {}                     # b -> (log10 z grid, ln 1F1(1; b; z) on it)


def _f_tab(b):
    if b not in _F_TAB:
        lz = np.linspace(-8.0, 6.0, 140001)
        _F_TAB[b] = (lz, _ln_1f1_exact(b, 10.0 ** lz))
    return _F_TAB[b]


def _ln_1f1_one(b, z):
    """ln 1F1(1; b; z), b a multiple of K_F (one per number of looks): interpolated
    in log z on a fine table between 1e-8 and 1e6, exact outside it."""
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), np.shape(z))
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    for bv in np.unique(b):
        sel = b == bv
        zz = z[sel]
        lz, F = _f_tab(float(bv))
        o = np.empty_like(zz)
        small, big = zz < 1e-8, zz > 1e6
        mid = ~(small | big)
        o[small] = np.log1p(zz[small] / bv)
        o[mid] = np.interp(np.log10(zz[mid]), lz, F)
        if big.any():
            o[big] = _ln_1f1_exact(float(bv), zz[big])
        out[sel] = o
    return out


def np2_lnlr(S, n, m):
    """ln LR of n looks' energy sum S (in measure()'s chi-square units, 2*K_F
    degrees of freedom per look): H1 a Rayleigh path with mean noncentrality 2m
    per look, constant over the looks, H0 noise only.  The Poisson mixture with an
    exponential mean is geometric, so the ratio is a confluent hypergeometric."""
    th = n * np.asarray(m, dtype=np.float64)
    z = np.asarray(S, dtype=np.float64) * th / (2.0 * (1.0 + th))
    return -np.log1p(th) + _ln_1f1_one(obs.K_F * n, z)


def _np2_t0(m):
    """The fade threshold t0 = |alpha|^2/sigma_a^2 at which a noise-free first look
    reaches the upper threshold, ln LR_1 = ln A: eq. (24)'s gamma, per receiver."""
    m = np.asarray(m, dtype=np.float64)
    t0 = np.full(m.shape, np.inf)
    ok = m > 1e-12
    if ok.any():
        mm = m[ok]
        # ln LR_1 = -ln(1+m) + F(z), z = X m / (2(1+m)), F = ln 1F1(1; K_F; .)
        # increasing: invert F at ln A + ln(1+m) on its table, then X_U and t0
        lz, F = _f_tab(float(obs.K_F))
        c = NP2_LNA + np.log1p(mm)
        assert float(c.max()) < float(F[-1]), "threshold beyond the 1F1 table"
        zs = 10.0 ** np.interp(c, F, lz)
        t0[ok] = (2.0 * (1.0 + mm) * zs / mm) / (2.0 * mm)
    return t0


def _np2_rate(S_k, t0):
    """Eq. (24)'s expected rate (bit/s/Hz) of a Rayleigh path of mean SNR S_k,
    counting only the fades the first look detects (t >= t0):
    e^-t0 log2(1 + S t0) + e^(1/S) E1(t0 + 1/S) / ln 2."""
    from scipy.special import exp1
    S_k = np.asarray(S_k, dtype=np.float64)
    out = np.zeros_like(S_k)
    ok = (S_k > 0) & np.isfinite(t0)
    S, t = S_k[ok], t0[ok]
    x = t + 1.0 / S
    # e^x E1(x), scaled: direct below x = 50, the asymptotic series above
    ex = np.where(x < 50, np.exp(np.minimum(x, 50)) * exp1(np.minimum(x, 50)),
                  (1 - 1 / x + 2 / x ** 2 - 6 / x ** 3 + 24 / x ** 4) / x)
    out[ok] = np.exp(-t) * np.log2(1 + S * t) + np.exp(-t) * ex / np.log(2)
    return out


def zhang_np2_descent(site, X1, Z, Q, t_probe, c_dec):
    """The T-independent part of the published descent: per receiver and layer,
    the winning codeword, the elapsed time through the layer, eq. (24)'s rate, and
    the test's outcomes.  X1 is the first look of every codeword (the raw statistic
    behind y, sim.measure_chi2); Z, Q (n, 126, >= NP2_NMAX - 1) build looks 2.. as
    (Z + sqrt(lam))^2 + Q, a noncentral chi-square with 2*K_F degrees of freedom."""
    n = site.n
    X1 = X1.cpu().numpy()
    lam = (2.0 * obs.K_F * obs.P_TX * site.pbar / obs.SIGMA_P2).cpu().numpy()
    sa2 = site.pn.mean(1).cpu().numpy()        # |alpha|^2 of a single path: the
    rows = np.arange(n)                        # mean over the 64 DFT beams
    b = np.zeros(n, dtype=np.int64)
    t = np.zeros(n)
    looks = np.zeros(n)
    win = np.zeros((n, 6), dtype=np.int64)
    D = np.zeros((n, 6))
    G = np.zeros((n, 6))
    nlook = np.zeros((n, 6))
    out_h1 = np.zeros((n, 6)); out_h0 = np.zeros((n, 6)); out_tr = np.zeros((n, 6))
    for k in range(1, 7):
        Ck = 2.0 ** (k - 1)
        m = 16.0 * obs.P_TX * sa2 * Ck / obs.SIGMA_P2
        ap, h1, h0, tr, nl = [], [], [], [], []
        for s in (0, 1):
            c = (2 ** k - 2) + 2 * b + s
            S = X1[rows, c].copy()
            r = np.ones(n)
            ln = np2_lnlr(S, 1.0, m)
            dec1, dec0 = ln >= NP2_LNA, ln <= NP2_LNB
            for j in range(2, NP2_NMAX + 1):
                go = ~(dec1 | dec0)
                if not go.any():
                    break
                Xj = (Z[rows, c, j - 2] + np.sqrt(lam[rows, c])) ** 2 + Q[rows, c, j - 2]
                S = np.where(go, S + Xj, S)
                r = np.where(go, r + 1, r)
                ln = np.where(go, np2_lnlr(S, r, m), ln)
                dec1 = dec1 | (go & (ln >= NP2_LNA))
                dec0 = dec0 | (go & (ln <= NP2_LNB))
            ap.append(S / r); nl.append(r)
            h1.append(dec1); h0.append(dec0); tr.append(~(dec1 | dec0))
            t = t + t_probe + (r - 1) * NP2_PILOT_US + r * c_dec
        w = (ap[1] > ap[0]).astype(np.int64)
        b = 2 * b + w
        t = t + c_dec                            # the layer rule's own decision
        looks = looks + nl[0] + nl[1]
        win[:, k - 1] = (2 ** k - 2) + b
        D[:, k - 1] = t
        G[:, k - 1] = _np2_rate(obs.P_TX * sa2 * Ck / obs.SIGMA_D2, _np2_t0(m))
        nlook[:, k - 1] = looks
        out_h1[:, k - 1] = (h1[0].astype(float) + h1[1]) / 2
        out_h0[:, k - 1] = (h0[0].astype(float) + h0[1]) / 2
        out_tr[:, k - 1] = (tr[0].astype(float) + tr[1]) / 2
    return dict(win=win, D=D, G=G, looks=nlook, h1=out_h1, h0=out_h0, tr=out_tr)


def zhang_np2(desc, T):
    """The published layer rule at period T on a descent: stop at the first layer k
    whose expected rate (1 - D_k/T)+ G_k does not exceed layer k-1's (or k = 6), and
    output layer k.  No budget rule, as in the paper: a descent that overruns T
    delivers nothing.  Returns (d, codeword, layer, diagnostics)."""
    D, G = desc['D'], desc['G']
    n = D.shape[0]
    R = np.clip(1.0 - D / T, 0.0, None) * G
    prev = np.concatenate([np.zeros((n, 1)), R[:, :-1]], 1)
    stop = ~(R > prev)
    stop[:, -1] = True
    kopt = stop.argmax(1)                       # first layer where the rule stops
    rows = np.arange(n)
    d = D[rows, kopt]
    lay = kopt + 1
    looks = desc['looks'][rows, kopt]
    trained = 2.0 * lay
    sel = np.arange(6)[None, :] <= kopt[:, None]
    share = lambda a: float((a * sel).sum() / sel.sum())
    diag = (float(looks.mean()), float(lay.mean()), float((looks / trained).mean()),
            share(desc['h1']), share(desc['h0']), share(desc['tr']),
            float((d >= T).mean()))
    return (torch.tensor(d, device=DEV, dtype=torch.float64),
            torch.tensor(desc['win'][rows, kopt], device=DEV, dtype=torch.long), lay, diag)


def zhang_np2_guarded(site, X1, Z, Q, T, t_probe, c_dec, c_guard, desc):
    """The published descent (zhang_np2_descent, zhang_np2) with zhang_guarded's deadline
    guard, at period T, on the same first looks X1 and repeat arrays Z, Q: common
    random numbers with the unguarded form.  Each codeword's time is added as the descent adds
    it, t_p + (r - 1) 10 us + r c_dec for r looks, and the checks' c_guard on top, so with
    c_guard = 0 and a period no descent can overrun this reproduces zhang_np2(desc, T).  A
    check precedes every look: the elapsed time plus the look, its test, the check and the
    layer decision that may follow must stay below T.  The layer rule is the published one on the elapsed time through each complete
    layer, and eq. (24)'s rates G are desc['G'], which do not depend on the looks.
    Returns (d, codeword, layer, diagnostics) as zhang_np2 does, diagnostics extended by the
    share of users the guard stopped or cut short and the share left without a beam."""
    n = site.n
    X1 = X1.cpu().numpy() if torch.is_tensor(X1) else X1
    lam = (2.0 * obs.K_F * obs.P_TX * site.pbar / obs.SIGMA_P2).cpu().numpy()
    sa2 = site.pn.mean(1).cpu().numpy()
    rows = np.arange(n)
    G = desc['G']
    b = np.zeros(n, dtype=np.int64)
    t = np.zeros(n)
    nchk = np.zeros(n)
    looks = np.zeros(n)
    layer = np.zeros(n, dtype=np.int64)
    alive = np.ones(n, dtype=bool)
    hit = np.zeros(n, dtype=bool)
    Rprev = np.zeros(n)
    tl = np.zeros(n)                              # elapsed through the deepest complete layer
    for k in range(1, 7):
        if not alive.any():
            break
        Ck = 2.0 ** (k - 1)
        m = 16.0 * obs.P_TX * sa2 * Ck / obs.SIGMA_P2
        ap, fit1 = [], []
        act = alive.copy()
        for s in (0, 1):
            c = (2 ** k - 2) + 2 * b + s
            el = t + nchk * c_guard
            f = act & (el + c_guard + t_probe + 2 * c_dec < T)
            nchk = nchk + act
            hit = hit | (act & ~f)
            S = X1[rows, c].copy()
            r = f.astype(float)
            ln = np2_lnlr(S, 1.0, m)
            dec1, dec0 = ln >= NP2_LNA, ln <= NP2_LNB
            run = t + f * (t_probe + c_dec)       # elapsed after this codeword's looks so far
            for j in range(2, NP2_NMAX + 1):
                want = f & ~(dec1 | dec0) & (r == j - 1)
                if not want.any():
                    continue
                ok = want & (run + (nchk + 1) * c_guard + NP2_PILOT_US + 2 * c_dec < T)
                nchk = nchk + want
                hit = hit | (want & ~ok)
                Xj = (Z[rows, c, j - 2] + np.sqrt(lam[rows, c])) ** 2 + Q[rows, c, j - 2]
                S = np.where(ok, S + Xj, S)
                r = np.where(ok, r + 1, r)
                run = run + ok * (NP2_PILOT_US + c_dec)
                ln = np.where(ok, np2_lnlr(S, np.maximum(r, 1.0), m), ln)
                dec1 = dec1 | (ok & (ln >= NP2_LNA))
                dec0 = dec0 | (ok & (ln <= NP2_LNB))
            rr = np.maximum(r, 1.0)
            ap.append(S / rr)
            fit1.append(f)
            t = np.where(f, t + t_probe + (r - 1) * NP2_PILOT_US + r * c_dec, t)
            looks = looks + r
            act = act & f                         # child 1 only after child 0's first look
        comp = alive & fit1[0] & fit1[1]
        w = (ap[1] > ap[0]).astype(np.int64)
        t = np.where(comp, t + c_dec, t)          # the layer rule's own decision
        b = np.where(comp, 2 * b + w, b)
        layer = np.where(comp, k, layer)
        Dk = t + nchk * c_guard
        tl = np.where(comp, Dk, tl)
        Rk = np.clip(1.0 - Dk / T, 0.0, None) * G[:, k - 1]
        cont = comp & (Rk > Rprev) & (k < 6)
        Rprev = np.where(comp, Rk, Rprev)
        # a descent the guard stopped inside layer k keeps layer k-1 and the time it spent
        stopped = alive & ~comp
        tl = np.where(stopped, t + nchk * c_guard, tl)
        alive = cont
    nobeam = layer == 0
    d = np.where(nobeam, float(T), tl)
    cw = (2 ** np.maximum(layer, 1) - 2) + b
    diag = (float(looks.mean()), float(layer.mean()), float(hit.mean()), float(nobeam.mean()),
            float(((d >= T) & ~nobeam).mean()))
    return (torch.tensor(d, device=DEV, dtype=torch.float64),
            torch.tensor(cw, device=DEV, dtype=torch.long), layer, diag)


# ------------------------------------------------------------- scheduler -----
def scheduled(site, y, model, table, tau, t_probe, c_look, reorder=True,
              spec=None, nets=None, visits=None, trace=None, c_rank=0.0,
              rank_exact=False, kappa_out=None):
    """Run a DP threshold table faithfully: the true probe identities, the true
    measurements, the real predictor.  Only the CONTROLLER is a table.

    spec (sim.Spec) is the compute action a variant runs (sim.variant_specs), read
    from the same object its planner's rollouts used; None is the headline controller.
    visits, if given, is a (N_EXIT+1, K_MAX+1, NB, NC) tensor that accumulates
    one count per DECISION the deployed policy takes in each estimation cell,
    so the coverage of its decisions by well-estimated cells can be measured on
    the evaluation sites rather than on the rollouts that built the model.
    trace, if a list, receives per decision step (live mask, and the level, k,
    fresh, best and mean measured SNR dB of every receiver) -- for the live ones,
    the states the table is actually consulted in, replayed by retime.py to
    time the consult on the real table.
    c_rank charges the masked argmax below that picks each next codeword
    once the exit's scores re-rank the search, on every probe of
    a receiver that has invoked the predictor, as sim.elapsed_us does for the
    planner; it applies only when `reorder` is true.
    rank_exact: `table` is kappa-major (sched.solve_exact); each user's
    probe count at its first inference call, kappa, picks the block it consults from then on,
    and c_rank is charged on the k - kappa probes after that call, the ranking operations the
    loop below performs (sim.elapsed_exact_us).  The trace then carries kappa as a sixth
    column, and kappa_out, if a list, receives every user's kappa (-1: never called)."""
    n = site.n
    yn = y[:, L6:]
    snr = feats.snr_db_t(yn)
    probed = torch.zeros((n, 64), dtype=torch.bool, device=DEV)
    score = sim.spread_score().expand(n, -1).clone()
    lvl = torch.zeros(n, dtype=torch.long, device=DEV)
    k = torch.zeros(n, dtype=torch.long, device=DEV)
    fresh = torch.zeros(n, dtype=torch.long, device=DEV)
    rec = torch.zeros(n, dtype=torch.long, device=DEV)
    bidx = torch.zeros(n, dtype=torch.long, device=DEV)
    bsnr = torch.full((n,), -1e9, device=DEV, dtype=torch.float64)
    msum = torch.zeros(n, device=DEV, dtype=torch.float64)
    done = torch.zeros(n, dtype=torch.bool, device=DEV)
    beam = torch.zeros(n, dtype=torch.long, device=DEV)
    used_pred = torch.zeros(n, dtype=torch.bool, device=DEV)
    tab = torch.tensor(table, device=DEV, dtype=torch.long)
    kap = torch.zeros(n, dtype=torch.long, device=DEV)        # probes at the first call

    for _ in range(K_MAX + N_EXIT + 2):
        any_p = k > 0
        b = sim.bucket(bsnr, any_p)
        c = sim.level_class(msum / k.clamp(min=1), any_p)
        act = tab[kap, lvl, k, b, c, fresh] if rank_exact else tab[lvl, k, b, c, fresh]
        act = torch.where(done, torch.full_like(act, -1), act)
        if trace is not None:
            cols = [lvl.double(), k.double(), fresh.double(), bsnr, msum / k.clamp(min=1)]
            trace.append(((~done).cpu().numpy(),
                          torch.stack(cols + ([kap.double()] if rank_exact else []),
                                      1).cpu().numpy()))
        if visits is not None:
            live = ~done
            visits.index_put_((lvl[live], k[live], b[live], c[live]),
                              torch.ones(int(live.sum()), device=DEV, dtype=visits.dtype),
                              accumulate=True)
        cm, cp = act == COMMIT_M, act == COMMIT_P
        if cm.any():
            beam = torch.where(cm, bidx, beam); done = done | cm
        if cp.any():
            beam = torch.where(cp, rec, beam)
            used_pred = used_pred | cp; done = done | cp
        pr, co = act == PROBE, act == COMPUTE
        if co.any():
            f, _ = feats.build(yn, probed)
            for e in range(N_EXIT):
                s = co & (lvl == e)
                if not s.any():
                    continue
                if spec is None:
                    lg = sim.exit_logits(model, f[s], e + 1)
                else:
                    lg = sim.compute_logits(spec, model, nets, yn, probed, f, s, e, k)
                if reorder:
                    score[s] = lg
                rec[s] = lg.argmax(1)
            kap = torch.where(co & (lvl == 0), k, kap)
            lvl = lvl + co.long(); fresh = torch.where(co, torch.ones_like(fresh), fresh)
        if pr.any():
            nxt = torch.where(probed, torch.full_like(score, NEG), score).argmax(1)
            probed[pr, nxt[pr]] = True
            ns = snr.gather(1, nxt[:, None])[:, 0]
            better = pr & (ns > bsnr)
            bidx = torch.where(better, nxt, bidx)
            bsnr = torch.where(better, ns, bsnr)
            msum = msum + torch.where(pr, ns, torch.zeros_like(ns))
            k = k + pr.long(); fresh = torch.where(pr, torch.zeros_like(fresh), fresh)
        if bool(done.all()):
            break
    beam = torch.where(done, beam, bidx)
    d = k * t_probe + torch.tensor(tau, device=DEV, dtype=torch.float64)[lvl] \
        + (k + lvl + 1) * c_look
    if c_rank and reorder:
        n_rank = (k - kap) if rank_exact else k     # rank_exact: only the probes after the first call
        d = d + c_rank * n_rank.double() * (lvl >= 1).double()
    if kappa_out is not None:
        kappa_out.append(torch.where(lvl >= 1, kap, torch.full_like(kap, -1)))
    return d, L6 + beam, k, lvl, used_pred


def oracle_stop(site, y, t_probe, T):
    """Upper bound on EVERY causal probe-and-commit rule in the frozen order:
    a clairvoyant that knows, per receiver, the stopping time that maximises
    max(0, 1 - k t_p / T) * SE(best of the first k probes).  No method can beat
    it, so it bounds the whole stopping class including the scheduler."""
    order = torch.tensor(obs.spread_order().copy(), device=DEV)
    yn = y[:, L6:][:, order]
    run = yn.cummax(1).indices                     # position of the best so far
    beams = order[run]                             # (n, 64) beam chosen after k+1 probes
    se = site.se[:, L6:].gather(1, beams)
    k = torch.arange(1, 65, device=DEV, dtype=torch.float64)[None, :]
    val = torch.clamp(1 - k * t_probe / T, min=0.0) * se
    best = val.max(1)
    return best.values, beams.gather(1, best.indices[:, None])[:, 0] + L6
