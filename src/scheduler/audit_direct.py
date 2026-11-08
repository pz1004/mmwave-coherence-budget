#!/usr/bin/env python3
"""Rule R26: the decision audit's independent auditor (protocol.json analysis 'decision_audit').

  python3 audit_direct.py --pilot       Los Angeles: this script's own rates and selections against
                                        the stored rates and the screen side's efficiencies
                                        (audit_pilot.json); the numerical check that sets the
                                        tolerance eps -> audit_direct_pilot.json (before the freeze)
  python3 audit_direct.py --rates       Houston and Phoenix: the rates, selections and population
                                        efficiencies -> audit_rates.npz (after the freeze; read by
                                        nothing the screen runs)
  python3 audit_direct.py               after every session: the audit -> audit_direct.json
  python3 audit_direct.py --check-only  recomputes audit_direct.json and compares, writing nothing

Independence.  This file computes everything it reports from the ray-traced paths, the declared
constants and the recorded completion times, with code of its own: it imports no module of this
repository.  It reads the raw path files (src/sites/scenarios/<city>/*_t001_tx000_r000.mat), the
qualification record's eligibility digest, the frozen record in protocol.json, the timing sessions
and, for the comparison only, audit_decisions.json and audit_bounds.json.  It does not redraw the
noise: realization r of a city is regenerated from its declared seed (numpy default_rng of the
CRC-32 of 'decision_audit|<city>|<r>', z = standard_normal((n, 64)), then q = chisquare(31,
(n, 64))), as R26 defines it.

Rates.  Per user and narrow beam b (the 64-beam DFT codebook on a 64-element half-wavelength ULA
along y), at the 16 sub-band centres f_j of the 100 MHz band, from the paths' powers P_l (dB),
phases phi_l, delays tau_l and departure angles (theta_l polar, varphi_l azimuth):
    a_l(f_j) = 10^(P_l / 20) exp(i phi_l) exp(-i 2 pi tau_l f_j),
    psi_l = pi sin(theta_l) sin(varphi_l),
    h_j^T f_b = sum_l a_l(f_j) D(psi_l - 2 pi b / 64) / 8,
with D(x) = sum_{n<64} exp(i n x) = exp(i 63 x / 2) sin(32 x) / sin(x / 2) (64 where sin(x/2) = 0),
    C(u, b) = (1/16) sum_j log2(1 + P_TX |h_j^T f_b|^2 / sigma^2),
    pbar(u, b) = (1/16) sum_j |h_j^T f_b|^2,
P_TX = 1 W (30 dBm) and sigma^2 = 1.9953e-12 W (Table I: 7 dB noise figure over 100 MHz).  The
genie-aided rate is C*(u) = max_b C(u, b).

Reference.  The sweep's measurement of beam b at pilot N, (s2 / 32)((z + sqrt(lam))^2 + q) - s2
clipped at 1e-3 s2, s2 = sigma^2 16 / N, lam = 2 N P_TX pbar / sigma^2; it selects the largest of
the 64 in the bit-reversal order, ties to the earlier.  Its rate is C(u, b_S).

Audit, per cell (city, pilot, build).  With uniform weights over the user-realization pairs and the
timing weights v of a record:
    U(T) = sum_{i,r} w_i v_r ((T - D_r)^+ / T) C*(i),   R_S(T) = sum_i w_i ((T - d_S)^+ / T) C(i, b_S),
    A(T) = (U(T) - R_S(T)) / G,   G = sum_i w_i C*(i),
the perfect predictor's advantage over the sweep as a share of the genie-aided rate, d_S the
declared reference completion time.  The calibration record weighs each of the five calibration
sessions equally (1 / (5 n_s) per call); a later session weighs its calls equally.  At the primary
period H = 10 ms: dominated if A(H) < -eps, an opportunity if A(H) > eps, a tie otherwise.  Every
period up to H (secondary): T A(T) - eps T is piecewise linear in T with breakpoints at the
completion times and d_S, so its largest value on (0, H] is attained at a breakpoint or at H; no
period has A > eps exactly when that largest value is at most zero.
"""
import argparse, hashlib, json, os, sys, time, zlib

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = os.path.join(HERE, '..', 'sites')
LAT = os.path.join(HERE, '..', 'latency')
OUT = os.path.join(HERE, 'audit_direct.json')
RATES = os.path.join(HERE, 'audit_rates.npz')
PILOT_OUT = os.path.join(HERE, 'audit_direct_pilot.json')
CITIES = ('city_3_houston_28', 'city_4_phoenix_28')
PILOT_CITY = 'city_1_losangeles_28'
SAMPLES = (1000, 2000, 3000)
PRIMARY, FRESH = tuple(range(1, 11)), tuple(range(11, 21))
BUILDS = ('resnet18_ts', 'vit_fp16_graph', 'vit_fp32')
K_F, M, N_ANT, BW = 16, 64, 64, 100e6
P_TX, SIGMA2 = 1.0, 1.9953e-12
CHUNK = 1024
H_US = 10_000.0


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def crc_seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def bit_reversal(m=M):
    bits = int(np.log2(m))
    return np.array([int(format(i, f'0{bits}b')[::-1], 2) for i in range(m)])


# ------------------------------------------------------------------ channel ---
def raw(city):
    d = os.path.join(SITES, 'scenarios', city)
    out = {}
    for k in ('power', 'phase', 'delay', 'aod_el', 'aod_az', 'rx_pos'):
        m = sio.loadmat(os.path.join(d, f'{k}_t001_tx000_r000.mat'))
        out[k] = np.asarray(m[k], dtype=np.float64)
    return out


def eligible(R):
    """The qualification rule: every path parameter finite in the same positions, at least one
    path, a finite position and a finite channel."""
    fin = [np.isfinite(R[k]) for k in ('power', 'phase', 'delay', 'aod_el', 'aod_az')]
    same = np.all([(f == fin[0]).all(1) for f in fin[1:]], axis=0)
    ok = same & (fin[0].sum(1) > 0) & np.isfinite(R['rx_pos']).all(1)
    return np.flatnonzero(ok)


def dirichlet(x):
    s = np.sin(x / 2)
    tiny = np.abs(s) < 1e-12
    val = np.exp(1j * (N_ANT - 1) * x / 2) * np.sin(N_ANT * x / 2) / np.where(tiny, 1.0, s)
    return np.where(tiny, complex(N_ANT), val)


def rates(city):
    """(C, pbar), each (n, 64) float64, for the eligible users in the qualification order, and the
    eligibility digest check."""
    R = raw(city)
    elig = eligible(R)
    q = json.load(open(os.path.join(SITES, 'qualification.json')))
    site = next(s for s in q['sites'] if s['scenario'] == city)
    want = next(t for t in site['transmitters'] if t['tx_index'] == site['chosen_tx_index'])
    assert (site['chosen_tx_set_id'], site['chosen_tx_index']) == (1, 0), city
    digest = hashlib.sha256(elig.astype(np.int64).tobytes()).hexdigest()[:16]
    f = (np.arange(K_F) + 0.5) * BW / K_F - BW / 2                     # sub-band centres
    grid = 2 * np.pi * np.arange(M) / M
    n = elig.size
    C, Pb = np.empty((n, M)), np.empty((n, M))
    for s in range(0, n, CHUNK):
        u = elig[s:s + CHUNK]
        ok = np.isfinite(R['power'][u])
        amp = np.where(ok, 10 ** (np.where(ok, R['power'][u], 0) / 20), 0.0)
        ph = np.deg2rad(np.where(ok, R['phase'][u], 0))
        tau = np.where(ok, R['delay'][u], 0)
        psi = np.pi * np.sin(np.deg2rad(np.where(ok, R['aod_el'][u], 0))) * \
            np.sin(np.deg2rad(np.where(ok, R['aod_az'][u], 0)))
        Dk = dirichlet(psi[:, :, None] - grid[None, None, :]) / np.sqrt(N_ANT)    # (u, l, b)
        acc_c, acc_p = np.zeros((u.size, M)), np.zeros((u.size, M))
        for fj in f:
            a = amp * np.exp(1j * (ph - 2 * np.pi * tau * fj))                    # (u, l)
            g = np.abs(np.einsum('ul,ulb->ub', a, Dk)) ** 2
            acc_p += g
            acc_c += np.log2(1 + P_TX * g / SIGMA2)
        C[s:s + CHUNK], Pb[s:s + CHUNK] = acc_c / K_F, acc_p / K_F
    assert np.isfinite(C).all() and np.isfinite(Pb).all()
    return C, Pb, {'eligible': int(n), 'digest': digest, 'digest_recorded':
                   want['eligible_sha256'], 'digest_equal': digest == want['eligible_sha256']}


# ---------------------------------------------------------------- reference ---
def noise(city, r, n, pilot=False):
    tag = ('decision_audit', 'pilot', city, r) if pilot else ('decision_audit', city, r)
    rng = np.random.default_rng(crc_seed(*tag))
    return rng.standard_normal((n, M)), rng.chisquare(2 * K_F - 1, (n, M))


def sweep(Pb, z, q, n_samp, order):
    s2 = SIGMA2 * K_F / n_samp
    lam = 2.0 * n_samp * P_TX * Pb / SIGMA2
    y = np.maximum(s2 / (2 * K_F) * ((z + np.sqrt(lam)) ** 2 + q) - s2, 1e-3 * s2)
    return order[y[:, order].argmax(1)]


def selections(city, C, Pb, reals, pilot=False):
    order = bit_reversal()
    n = C.shape[0]
    sel = np.empty((len(reals), len(SAMPLES), n), dtype=np.int8)
    for i, r in enumerate(reals):
        z, q = noise(city, r, n, pilot)
        for j, ns in enumerate(SAMPLES):
            sel[i, j] = sweep(Pb, z, q, ns, order)
    return sel


def eta(C, sel):
    """The ratio of user-averaged rates: the sweep's over the genie-aided, per pilot."""
    rows = np.arange(C.shape[0])
    G = C.max(1).mean()
    return [float(np.mean([C[rows, sel[i, j]].mean() for i in range(sel.shape[0])]) / G)
            for j in range(sel.shape[1])]


# ------------------------------------------------------------------- timing ---
def session(stem):
    z = np.load(os.path.join(LAT, 'profile_audit', stem + '.npz'))
    s, e = z['start_ns'], z['end_ns']
    win = s >= int(z['window_from_ns'])
    return (e[win] - s[win]) / 1e3                                    # us


def advantage(D, v, T, eta_pop, d_s):
    """A(T) for completion times D with weights v (summing to one)."""
    return float(np.dot(v, np.maximum(T - D, 0.0)) / T - eta_pop * max(T - d_s, 0.0) / T)


def worst_period(D, v, H, eta_pop, d_s, eps):
    """max over (0, H] of T A(T) - eps T, at the breakpoints in (0, H] and at H, with the
    argmax; exact because the function is piecewise linear in T."""
    o = np.argsort(D)
    Ds, vs = D[o], v[o]
    cw, cwd = np.cumsum(vs), np.cumsum(vs * Ds)
    pts = np.unique(np.concatenate([Ds[Ds <= H], [d_s] if d_s <= H else [], [H]]))
    k = np.searchsorted(Ds, pts, side='right')            # calls with D <= T
    w = np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
    wd = np.where(k > 0, cwd[np.maximum(k - 1, 0)], 0.0)
    f = pts * w - wd - eta_pop * np.maximum(pts - d_s, 0.0) - eps * pts
    i = int(np.argmax(f))
    return float(f[i]), float(pts[i])


# -------------------------------------------------------------------- pilot ---
def pilot():
    assert not os.path.exists(PILOT_OUT), f"{PILOT_OUT} exists: written once"
    t0 = time.time()
    C, Pb, dg = rates(PILOT_CITY)
    t_rates = time.time() - t0
    st = np.load(os.path.join(HERE, 'data', f'{PILOT_CITY}.npz'))
    se_st = st['se'][:, 62:].astype(np.float64)
    pb_st = st['pbar'][:, 62:].astype(np.float64)
    t1 = time.time()
    sel = selections(PILOT_CITY, C, Pb, PRIMARY, pilot=True)
    t_sel = time.time() - t1
    e_dir = eta(C, sel)
    AP = json.load(open(os.path.join(HERE, 'audit_pilot.json')))
    e_scr = [AP['eta_screen_side'][str(ns)] for ns in SAMPLES]
    delta = max(abs(a - b) for a, b in zip(e_dir, e_scr))
    eps = max(1e-6, 10.0 ** np.ceil(np.log10(100 * delta))) if delta > 0 else 1e-6
    out = {'rule': 'R26', 'city': PILOT_CITY, 'eligibility': dg,
           'rate_vs_stored': {
               'max_abs': float(np.abs(C - se_st).max()),
               'max_rel': float((np.abs(C - se_st) / np.maximum(se_st, 1e-12)).max()),
               'mean_genie_rel': float(abs(C.max(1).mean() / se_st.max(1).mean() - 1))},
           'pbar_vs_stored_max_rel': float((np.abs(Pb - pb_st) / pb_st).max()),
           'genie_beam_agreement': float((C.argmax(1) == se_st.argmax(1)).mean()),
           'eta_direct': dict(zip(map(str, SAMPLES), e_dir)),
           'eta_screen_side': dict(zip(map(str, SAMPLES), e_scr)),
           'delta': delta,
           'eps_rule': "max(1e-6, 10^ceil(log10(100 delta))), delta the largest difference "
                       "between the two sides' efficiencies over the pilots",
           'eps': float(eps),
           'seconds': {'rates': round(t_rates, 1), 'selections': round(t_sel, 1)},
           'script_sha256': sha(os.path.abspath(__file__))}
    assert dg['digest_equal'], dg
    json.dump(out, open(PILOT_OUT, 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('rate_vs_stored', 'pbar_vs_stored_max_rel',
                                          'genie_beam_agreement', 'eta_direct',
                                          'eta_screen_side', 'delta', 'eps', 'seconds')},
                     indent=1))


# ---------------------------------------------------------------- the audit ---
def frozen():
    P = json.load(open(os.path.join(HERE, 'protocol.json')))
    A = next(a for a in P['analyses'] if a['key'] == 'decision_audit')
    assert A.get('frozen'), "not frozen"
    return A


def compute_rates():
    arrays, rec = {}, {}
    for city in CITIES:
        C, Pb, dg = rates(city)
        assert dg['digest_equal'], dg
        sel = selections(city, C, Pb, PRIMARY + FRESH)
        arrays[f'{city}.C'] = C
        arrays[f'{city}.sel'] = sel
        rec[city] = {'eligibility': dg,
                     'eta_pop': dict(zip(map(str, SAMPLES), eta(C, sel[:len(PRIMARY)]))),
                     'eta_fresh': dict(zip(map(str, SAMPLES), eta(C, sel[len(PRIMARY):])))}
        print(f"  {city}: eta_pop {rec[city]['eta_pop']}", flush=True)
    return arrays, rec


def records(fz):
    """Every session's window completion times, by build: the calibration sessions (rounds 1-5)
    and the later ones (rounds 6-15), each the attempt that did not fail."""
    cal, later, used = {b: [] for b in BUILDS}, {b: [] for b in BUILDS}, {}
    for o in fz['order']:
        stem = None
        for st in (f"a{o['slot']:02d}_a2", f"a{o['slot']:02d}"):
            p = os.path.join(LAT, 'profile_audit', st + '.json')
            if os.path.exists(p) and not json.load(open(p))['failed']:
                stem = st
                break
        assert stem, f"slot {o['slot']} has no record that did not fail"
        rec = json.load(open(os.path.join(LAT, 'profile_audit', stem + '.json')))
        assert rec['frozen'] == fz and rec['runs'][0]['build'] == o['build']
        D = session(stem)
        (cal if o['round'] <= fz['calibration_rounds'] else later)[o['build']].append(
            (o['round'], stem, D))
        used[o['slot']] = stem
    return cal, later, used


def mixture(sessions):
    D = np.concatenate([d for _, _, d in sessions])
    v = np.concatenate([np.full(d.size, 1.0 / (len(sessions) * d.size)) for _, _, d in sessions])
    return D, v


def compute():
    A = frozen()
    fz = A['frozen']
    eps, H = fz['eps'], fz['H_us']
    assert H == H_US
    dS = {int(k): v for k, v in fz['reference']['d_S_us'].items()}
    R0 = json.load(open(RATES[:-4] + '.json'))
    assert sha(RATES) == R0['_npz_sha256'], "audit_rates.npz differs from its record"
    DEC = json.load(open(os.path.join(HERE, 'audit_decisions.json')))
    BND = json.load(open(os.path.join(HERE, 'audit_bounds.json')))
    assert sha(os.path.join(HERE, 'audit_decisions.json')) == A['decisions_sha256']
    cal, later, used = records(fz)
    grid = [float(t) for t in fz['grid_us']]
    out = {'rule': 'R26', 'eps': eps, 'H_us': H, 'sessions_used': used, 'cells': {},
           'populations': R0, 'script_sha256': sha(os.path.abspath(__file__))}
    # the timing records
    tim = {}
    for b in BUILDS:
        D, v = mixture(cal[b])
        tim[b] = {'calibration': {'n_calls': [int(d.size) for _, _, d in cal[b]],
                                  'mean_us': float(np.dot(v, D)), 'min_us': float(D.min()),
                                  'max_us': float(D.max()),
                                  'session_means_us': [float(d.mean()) for _, _, d in cal[b]],
                                  'share_at_or_above_H': float(np.dot(v, D >= H))},
                  'later': [{'round': rnd, 'stem': st, 'n_calls': int(d.size),
                             'mean_us': float(d.mean()), 'min_us': float(d.min()),
                             'max_us': float(d.max())} for rnd, st, d in later[b]]}
    out['timing'] = tim
    for city in CITIES:
        for j, ns in enumerate(SAMPLES):
            ep = R0[city]['eta_pop'][str(ns)]
            ef = R0[city]['eta_fresh'][str(ns)]
            ds = dS[ns]
            for b in BUILDS:
                D, v = mixture(cal[b])
                key = f"{city}|{ns}|{b}"
                a_h = advantage(D, v, H, ep, ds)
                wp, wt = worst_period(D, v, H, ep, ds, eps)
                lat = [advantage(d, np.full(d.size, 1.0 / d.size), H, ep, ds)
                       for _, _, d in later[b]]
                lat_wp = [worst_period(d, np.full(d.size, 1.0 / d.size), H, ep, ds, eps)[0]
                          for _, _, d in later[b]]
                lab = 'dominated' if a_h < -eps else ('opportunity' if a_h > eps else 'tie')
                cell = {'city': city, 'samples': ns, 'build': b, 'd_S_us': ds, 'eta_pop': ep,
                        'A_H': a_h, 'direct_label': lab,
                        'all_periods_no_opportunity': wp <= 0.0, 'worst_period_value': wp,
                        'worst_period_T_us': wt,
                        'A_grid': [advantage(D, v, T, ep, ds) for T in grid],
                        'later_A_H': lat, 'later_all_periods_no_opportunity':
                            [x <= 0.0 for x in lat_wp],
                        'later_dominated': int(sum(x < -eps for x in lat)),
                        'later_tie': int(sum(abs(x) <= eps for x in lat)),
                        'later_opportunity': int(sum(x > eps for x in lat)),
                        'fresh_eta': ef, 'fresh_A_H': advantage(D, v, H, ef, ds)}
                cell['fresh_label'] = ('dominated' if cell['fresh_A_H'] < -eps else
                                       'opportunity' if cell['fresh_A_H'] > eps else 'tie')
                # the screen's decisions, draw by draw, against the direct labels
                by_k = {}
                for k in map(str, fz['budgets']):
                    dl = DEC['cells'][key][k]['labels']                  # one per draw
                    n_ex = sum(x == 'excluded' for x in dl)
                    n_un = sum(x == 'unresolved' for x in dl)
                    n_no = sum(x == 'not excluded' for x in dl)
                    contra = sum(x == 'excluded' and a_h > eps for x in dl)
                    missed = sum(x == 'not excluded' and a_h < -eps for x in dl)
                    bl = BND['bounds'][f"{city}|{ns}|{k}"]
                    lbs = [x['eta_lb'] for x in bl]
                    cov = [x is not None and x <= ep for x in lbs]
                    by_k[k] = {'draws': len(dl), 'excluded': n_ex, 'unresolved': n_un,
                               'not_excluded': n_no, 'contradicted': contra,
                               'contradiction_fraction': (contra / n_ex) if n_ex else None,
                               'missed': missed,
                               'coverage': float(np.mean(cov)),
                               'failed_bounds': sum(x is None for x in lbs),
                               'persistence': {
                                   'later_rounds_dominated': cell['later_dominated'],
                                   'later_rounds': len(lat)} if n_ex else None}
                cell['screen'] = by_k
                out['cells'][key] = cell
    # yield per replication: the share of the 18 cells (all feasible) excluded in each draw
    out['feasible'] = {b: tim[b]['calibration']['share_at_or_above_H'] < 1.0 for b in BUILDS}
    yields = {}
    for k in map(str, fz['budgets']):
        n_draws = fz['draws']
        per = []
        for d in range(n_draws):
            per.append(sum(DEC['cells'][key][k]['labels'][d] == 'excluded'
                           for key in out['cells']))
        cells = [key for key in out['cells'] if out['feasible'][out['cells'][key]['build']]]
        yields[k] = {'per_draw_excluded': per, 'feasible_cells': len(cells),
                     'median': float(np.median(per)), 'min': int(min(per)), 'max': int(max(per)),
                     'q05': float(np.quantile(per, 0.05)), 'q95': float(np.quantile(per, 0.95)),
                     'contradictions_total': sum(c['screen'][k]['contradicted']
                                                 for c in out['cells'].values()),
                     'missed_total': sum(c['screen'][k]['missed'] for c in out['cells'].values()),
                     'unresolved_total': sum(c['screen'][k]['unresolved']
                                             for c in out['cells'].values())}
    out['yield'] = yields
    out['burden'] = {k: BND['burden'][k] for k in map(str, fz['budgets'])}
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--pilot', action='store_true')
    g.add_argument('--rates', action='store_true')
    g.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.pilot:
        return pilot()
    A = frozen()
    assert A['frozen']['scripts_sha256']['audit_direct.py'] == sha(os.path.abspath(__file__)), \
        "audit_direct.py differs from the frozen script"
    if a.rates:
        assert not os.path.exists(RATES), f"{RATES} exists: written once"
        t0 = time.time()
        arrays, rec = compute_rates()
        np.savez_compressed(RATES, **arrays)
        rec['_npz_sha256'] = sha(RATES)
        rec['_seconds'] = round(time.time() - t0, 1)
        json.dump(rec, open(RATES[:-4] + '.json', 'w'), indent=1)
        print(f"wrote {RATES} in {rec['_seconds']} s")
        return
    out = json.loads(json.dumps(compute()))
    if a.check_only:
        same = out == json.load(open(OUT))
        print(f"audit_direct.json {'reproduces' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    assert not os.path.exists(OUT), f"{OUT} exists: written once (use --check-only)"
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")
    for k, y in out['yield'].items():
        print(f"  k = {k}: excluded per draw median {y['median']} of {y['feasible_cells']} "
              f"(range {y['min']}-{y['max']}); contradictions {y['contradictions_total']}, "
              f"missed {y['missed_total']}, unresolved {y['unresolved_total']}")


if __name__ == '__main__':
    main()
