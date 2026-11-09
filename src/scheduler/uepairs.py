#!/usr/bin/env python3
"""Rule R44: transmit-receive beam pairs with multi-antenna users (protocol.json analysis 'uepairs',
declared before it ran; R44 is post hoc).

The paper's efficiencies are those of single-antenna users; its receive-beam budgets (rule R35)
borrow the per-probe sweep's.  R44 computes them.

Channels.  Each user carries a horizontal half-wavelength ULA of N_r = 1, 2 or 4 elements whose
axis lies at an azimuth alpha_u drawn uniformly on [0, 2 pi) per user (numpy default_rng(
evaluate.stable_seed('uepairs', 'orientation', scenario))), the same for every N_r and seed.  From
the stored ray-traced paths of the qualified transmitter (audit_direct.raw's files and their arrival
angles, theta polar and phi azimuth as for departure), at the 16 sub-band centres f_j,
    h_j[r, n] = sum_l 10^(P_l / 20) e^{i (phi_l - 2 pi tau_l f_j)} e^{i n psi_l} e^{i r chi_l},
    psi_l = pi sin(theta^D_l) sin(phi^D_l),      chi_l = pi sin(theta^A_l) cos(phi^A_l - alpha_u),
and the pair of the 64-beam DFT codebook's beam b with the N_r-beam DFT receive codebook's beam q
has the power |FFT2(h_j)[q, b]|^2 / (64 N_r); its band-averaged power pbar and rate C are the means
over the sub-bands, as build.py forms them.  With N_r = 1 these are the stored channels (gate U1).

Sweep.  Every pair is probed once under Section III's probe law (sim.measure on the pairs' pbar,
1,000 samples): 64 N_r probes, the largest measurement selected; per seed of the protocol's ten on
torch.Generator(evaluate.stable_seed('uepairs', scenario, N_r, seed)).  With N_r = 1 the stored
channels and the evaluation's own streams are used (sim.measure on Site.pbar seeded with
evaluate.stable_seed(scenario, seed), methods.topk over the 64 beams), so its efficiency is
csirs.json's coherent|1000 (gate U2) and its bounds csirs.json's (gate U3).

Statistics, per N_r, scenario (the seven non-training) and population (all eligible users; rule
R37's covered population, fixed by the stored single-antenna genie-aided rate, the same users for
every N_r): eta, the mean over seeds of the selected pair's user-mean rate over the genie-aided
pair's; the lower bound at 0.05/7, simultaneous over the seven scenarios for each N_r and
population (csirs.lin_se and effbound.lower_bound on tag 'uepairs|scenario|N_r|pop', window 0.25,
then 0.5 and 1.0 if the scan's lowest point does not pass); and per user the mean over seeds of the
selected pair's rate over the user's genie-aided pair rate (uepairs.npz).

Gates, before anything is written: U0 the eligible users and their order are the dataset's; U1a
build.py's own arithmetic (g1.narrowband_at on the paths in single precision, as the dataset was
built) reproduces the stored pbar to 1e-6 relative and C to 1e-5 bit/s/Hz; U1b with N_r = 1 the
pair builder fed build.py's single-precision departure-side path factors reproduces U1a's channels
to 1e-9 of each user's strongest beam and 1e-9 bit/s/Hz (in double precision it departs from them by
the dataset's own single-precision phase error, recorded as U1c, not gated); U2 the N_r = 1 efficiency is
csirs.json's coherent|1000 to 1e-12 in every scenario; U3 the N_r = 1 bounds on all users are
csirs.json's coherent|1000 bounds exactly.

usage:  python3 uepairs.py [--gate]    -> uepairs.json, uepairs.npz  (--gate runs U0-U3, writes nothing)
"""
import hashlib, json, math, multiprocessing, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
KEY = 'uepairs'
OUT, OUT_NPZ = os.path.join(HERE, 'uepairs.json'), os.path.join(HERE, 'uepairs.npz')
N_RS = (1, 2, 4)
CQI1 = 2 * 78 / 1024
WINDOWS = (0.25, 0.5, 1.0)
CHUNK = 1024


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def paths(name):
    """The eligible users' paths with arrival angles, in the dataset's order (gate U0)."""
    import scipy.io as sio
    import audit_direct as AD
    R = AD.raw(name)
    d = os.path.join(AD.SITES, 'scenarios', name)
    for k in ('aoa_el', 'aoa_az'):
        R[k] = np.asarray(sio.loadmat(os.path.join(d, f'{k}_t001_tx000_r000.mat'))[k], np.float64)
    elig = AD.eligible(R)
    z = np.load(os.path.join(HERE, 'data', f'{name}.npz'))
    u0 = bool(np.array_equal(elig, z['elig']))
    return {k: R[k][elig] for k in ('power', 'phase', 'delay', 'aod_el', 'aod_az', 'aoa_el', 'aoa_az')}, u0


def channels(P, alpha, n_r, dev, ref32=False):
    """pbar and C of every (b, q) pair, (n, 64 n_r) float64 numpy, pairs ordered b * n_r + q.
    ref32: the departure-side path factors from g1's own functions on the single-precision paths,
    as build.py formed them (gate U1b); otherwise every factor in double precision."""
    import torch, obs, g1
    f = obs.subband_offsets()
    n, M = P['power'].shape[0], 64
    pb, cc = np.empty((n, M * n_r)), np.empty((n, M * n_r))
    ant = torch.arange(M, device=dev, dtype=torch.float64)
    rr = torch.arange(n_r, device=dev, dtype=torch.float64)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        ok = np.isfinite(P['power'][s:e])
        g = lambda k: torch.tensor(np.where(ok, P[k][s:e], 0.0), device=dev)
        amp = torch.where(torch.tensor(ok, device=dev), 10 ** (g('power') / 20), torch.zeros(1, device=dev,
                                                                                              dtype=torch.float64))
        ph, tau = torch.deg2rad(g('phase')), g('delay')
        psi = math.pi * torch.sin(torch.deg2rad(g('aod_el'))) * torch.sin(torch.deg2rad(g('aod_az')))
        al = torch.tensor(alpha[s:e], device=dev)[:, None]
        chi = math.pi * torch.sin(torch.deg2rad(g('aoa_el'))) * torch.cos(torch.deg2rad(g('aoa_az')) - al)
        if ref32:
            q = {k: P[k][s:e].astype(np.float32) for k in ('power', 'phase', 'delay', 'aod_el', 'aod_az')}
            g32 = g1.path_gains(q['power'], q['phase'])
            tau32 = np.where(np.isfinite(q['delay']), q['delay'], 0.0)
            p32 = g1.spatial_freq(q['aod_el'], q['aod_az'])
            p32 = np.where(np.isfinite(p32), p32, 0.0)
            et = torch.tensor(np.exp(1j * np.arange(M)[None, :, None] * p32[:, None, :]),
                              device=dev).permute(0, 2, 1)                          # (B, L, 64)
        else:
            et = torch.exp(1j * psi[:, :, None] * ant[None, None, :])               # (B, L, 64)
        er = torch.exp(1j * chi[:, :, None] * rr[None, None, :])                    # (B, L, n_r)
        acc_p = torch.zeros((e - s, n_r, M), device=dev, dtype=torch.float64)
        acc_c = torch.zeros_like(acc_p)
        for fj in f:
            if ref32:
                a = torch.tensor(g32 * np.exp(-2j * np.pi * tau32 * float(fj)), device=dev,
                                 dtype=torch.complex128)                      # exact widening
            else:
                a = amp * torch.exp(1j * (ph - 2 * math.pi * tau * float(fj)))      # (B, L)
            h = torch.einsum('bl,blr,bln->brn', a, er, et)
            p = torch.abs(torch.fft.fft2(h)) ** 2 / (M * n_r)                    # (B, n_r, 64): [q, b]
            acc_p += p
            acc_c += torch.log2(1 + obs.P_TX * p / obs.SIGMA_D2)
        # pair index b * n_r + q
        pb[s:e] = (acc_p / len(f)).permute(0, 2, 1).reshape(e - s, -1).cpu().numpy()
        cc[s:e] = (acc_c / len(f)).permute(0, 2, 1).reshape(e - s, -1).cpu().numpy()
    return pb, cc


def build_reference(P):
    """build.py's arithmetic for the 64 narrow beams: g1.narrowband_at on the paths in the single
    precision the dataset was built from, then |h F|^2 with obs's codebook (gate U1a)."""
    import g1, obs
    F = obs.hierarchical_codebook()[:, obs.LAYER6_OFFSET:]
    f32 = {k: P[k].astype(np.float32) for k in ('power', 'phase', 'delay', 'aod_el', 'aod_az')}
    n = f32['power'].shape[0]
    acc_p, acc_s = np.zeros((n, 64)), np.zeros((n, 64))
    for df in obs.subband_offsets():
        h = g1.narrowband_at(f32['power'], f32['phase'], f32['aod_el'], f32['aod_az'], f32['delay'],
                             float(df), chunk=CHUNK)
        p = np.abs(h @ F) ** 2
        acc_p += p
        acc_s += np.log2(1.0 + obs.P_TX * p / obs.SIGMA_D2)
    return acc_p / obs.K_F, acc_s / obs.K_F


def _bound(task):
    import csirs, effbound
    key, selk, gse, blocks, nb, tag = task
    eta = float(selk.mean(1).mean() / gse.mean())
    rec = {'eta': eta, 'users': int(gse.size), 'blocks': int(nb)}
    se_lin = csirs.lin_se(selk, gse, blocks, nb, eta, tag + '|lin')
    for w in WINDOWS:
        try:
            lb, e_b, e_t, flips = effbound.lower_bound(selk, gse, blocks, nb, eta, se_lin, tag + '|lb',
                                                       0.05 / 7, w)
            return key, {**rec, 'se_lin': se_lin, 'eta_lb': lb, 'eta_lb_boot': e_b, 'eta_lb_t': e_t,
                         'scan_transitions': flips, 'window': w}
        except AssertionError as x:
            if "widen the window" not in str(x):
                raise
    raise SystemExit(f'{key}: no window brackets the bound')


def main():
    import torch, csirs, evaluate, protocol, sim, methods
    gate_only = '--gate' in sys.argv
    if not gate_only:
        P = json.load(open(os.path.join(HERE, 'protocol.json')))
        A = [a for a in P['analyses'] if a['key'] == KEY]
        assert len(A) == 1 and A[0]['scripts_sha256']['uepairs.py'] == sha(os.path.abspath(__file__)), \
            'uepairs.py is not the declared script'
        assert not os.path.exists(OUT) and not os.path.exists(OUT_NPZ), 'written once'
    seeds = protocol.seeds()
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))['T_PROBE']
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))['configs']['coherent|1000']['per_scenario']
    dev = sim.DEV
    t0 = time.time()
    gates = {'U0': {}, 'U1': {}, 'U2': {}, 'U3': {}}
    data, npz = {}, {}
    for name in csirs.SCEN:
        P, gates['U0'][name] = paths(name)
        site = sim.Site(name)
        n = site.n
        alpha = np.random.default_rng(evaluate.stable_seed(KEY, 'orientation', name)).uniform(0, 2 * math.pi, n)
        gse1 = site.oracle_se.double().cpu().numpy()
        keep = gse1 >= CQI1
        blocks = evaluate.block_ids(site.rx)
        nb = int(blocks.max()) + 1
        rec = {}
        for n_r in (N_RS if not gate_only else (1,)):
            pb, cc = channels(P, alpha, n_r, dev)
            if n_r == 1:
                z = np.load(os.path.join(HERE, 'data', f'{name}.npz'))
                sp, sc = z['pbar'][:, 62:].astype(np.float64), z['se'][:, 62:].astype(np.float64)
                bp, bc = build_reference(P)
                rp, rc = channels(P, alpha, 1, dev, ref32=True)
                rel = lambda x: float(np.max(np.abs(x - bp) / bp.max(1, keepdims=True)))
                gates['U1'][name] = {
                    # U1a: build.py's own arithmetic on these paths is the stored dataset
                    'U1a': bool(np.max(np.abs(bp - sp) / np.maximum(sp, 1e-300)) <= 1e-6 and
                                np.max(np.abs(bc - sc)) <= 1e-5),
                    # U1b: the pair builder on build.py's single-precision path factors is build.py
                    'U1b': bool(rel(rp) <= 1e-9 and np.max(np.abs(rc - bc)) <= 1e-9),
                    # U1c: in double precision it departs from build.py by the dataset's own
                    # single-precision phase error, recorded
                    'U1c_rel_to_strongest': rel(pb), 'U1c_rate_abs': float(np.max(np.abs(cc - bc)))}
                gates['U1'][name]['holds'] = gates['U1'][name]['U1a'] and gates['U1'][name]['U1b']
                sel = []
                for s in seeds:
                    y = sim.measure(site.pbar, torch.Generator(device=dev).manual_seed(
                        evaluate.stable_seed(name, s)))
                    d, c = methods.topk(site, y, 64, t_probe)
                    sel.append(site.se.gather(1, c[:, None])[:, 0].double().cpu().numpy())
                selk, gse = np.stack(sel), gse1
                eta = float(selk.mean(1).mean() / gse.mean())
                gates['U2'][name] = abs(eta - CS[name]['eta']) <= 1e-12
            else:
                pbt = torch.tensor(pb, device=dev)
                sel = []
                for s in seeds:
                    y = sim.measure(pbt, torch.Generator(device=dev).manual_seed(
                        evaluate.stable_seed(KEY, name, n_r, s)))
                    c = y.argmax(1).cpu().numpy()
                    sel.append(cc[np.arange(n), c])
                    del y
                selk, gse = np.stack(sel), cc.max(1)
                del pbt
                torch.cuda.empty_cache()
            rec[n_r] = (selk, gse)
            npz[f'{name}|{n_r}|eta_user'] = (selk.mean(0) / gse).astype(np.float32)
        npz[f'{name}|covered'] = keep
        data[name] = (rec, keep, blocks, nb)
        print(f'  {name}: channels and selections ({time.time() - t0:.0f} s)', flush=True)
    # U3: the N_r = 1 bounds on all users are csirs.json's, on csirs.py's own tags
    for name in csirs.SCEN:
        selk, gse = data[name][0][1]
        blocks, nb = data[name][2], data[name][3]
        eta = float(selk.mean(1).mean() / gse.mean())
        se_lin = csirs.lin_se(selk, gse, blocks, nb, eta, f'{name}|coherent|1000|lin')
        lb = csirs.lower_bound(selk, gse, blocks, nb, eta, se_lin, f'{name}|coherent|1000|lb')[0]
        gates['U3'][name] = lb == CS[name]['eta_lb']
    ok = {k: all((x['holds'] if isinstance(x, dict) else x) for x in v.values()) for k, v in gates.items()}
    print(json.dumps(ok), flush=True)
    if not all(ok.values()):
        raise SystemExit(f'gate failed: {gates}; nothing written')
    if gate_only:
        print('gates U0-U3 hold; nothing written')
        return
    tasks = []
    for name, (rec, keep, blocks, nb) in data.items():
        present = np.unique(blocks[keep])
        bk = np.searchsorted(present, blocks[keep])
        for n_r, (selk, gse) in rec.items():
            tasks.append(((name, n_r, 'all'), selk, gse, blocks, nb, f'{KEY}|{name}|{n_r}|all'))
            tasks.append(((name, n_r, 'covered'), selk[:, keep], gse[keep], bk, int(present.size),
                          f'{KEY}|{name}|{n_r}|covered'))
    with multiprocessing.get_context('spawn').Pool(min(24, len(tasks))) as pool:
        res = dict(pool.map(_bound, tasks))
    per = {}
    for (name, n_r, pop), r in res.items():
        per.setdefault(name, {}).setdefault(str(n_r), {})[pop] = r
    summary = {}
    for n_r in map(str, N_RS):
        for pop in ('all', 'covered'):
            rs = [per[nm][n_r][pop] for nm in per]
            summary[f'{n_r}|{pop}'] = {'eta_min': min(r['eta'] for r in rs),
                                       'eta_max': max(r['eta'] for r in rs),
                                       'eta_lb_min': min(r['eta_lb'] for r in rs),
                                       'eta_lb_t_min': min(r['eta_lb_t'] for r in rs),
                                       'argmin_lb': min(per, key=lambda nm: per[nm][n_r][pop]['eta_lb'])}
    out = {'rule': 'R44', 'script_sha256': sha(os.path.abspath(__file__)), 'n_r': list(N_RS),
           'seeds': list(seeds), 'alpha': 0.05 / 7, 'cqi1': CQI1, 'gate': gates, 'per_scenario': per,
           'summary': summary, 'seconds': round(time.time() - t0, 1)}
    np.savez_compressed(OUT_NPZ, **npz)
    out['npz_sha256'] = sha(OUT_NPZ)
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, v in summary.items():
        print(f'  N_r|pop {k}: eta {v["eta_min"]:.5f}-{v["eta_max"]:.5f}, bound {v["eta_lb_min"]:.5f} '
              f'({v["argmin_lb"]})')
    print(f'wrote {OUT} and {OUT_NPZ} in {out["seconds"]} s')


if __name__ == '__main__':
    main()
