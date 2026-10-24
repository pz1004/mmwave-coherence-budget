"""How often does a probe land BETWEEN two compute actions?  -> interleave.json

A cumulative charge tau_e would assume block e resumes from block e-1's
activations.  A probe in between changes the network input, and the executor
rebuilds the feature vector and recomputes from block 0.  This replays the
shipped policy and counts how often that actually happens.  Read-only.
"""
import json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# --suffix _rr replays the re-solved table; '' is the original
SUF = sys.argv[sys.argv.index('--suffix') + 1] if '--suffix' in sys.argv else ''
import sim, feats
from sim import DEV, K_MAX, N_EXIT, COMMIT_M, COMMIT_P, PROBE, COMPUTE
L6 = 126 - 64

GRID = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
TAU = np.array([0.0, *[lat['tau_restart_us'][str(i)] for i in (1, 2, 3)]])
CH = {r['component']: r['charge_us'] for r in lat['rows']}
site = sim.Site('city_2_chicago_28')
seed = 17
model = sim.load_predictor(seed)
z = np.load(os.path.join(HERE, f'tables_{seed}{SUF}.npz'))

rows = []
print(f"{'T us':>7} {'mean exits':>11} {'recv w/ >=2 computes':>21} "
      f"{'of those, any interleaved':>26}")
for T in GRID:
    tab = torch.tensor(z[f'dppi_{T}'], device=DEV, dtype=torch.long)
    kmaj = tab.dim() == 6           # a kappa-major table (sched.solve_exact)
    gen = torch.Generator(device=DEV).manual_seed(12345)
    # site.pn is ALREADY the narrow 64-codeword block (pbar[:, LAYER6_OFFSET:]),
    # so it must not be sliced by L6 a second time.
    yn = sim.measure(site.pn, gen)
    snr = feats.snr_db_t(yn)
    n = site.n
    probed = torch.zeros((n, 64), dtype=torch.bool, device=DEV)
    score = sim.spread_score().expand(n, -1).clone()
    lvl = torch.zeros(n, dtype=torch.long, device=DEV)
    k = torch.zeros(n, dtype=torch.long, device=DEV)
    fresh = torch.zeros(n, dtype=torch.long, device=DEV)
    bidx = torch.zeros(n, dtype=torch.long, device=DEV)
    bsnr = torch.full((n,), -1e9, device=DEV, dtype=torch.float64)
    msum = torch.zeros(n, device=DEV, dtype=torch.float64)
    done = torch.zeros(n, dtype=torch.bool, device=DEV)
    # probes taken since this receiver's last COMPUTE; and whether any compute
    # was ever preceded by such a probe
    since = torch.zeros(n, dtype=torch.long, device=DEV)
    ever_comp = torch.zeros(n, dtype=torch.bool, device=DEV)
    interleaved = torch.zeros(n, dtype=torch.bool, device=DEV)
    kap = torch.zeros(n, dtype=torch.long, device=DEV)    # probes at the first call
    for _ in range(K_MAX + N_EXIT + 2):
        any_p = k > 0
        b = sim.bucket(bsnr, any_p)
        c = sim.level_class(msum / k.clamp(min=1), any_p)
        act = tab[kap, lvl, k, b, c, fresh] if kmaj else tab[lvl, k, b, c, fresh]
        act = torch.where(done, torch.full_like(act, -1), act)
        done = done | (act == COMMIT_M) | (act == COMMIT_P)
        co = (act == COMPUTE) & ~done
        pr = (act == PROBE) & ~done
        if co.any():
            # a second-or-later compute with probes since the previous one
            interleaved |= co & ever_comp & (since > 0)
            ever_comp |= co
            since = torch.where(co, torch.zeros_like(since), since)
            kap = torch.where(co & (lvl == 0), k, kap)
            lvl = lvl + co.long()
            fresh = torch.where(co, torch.ones_like(fresh), fresh)
        if pr.any():
            nxt = torch.where(probed, torch.full_like(score, -1e30), score).argmax(1)
            probed[pr, nxt[pr]] = True
            ns = snr.gather(1, nxt[:, None])[:, 0]
            better = pr & (ns > bsnr)
            bidx = torch.where(better, nxt, bidx); bsnr = torch.where(better, ns, bsnr)
            msum = msum + torch.where(pr, ns, torch.zeros_like(ns))
            k = k + pr.long()
            since = since + pr.long()
            fresh = torch.where(pr, torch.zeros_like(fresh), fresh)
        if bool(done.all()):
            break
    multi = lvl >= 2
    nm = int(multi.sum())
    ni = int((interleaved & multi).sum())
    rows.append({"T_us": T, "mean_exit_level": round(float(lvl.float().mean()), 3),
                 "n": int(n), "n_multi_compute": nm,
                 "frac_multi_compute": round(nm / n, 4),
                 "n_interleaved": ni,
                 "frac_interleaved": round(ni / nm if nm else 0.0, 4)})
    print(f"{T:7d} {float(lvl.float().mean()):11.2f} {nm:10d} ({nm/n:5.1%}) "
          f"{ni:14d} ({(ni/nm if nm else 0):5.1%})")

json.dump({"site": site.name if hasattr(site, 'name') else 'city_2_chicago_28',
           "seed": seed, "controller": "dppi",
           "note": ("fraction of receivers that compute at least twice and have a "
                    "probe between two of those computes -- the case the cumulative "
                    "tau charge assumes away"),
           "rows": rows},
          open(os.path.join(HERE, f'interleave{SUF}.json'), 'w'), indent=1)
