#!/usr/bin/env python3
"""Timing declared with the ten-seed analysis (protocol.json): the policy lookup on the
real table, and a NumPy profile of the networks the controller schedules.

  --lookup   latency.py times the consult on a stand-in (4,25,40,10) table
             with made-up indexing.  The controller consults
             a (4,65,40,10,2) int8 table, 208 kB, once per decision -- K+e+1
             times per receiver -- so its charge weighs as much as tau_1.  Here
             the consult is timed on the real headline tables of seeds 17/29/43,
             computing the bucket and class from the measured SNRs exactly as the
             executor does, over states replayed in order from the deployed
             controller's own trajectories on the validation city.  Timing is
             BATCHED, as the ten-seed analysis fixed in advance: each timed call
             is a chunk of 100 consecutive consults, so the timer's overhead does
             not dominate a sub-microsecond operation.  G3's shape is kept -- 1,000
             warmup and 10,000 timed calls, nearest-rank p95 per seed, the largest
             of the three seeds -- and every chunk is timed twice for the
             contention rule.  The consult is ALSO timed one per call, as G3 times
             every other component; that per-call figure is the one CHARGED
             (protocol.json records the deviation from the declared wording and
             why).

  --numpy    A descriptive profile: the same early-exit passes and FC networks
             as latency.py, in NumPy float32 with identical weights and argmax,
             timed per call under G3.  It measures how much of the PyTorch
             charge is the framework, and gives the rho sweep a measured point.
             The CNN is not reimplemented.

Run pinned to one P-core on a quiet machine:
  OMP_NUM_THREADS=1 taskset -c 2 python3 retime.py --lookup --out lat_lookup_real.json
  OMP_NUM_THREADS=1 taskset -c 2 python3 retime.py --numpy  --out lat_cpu1t_pcore_numpy.json
"""
from __future__ import annotations
import argparse, json, math, os, platform, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
from calibrate import nearest_rank_p95, timed

SEEDS = (17, 29, 43)
WARMUP, TIMED, CHUNK = 1000, 10000, 100
DEADLINE = 1000                     # the headline table the consults replay (1 ms)


def env(profile):
    gov = ''
    try:
        gov = open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()
    except OSError:
        pass
    return {"profile": profile, "python": platform.python_version(), "numpy": np.__version__,
            "governor": gov, "affinity": sorted(os.sched_getaffinity(0)),
            "utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def contention(rows):
    worst = max((r['drift'], r['component']) for r in rows)
    return {"loadavg": open('/proc/loadavg').read().split()[:3],
            "worst_recheck_drift": round(worst[0], 4), "worst_component": worst[1],
            "contended": bool(worst[0] > 0.10),
            "rule": "each component timed twice; >10% drift between passes means "
                    "the machine was not quiet"}


def replay_states(seed, n_rx=2000):
    """States the deployed headline controller consults the table in, recorded
    in decision order on the validation city (GPU)."""
    import torch
    import sim, methods, evaluate
    lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
    q = evaluate.charges(lat)
    site = sim.Site('city_1_losangeles_28')
    tab = np.load(os.path.join(HERE, f'tables_{seed}.npz'))[f'dppi_{DEADLINE}']
    gen = torch.Generator(device=sim.DEV).manual_seed(evaluate.stable_seed(site.name, seed))
    y = sim.measure(site.pbar, gen)
    ee = sim.load_predictor(seed)
    tr = []
    methods.scheduled(site, y, ee, tab, q['TAU'], q['T_PROBE'], q['C_LOOK'], trace=tr)
    # one receiver's decisions in order, then the next: the executor's access pattern
    live = np.stack([m for m, _ in tr])                  # (steps, n)
    S = np.stack([s for _, s in tr])                     # (steps, n, 5)
    flat = np.concatenate([S[live[:, r], r] for r in range(min(n_rx, live.shape[1]))])
    return tab, flat


def time_lookup(out_path):
    import sim
    B_LO, B_W, NB, C_LO, C_W, NC = sim.B_LO, sim.B_W, sim.NB, sim.C_LO, sim.C_W, sim.NC
    rows = []
    for seed in SEEDS:
        tab, st = replay_states(seed)
        lv = st[:, 0].astype(int).tolist(); kk = st[:, 1].astype(int).tolist()
        fr = st[:, 2].astype(int).tolist(); best = st[:, 3].tolist(); mean = st[:, 4].tolist()
        n = len(lv)
        pos = [0]

        def consult(i):
            # what the executor computes per decision: bucket, class, 5-D index
            k = kk[i]
            if k == 0:
                b = c = 0
            else:
                b = int((best[i] - B_LO) // B_W) + 1
                b = 1 if b < 1 else (NB - 1 if b > NB - 1 else b)
                c = int((mean[i] - C_LO) // C_W) + 1
                c = 1 if c < 1 else (NC - 1 if c > NC - 1 else c)
            return tab[lv[i], k, b, c, fr[i]]

        def chunk():
            p = pos[0]
            for j in range(CHUNK):
                consult((p + j) % n)
            pos[0] = (p + CHUNK) % n

        a = nearest_rank_p95(timed(chunk, WARMUP, TIMED)) / 1000.0 / CHUNK
        b = nearest_rank_p95(timed(chunk, WARMUP // 4, TIMED // 4)) / 1000.0 / CHUNK

        # The same consult timed one per call, the way G3 times every other
        # component -- timer overhead included.  Both figures are recorded, so
        # the effect of the batching fixed in advance is visible.
        def one():
            p = pos[0]
            consult(p)
            pos[0] = (p + 1) % n
        pc = nearest_rank_p95(timed(one, WARMUP, TIMED)) / 1000.0
        pc2 = nearest_rank_p95(timed(one, WARMUP // 4, TIMED // 4)) / 1000.0
        rows.append({"seed": seed, "component": "lookup_real", "charge_us": a, "recheck_us": b,
                     "drift": max(abs(b - a) / a, abs(pc2 - pc) / pc),
                     "percall_p95_us": pc, "percall_recheck_us": pc2, "n_states": n,
                     "table_shape": list(tab.shape), "table_bytes": int(tab.nbytes)})
        print(f"  seed {seed}: {a:.4f} us per consult (recheck {b:.4f}), {n} replayed "
              f"states, table {tab.shape} {tab.nbytes} B", flush=True)
    # CHARGED: the per-call figure.  The ten-seed analysis named batched timing, but
    # the two diverge in opposite directions from the stand-in's 0.261 us, and per
    # call is how G3 times every other component, including the other
    # sub-microsecond ones (probe bookkeeping, the feature refresh, the
    # Neyman-Pearson test).  Charging the batched figure would give the
    # controller's consult a timing rule nothing else gets, and in its favour.
    # protocol.json records the deviation.
    out = {"env": env('cpu-1t-pcore'),
           "method": ("real (4,65,40,10,2) tables of seeds 17/29/43, consults replayed in "
                      "order from the deployed controller's trajectories on the validation "
                      "city; nearest-rank p95 per seed, largest of three seeds"),
           "rows": rows, "contention": contention(rows),
           "percall_us": round(max(r['percall_p95_us'] for r in rows), 3),
           "batched_us": round(max(r['charge_us'] for r in rows), 3),
           "charged_rule": "per call, as G3 times every other component"}
    out["charge_us"] = out["percall_us"]
    json.dump(out, open(out_path, 'w'), indent=1)
    print(f"lookup_real charged {out['charge_us']} us per call; batched {out['batched_us']} us "
          f"(stand-in timed 0.261)  "
          f"contention {out['contention']}")


def time_numpy(out_path):
    import torch
    import models as M
    import obs
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(881)
    y = (10 ** (rng.uniform(-4, 5, 64) / 10)) * obs.SIGMA_D2
    meas = np.zeros(64, bool); meas[obs.spread_order()[:16]] = True
    sdb = np.clip(10 * np.log10(np.maximum(obs.P_TX * y, 1e-30) / obs.SIGMA_D2), -40.0, 60.0)
    fbuf = np.zeros(194, np.float32)
    fbuf[:64] = np.where(meas, sdb, 0.0) / 20.0
    fbuf[64:128] = meas
    st = [int(np.argmax(np.where(meas, sdb, -np.inf))), 16, 0]
    spread = obs.spread_order()

    def feat_build_opt():
        fbuf[128 + st[2]] = 0.0
        fbuf[128 + st[0]] = 1.0
        fbuf[192] = st[1] * 0.015625
        fbuf[193] = sdb[st[0]] * 0.05
        st[2] = st[0]
        return fbuf

    rows = []
    for seed in SEEDS:
        ee = M.EarlyExitMLP(); M.glorot_(ee, seed); ee.eval()
        W = [b[0].weight.numpy().T.copy() for b in ee.blocks]
        bs = [b[0].bias.numpy().copy() for b in ee.blocks]
        H = [e.weight.numpy().T.copy() for e in ee.exits]
        hb = [e.bias.numpy().copy() for e in ee.exits]
        fcs = {}
        for k in (4, 8, 16, 32, 64):
            m = M.BeamMLP(n_in=k); M.glorot_(m, seed); m.eval()
            lin = [l for l in m.modules() if isinstance(l, torch.nn.Linear)]
            fcs[k] = [(l.weight.numpy().T.copy(), l.bias.numpy().copy()) for l in lin]
            # identical output to the PyTorch module, or the timing is of a different network
            x = torch.from_numpy((sdb[spread[:k]] * 0.05).astype(np.float32)).unsqueeze(0)
            ref = int(torch.argmax(m(x)[0]))
            h = x.numpy()[0]
            for i, (w_, b_) in enumerate(fcs[k]):
                h = h @ w_ + b_
                if i < len(fcs[k]) - 1:
                    h = np.maximum(h, 0.0)
            assert int(np.argmax(h)) == ref, f"NumPy FC({k}) disagrees with PyTorch"

        def fresh(j):
            def f():
                x = feat_build_opt()
                for i in range(j):
                    x = np.maximum(x @ W[i] + bs[i], 0.0)
                return int(np.argmax(x @ H[j - 1] + hb[j - 1]))
            return f
        h1 = np.maximum(fbuf @ W[0] + bs[0], 0.0)
        h2 = np.maximum(h1 @ W[1] + bs[1], 0.0)
        comps = [("feat_build_opt", feat_build_opt),
                 ("ee_net1", lambda: int(np.argmax(np.maximum(fbuf @ W[0] + bs[0], 0.0) @ H[0] + hb[0]))),
                 ("ee_net2", lambda: int(np.argmax(np.maximum(h1 @ W[1] + bs[1], 0.0) @ H[1] + hb[1]))),
                 ("ee_net3", lambda: int(np.argmax(np.maximum(h2 @ W[2] + bs[2], 0.0) @ H[2] + hb[2]))),
                 ("ee_fresh1", fresh(1)), ("ee_fresh2", fresh(2)), ("ee_fresh3", fresh(3))]
        for k in (4, 8, 16, 32, 64):
            idx = spread[:k]

            def fc(k=k, idx=idx):
                h = (sdb[idx] * 0.05).astype(np.float32)
                for i, (w_, b_) in enumerate(fcs[k]):
                    h = h @ w_ + b_
                    if i < len(fcs[k]) - 1:
                        h = np.maximum(h, 0.0)
                return int(np.argmax(h))
            comps.append((f"fc_k{k}", fc))
        # the PyTorch fresh passes, for the agreement check below
        with torch.no_grad():
            t_in = torch.from_numpy(feat_build_opt().copy()).unsqueeze(0)
            for j in (1, 2, 3):
                x = t_in
                for i in range(j):
                    x = ee.blocks[i](x)
                assert int(torch.argmax(ee.exits[j - 1](x)[0])) == fresh(j)(), \
                    f"NumPy exit {j} disagrees with PyTorch"
        for name, fn in comps:
            a = nearest_rank_p95(timed(fn, WARMUP, TIMED)) / 1000.0
            b = nearest_rank_p95(timed(fn, WARMUP // 4, max(TIMED // 4, 200))) / 1000.0
            rows.append({"seed": seed, "component": name, "charge_us": a,
                         "recheck_us": b, "drift": abs(b - a) / max(a, 1e-9)})
    agg = {}
    for r in rows:
        agg.setdefault(r['component'], []).append(r['charge_us'])
    t = {k: round(max(v), 3) for k, v in agg.items()}
    out = {"env": env('cpu-1t-pcore-numpy'),
           "protocol": "G3 verbatim; largest p95 over seeds 17/29/43; NumPy float32, "
                       "identical weights and argmax to the PyTorch profile",
           "rows": [{"component": k, "charge_us": v, "per_seed_us": [round(x, 3) for x in agg[k]]}
                    for k, v in t.items()],
           "contention": contention(rows),
           "tau_fresh_us": {str(j): t[f'ee_fresh{j}'] for j in (1, 2, 3)},
           "tau_restart_us": {str(e): round(sum(t[f'ee_fresh{j}'] for j in range(1, e + 1)), 3)
                              for e in (1, 2, 3)}}
    pc = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
    out['rho_vs_pcore'] = round(out['tau_restart_us']['3'] / pc['tau_restart_us']['3'], 3)
    json.dump(out, open(out_path, 'w'), indent=1)
    for k, v in t.items():
        print(f"  {k:15s} {v:8.3f} us")
    print(f"tau restart {out['tau_restart_us']}  rho vs PyTorch P-core {out['rho_vs_pcore']}  "
          f"contention {out['contention']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lookup', action='store_true')
    ap.add_argument('--numpy', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    time_lookup(a.out) if a.lookup else time_numpy(a.out)
