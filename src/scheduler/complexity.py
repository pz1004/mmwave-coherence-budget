#!/usr/bin/env python3
"""What the exact program costs against the myopic and blinkered rules.

The question is whether the dynamic program is worth its complexity when
a blinkered rule recovers most of its value.  The three are built the same way
-- one estimated model (Monte-Carlo rollouts on the fit city, then on-policy
refinement), one backward sweep per deadline -- and deployed the same way: a
(4, 65, 40, 10, 2) int8 table consulted once per decision.  So the run-time
footprint is identical by construction, and what differs is the offline
backup.  This script measures that, and the estimation it shares, without
touching any committed artifact: it reads tables_17.npz and writes only
complexity.json.  Descriptive; no reading rule consults it.

scheduler.json's "seconds" (2192.3) is the wall time of the whole sched.py
run -- every rollout and every solve -- not the solve time; this file
supplies the split.

usage: python3 complexity.py [--seed 17] [--reps 5] [--out complexity.json]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, sched


def gpu_util():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                   '--format=csv,noheader,nounits'], capture_output=True,
                                  text=True).stdout.split()[0])
    except Exception:
        return None


def timed(fn, reps, sync):
    fn(); sync()                                   # one warm-up
    out = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); out.append(time.perf_counter() - t0)
    return out


def solvers(TAU, T_PROBE, C_LOOK):
    TAU0 = np.zeros_like(TAU)
    return {
        'dp': lambda pr, T: sched.solve(*pr, TAU, float(T), T_PROBE, C_LOOK),
        'noc': lambda pr, T: sched.solve(*pr, TAU, float(T), T_PROBE, C_LOOK,
                                         allow_compute=False),
        'blind': lambda pr, T: sched.solve(*pr, TAU0, float(T), T_PROBE, C_LOOK),
        'myopic': lambda pr, T: sched.solve_osla(*pr, TAU, float(T), T_PROBE, C_LOOK),
        'blinkered': lambda pr, T: sched.solve_blinkered(*pr, TAU, float(T), T_PROBE, C_LOOK)}


def bench(pr, GRID, S, reps, dev):
    sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
    res = {}
    for key, fn in S.items():
        per = {}
        for T in GRID:
            ts = timed(lambda: fn(pr, T), reps, sync)
            per[str(T)] = {"median_s": float(np.median(ts)), "max_s": float(np.max(ts))}
        res[key] = {"per_T": per,
                    "total_median_s": float(sum(v['median_s'] for v in per.values()))}
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--out', default='complexity.json')
    a = ap.parse_args()
    lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
    tc = lat['tau_restart_us']
    TAU = np.array([0.0, tc['1'], tc['2'], tc['3']])
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    C_LOOK = sim.lookup_charge(lat)
    T_PROBE = sim.obs.T_P_US + ch['probe_update']
    GRID = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    SC = json.load(open(os.path.join(HERE, 'scheduler.json')))
    LK = json.load(open(os.path.join(HERE, 'lat_lookup_real.json')))
    tabs = np.load(os.path.join(HERE, f'tables_{a.seed}.npz'))
    rec = {"seed": a.seed, "reps": a.reps, "gpu_util_before": gpu_util(),
           "fit": sched.FIT, "grid_us": GRID}
    # ---- estimation, shared by every rule: the rollouts sched.py runs -------
    site = sim.Site(sched.FIT)
    model = sim.load_predictor(a.seed)
    sync = torch.cuda.synchronize if sim.DEV.type == 'cuda' else (lambda: None)
    t0 = time.perf_counter()
    off = sched.explore(site, model, a.seed, TAU, reorder=True); sync()
    t_off = time.perf_counter() - t0
    pr = sched.primitives(off)[:3]
    A0 = tabs[f'dppi_{GRID[-1]}']
    t0 = time.perf_counter()
    on = sched.explore(site, model, a.seed, TAU, table=A0, eps=sched.EPS,
                       reps=sched.ON_REPS, reorder=True); sync()
    t_on = time.perf_counter() - t0
    # sched.py, per seed: 2 off-policy explores for the base variants, one per
    # compute-action variant, and one on-policy explore per (variant, deadline)
    n_base, n_new = 7, 5
    n_off, n_on = 2 + n_new, (n_base + n_new) * len(GRID)
    n_solve = 2 * (n_base + n_new) * len(GRID)
    rec["estimation"] = {
        "off_policy_explore_s": t_off, "off_policy_replicates": sched.REPLICATES,
        "on_policy_explore_s": t_on, "on_policy_replicates": sched.ON_REPS,
        "per_seed_explores": {"off_policy": n_off, "on_policy": n_on},
        "per_seed_solves": n_solve,
        "seeds_in_sched_run": len(SC['seeds']),
        "projected_estimation_s_per_seed": n_off * t_off + n_on * t_on}
    S = solvers(TAU, T_PROBE, C_LOOK)
    rec["solve_gpu"] = bench(pr, GRID, S, a.reps, sim.DEV)
    rec["solve_gpu_recheck"] = bench(pr, GRID, S, a.reps, sim.DEV)
    rec["gpu_util_after"] = gpu_util()
    # ---- the same solves on one pinned CPU core (the cpu-1t-pcore profile) --
    cpu = torch.device('cpu')
    aff = os.sched_getaffinity(0); thr = torch.get_num_threads()
    os.sched_setaffinity(0, {2}); torch.set_num_threads(1)
    dev0 = sched.DEV
    sched.DEV = cpu
    try:
        prc = tuple(x.to(cpu) for x in pr)
        rec["solve_cpu1t"] = bench(prc, GRID, S, a.reps, cpu)
    finally:
        sched.DEV = dev0
        os.sched_setaffinity(0, aff); torch.set_num_threads(thr)
    _, A = S['dp'](pr, GRID[0])
    nb = int(A.cpu().numpy().nbytes)
    assert all(s['table_bytes'] == nb for s in SC['seeds']), "table size differs from sched.py's"
    drift = max(abs(rec['solve_gpu'][k]['total_median_s'] / rec['solve_gpu_recheck'][k]['total_median_s'] - 1)
                for k in S)
    rec["deployed"] = {"table_shape": list(A.shape), "table_bytes": nb,
                       "consult_us_per_call": LK['percall_us'], "consult_us_batched": LK['batched_us'],
                       "same_for": sorted(S)}
    rec["gpu_two_pass_drift"] = drift
    rec["sched_run_seconds"] = SC['seconds']
    rec["note"] = ("scheduler.json 'seconds' is the whole sched.py wall time: every explore "
                   "and every solve, all seeds; it is not solve time")
    json.dump(rec, open(os.path.join(HERE, a.out), 'w'), indent=1)
    for k in S:
        print(f"{k:10s} GPU {rec['solve_gpu'][k]['total_median_s']:7.3f} s / 12 deadlines   "
              f"CPU-1t {rec['solve_cpu1t'][k]['total_median_s']:7.3f} s")
    print(f"explore: off-policy {t_off:.1f}s ({sched.REPLICATES} reps), on-policy {t_on:.1f}s "
          f"({sched.ON_REPS} reps); projected estimation per seed "
          f"{rec['estimation']['projected_estimation_s_per_seed']:.0f}s; drift {drift:.3f}")
