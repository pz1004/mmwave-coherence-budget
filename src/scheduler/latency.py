#!/usr/bin/env python3
"""G3 latency for every Phase 4 decision component.  Protocol verbatim from
Phase 1: 1,000 warmup calls, 10,000 timed calls, perf_counter_ns, nearest-rank
p95, untrained Glorot weights under seeds 17/29/43, the LARGEST of the three
p95s reported, and the WHOLE decision charged -- feature construction from the
measurement buffer, the network, the exit head, and the argmax that produces a
beam index.  Nothing is free.

Two charging models are measured, because the controller's executor does not
resume:

  tau_cumulative_us  step e reuses block e-1's activations, so tau_e = sum of
                     steps 1..e.  Correct for an early-exit network that holds
                     its activations across the gap between two compute actions.
  tau_fresh_us       ONE invocation run from the feature vector to exit j with
                     nothing cached -- blocks 1..j and head j only, mirroring
                     sim.exit_logits exactly.
  tau_restart_us     sum of tau_fresh over j = 1..e: what reaching exit e costs
                     when every compute action restarts.

The executor rebuilds the feature vector and recomputes from block 0 on every
compute, because an intervening probe changes the network's input.  It therefore
pays tau_restart, and the cumulative charge understates it by 1.5x at exit 2 and
2.0x at exit 3.  Summing tau_cumulative is NOT the same quantity: tau_2 there
includes exit head 1, which a fresh pass to exit 2 never evaluates.  These are
measured rather than derived for that reason.

Because COMPUTE advances the exit index by exactly +1 and never decreases it,
each exit is reached at most once on any trajectory, so tau_restart[e] is a
function of e alone -- the backward induction stays exact.

Phase 1 measured the same network monolithically; all of them are reported.
"""
from __future__ import annotations
import argparse, json, math, os, platform, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
import models as M
from calibrate import nearest_rank_p95, timed
import obs

WARMUP, TIMED = 1000, 10000
SEEDS = (17, 29, 43)
CLIP_LO, CLIP_HI = -40.0, 60.0


def build(profile, device, threads):
    torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    dev = torch.device(device)
    rng = np.random.default_rng(881)
    # receiver-side state: a 64-slot band-averaged power buffer, 16 probed
    y = (10 ** (rng.uniform(-4, 5, 64) / 10)) * obs.SIGMA_D2
    meas = np.zeros(64, bool); meas[obs.spread_order()[:16]] = True
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)
    rows = []

    for seed in SEEDS:
        def net(m):
            M.glorot_(m, seed)
            return m.eval().to(dev)
        ee = net(M.EarlyExitMLP())
        fcs = {k: net(M.BeamMLP(n_in=k)) for k in (4, 8, 16, 32, 64)}
        cnn = net(M.BeamCNN())
        spread = obs.spread_order()
        table = np.zeros((4, 25, 40, 10), np.int8)      # the DP threshold table
        state = [1, 9, 0.0]

        # ---- receiver-side front end -------------------------------------
        # naive: rebuild the 194-vector from the power buffer on every decision
        def feat_build():
            sd = np.clip(10 * np.log10(np.maximum(obs.P_TX * y, 1e-30) / obs.SIGMA_D2),
                         CLIP_LO, CLIP_HI)
            sm = np.where(meas, sd, 0.0)
            b = int(np.argmax(np.where(meas, sd, -np.inf)))
            f = np.empty(194, np.float32)
            f[:64] = sm / 20.0; f[64:128] = meas
            f[128:192] = 0.0; f[128 + b] = 1.0
            f[192] = meas.sum() / 64.0; f[193] = sd[b] / 20.0
            return torch.from_numpy(f).unsqueeze(0).to(dev)

        # optimised: the buffer is maintained in dB as each probe lands, so the
        # decision only refreshes the one-hot and the two scalars.  This is what
        # a receiver that is already tracking its running best actually does.
        sdb = np.clip(10 * np.log10(np.maximum(obs.P_TX * y, 1e-30) / obs.SIGMA_D2),
                      CLIP_LO, CLIP_HI)
        fbuf = np.zeros(194, np.float32)
        fbuf[:64] = np.where(meas, sdb, 0.0) / 20.0
        fbuf[64:128] = meas
        ft = torch.from_numpy(fbuf).unsqueeze(0).to(dev)
        st = [int(np.argmax(np.where(meas, sdb, -np.inf))), 16, 0]

        def feat_build_opt():
            fbuf[128 + st[2]] = 0.0
            fbuf[128 + st[0]] = 1.0
            fbuf[192] = st[1] * 0.015625
            fbuf[193] = sdb[st[0]] * 0.05
            st[2] = st[0]
            return ft

        # per-probe bookkeeping, charged to EVERY method that probes.  It also
        # maintains the running sum the controller's second state variable needs,
        # so that variable costs the scheduler nothing the baselines do not pay.
        acc = [0.0]

        def probe_update():
            c = spread[st[1] % 64]
            v = 10.0 * math.log10(max(obs.P_TX * float(y[c]), 1e-30) / obs.SIGMA_D2)
            fbuf[c] = v * 0.05
            fbuf[64 + c] = 1.0
            acc[0] += v
            return v > sdb[st[0]]

        def ee_net1():
            h = ee.blocks[0](ft)
            b = int(torch.argmax(ee.exits[0](h)[0])); sync(); return h, b
        h1, _ = ee_net1()

        def ee_net2():
            h = ee.blocks[1](h1)
            b = int(torch.argmax(ee.exits[1](h)[0])); sync(); return h, b
        h2, _ = ee_net2()

        def ee_net3():
            b = int(torch.argmax(ee.exits[2](ee.blocks[2](h2))[0])); sync(); return b

        # One fresh invocation to exit j: no cached activation, blocks 1..j and
        # head j only.  This mirrors sim.exit_logits(model, f, j) exactly, which
        # is what the executor calls -- note it evaluates ONLY head j, so this is
        # not the sum of the incremental steps.
        def mk_fresh(j):
            def f():
                x = feat_build_opt()
                for i in range(j):
                    x = ee.blocks[i](x)
                b = int(torch.argmax(ee.exits[j - 1](x)[0])); sync(); return b
            return f

        def ee_step1():
            h = ee.blocks[0](feat_build_opt())
            b = int(torch.argmax(ee.exits[0](h)[0])); sync(); return b

        def mk_fc(k):
            idx = spread[:k]
            def f():
                x = torch.from_numpy((sdb[idx] * 0.05).astype(np.float32)).unsqueeze(0).to(dev)
                b = int(torch.argmax(fcs[k](x)[0])); sync(); return b
            return f

        def cnn_dec():
            x = torch.from_numpy(np.stack([fbuf[:64], fbuf[64:128]])).unsqueeze(0).to(dev)
            b = int(torch.argmax(cnn(x)[0])); sync(); return b

        def sweep_select():
            return int(np.argmax(np.where(meas, y, -np.inf)))

        # the controller: the running best is already maintained, so a consult
        # is a bucket computation and one table index -- nothing else.
        def lookup():
            bkt = int((sdb[st[0]] + 48.0) * 0.3333333333) + 1
            lvl = int((acc[0] / st[1] + 40.0) * 0.1) + 1
            return int(table[st[2] & 3, st[1] & 24, bkt if bkt < 40 else 39,
                             lvl if lvl < 10 else 9])

        def np_test():
            return (float(y[spread[st[1] % 64]]) > 2.5 * obs.SIGMA_P2) and (st[1] < 12)

        comps = [("feat_build", feat_build, "194-vector rebuilt from the power buffer"),
                 ("feat_build_opt", feat_build_opt, "194-vector refreshed incrementally"),
                 ("probe_update", probe_update, "per-probe dB conversion + running best"),
                 ("ee_net1", lambda: ee_net1()[1], "block 1 + exit 1 + argmax, network only"),
                 ("ee_net2", lambda: ee_net2()[1], "block 2 + exit 2 + argmax (reuses block 1)"),
                 ("ee_net3", ee_net3, "block 3 + exit 3 + argmax (reuses block 2)"),
                 ("ee_step1", ee_step1, "front end + block 1 + exit 1 + argmax"),
                 ("ee_fresh1", mk_fresh(1), "fresh invocation: front end + blocks 1..1 + exit 1 + argmax"),
                 ("ee_fresh2", mk_fresh(2), "fresh invocation: front end + blocks 1..2 + exit 2 + argmax"),
                 ("ee_fresh3", mk_fresh(3), "fresh invocation: front end + blocks 1..3 + exit 3 + argmax"),
                 ("fc_k4", mk_fc(4), "BeamMLP(4) full decision"),
                 ("fc_k8", mk_fc(8), "BeamMLP(8) full decision"),
                 ("fc_k16", mk_fc(16), "BeamMLP(16) full decision"),
                 ("fc_k32", mk_fc(32), "BeamMLP(32) full decision"),
                 ("fc_k64", mk_fc(64), "BeamMLP(64) full decision"),
                 ("cnn", cnn_dec, "BeamCNN over (2,64) full decision"),
                 ("sweep_select", sweep_select, "argmax over the measurement buffer"),
                 ("lookup", lookup, "DP threshold-table consult (the controller)"),
                 ("np_test", np_test, "Zhang Neyman-Pearson slot decision")]
        # Contention guard.  G3 says nothing about what else may be running on
        # the machine, and it must: a repeat of this sweep taken while other work
        # was running returned tau_1 = 21.10 us against 13.76-13.91 us on three
        # quiet repeats -- a 53% inflation that would have moved every T*.  Each
        # component is therefore timed TWICE, and the two passes must agree.
        for name, fn, desc in comps:
            a = nearest_rank_p95(timed(fn, WARMUP, TIMED)) / 1000.0
            b = nearest_rank_p95(timed(fn, WARMUP // 4, max(TIMED // 4, 200))) / 1000.0
            rows.append({"seed": seed, "component": name, "charge_us": a,
                         "recheck_us": b, "drift": abs(b - a) / max(a, 1e-9),
                         "desc": desc})
    agg, drift = {}, {}
    for r in rows:
        agg.setdefault(r['component'], []).append(r['charge_us'])
        drift.setdefault(r['component'], []).append(r['drift'])
    worst = max((max(v), k) for k, v in drift.items())
    out = {"env": {"profile": profile, "device": device, "threads": threads,
                   "torch": torch.__version__, "python": platform.python_version(),
                   "cpu": platform.processor(),
                   "utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
           "protocol": "G3 verbatim; largest p95 over seeds 17/29/43",
           "n_warmup": WARMUP, "n_timed": TIMED,
           "contention": {"loadavg": open('/proc/loadavg').read().split()[:3],
                          "worst_recheck_drift": round(worst[0], 4),
                          "worst_component": worst[1],
                          "contended": bool(worst[0] > 0.10),
                          "rule": "each component timed twice; >10% drift between "
                                  "passes means the machine was not quiet"},
           "rows": [{"component": k, "charge_us": round(max(v), 3),
                     "per_seed_us": [round(x, 3) for x in v],
                     "desc": [c[2] for c in comps if c[0] == k][0]}
                    for k, v in agg.items()]}
    t = {r['component']: r['charge_us'] for r in out['rows']}
    # Declared accounting rule, Phase 2's min(conservative, optimised) applied
    # to the front end: tau_e = front end + network steps 1..e.  The per-probe
    # bookkeeping is NOT in tau; it is charged per probe to every method.
    fe = min(t['feat_build'], t['feat_build_opt'])
    out['front_end_us'] = {"naive": t['feat_build'], "optimised": t['feat_build_opt'],
                           "charged": round(fe, 3)}
    out['tau_cumulative_us'] = {
        "1": round(fe + t['ee_net1'], 3),
        "2": round(fe + t['ee_net1'] + t['ee_net2'], 3),
        "3": round(fe + t['ee_net1'] + t['ee_net2'] + t['ee_net3'], 3)}
    # What the executor actually pays.  Each compute action is a fresh pass, so
    # reaching exit e costs the sum of the fresh invocations 1..e.  Path
    # independent, because the exit index only ever advances by one.
    out['tau_fresh_us'] = {str(j): round(t[f'ee_fresh{j}'], 3) for j in (1, 2, 3)}
    out['tau_restart_us'] = {
        str(e): round(sum(t[f'ee_fresh{j}'] for j in range(1, e + 1)), 3)
        for e in (1, 2, 3)}
    out['tau_network_only_us'] = {
        "1": round(t['ee_net1'], 3), "2": round(t['ee_net1'] + t['ee_net2'], 3),
        "3": round(t['ee_net1'] + t['ee_net2'] + t['ee_net3'], 3)}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=1); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    o = build(a.profile, a.device, a.threads)
    json.dump(o, open(a.out, 'w'), indent=1)
    c = o['contention']
    print(f"  loadavg {c['loadavg']}  worst recheck drift {c['worst_recheck_drift']:.1%} "
          f"on {c['worst_component']}  -> {'CONTENDED' if c['contended'] else 'quiet'}")
    for r in o['rows']:
        print(f"  {r['component']:14s} {r['charge_us']:10.3f} us   {r['per_seed_us']}")
    print("  tau cumulative:", o['tau_cumulative_us'])
    print("  tau fresh     :", o['tau_fresh_us'])
    print("  tau restart   :", o['tau_restart_us'], " <- what the executor pays")
