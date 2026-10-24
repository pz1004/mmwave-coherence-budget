#!/usr/bin/env python3
"""Phase 1 latency atlas: frozen-protocol per-decision inference calibration.

Protocol (G3, candidate-specification addendum, unchanged):
  * batch one, per-decision operation, not per-epoch training cost
  * 1,000 untimed warmup calls, then 10,000 timed calls
  * time.perf_counter_ns
  * nearest-rank empirical 95th percentile: sorted observation ceil(0.95 n)
  * untrained Glorot weights under seeds 17, 29, 43; deterministic synthetic
    observation-valid inputs under seed 881
  * the reported charge is the LARGEST of the three seeds' 95th percentiles
  * every operation the method actually performs is charged: feature
    encoding, buffer update, beam selection, comparison, action emission.
    Nothing is free.

Documented deviation: components whose median call exceeds 200 us use fewer
warmup and timed calls, because 10,000 calls of a 100 ms network does not
finish.  The reduced counts are recorded per row in n_warmup / n_timed and
the nearest-rank rule is applied at the reduced n.  No other deviation.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WARMUP_FULL, TIMED_FULL = 1_000, 10_000
SEEDS = (17, 29, 43)
INPUT_SEED = 881
SIGMA_D2 = 1.9953e-12        # G1 data-noise power, W
SIGMA_P2 = SIGMA_D2 / 1000.  # G1 pilot noise after 1000-symbol despreading


def nearest_rank_p95(samples: np.ndarray) -> int:
    """Nearest-rank 95th percentile: sorted observation number ceil(0.95 n)."""
    s = np.sort(samples)
    return int(s[int(math.ceil(0.95 * s.size)) - 1])


def timed(fn, n_warmup: int, n_timed: int) -> np.ndarray:
    for _ in range(n_warmup):
        fn()
    out = np.empty(n_timed, dtype=np.int64)
    pc = time.perf_counter_ns
    for i in range(n_timed):
        t0 = pc()
        fn()
        out[i] = pc() - t0
    return out


def budget(fn) -> tuple:
    """Probe the call cost, then pick warmup/timed counts that terminate.

    The probe itself is bounded: for a network costing a second a call we
    cannot afford twenty probe calls either.
    """
    fn(); fn()
    probe, spent = [], 0
    while len(probe) < 15 and spent < 500_000_000:      # 0.5 s of probing
        d = _one(fn)
        probe.append(d)
        spent += d
    med = float(np.median(probe))
    if med < 200_000:            # < 200 us
        return WARMUP_FULL, TIMED_FULL
    if med < 2_000_000:          # < 2 ms
        return 200, 2_000
    if med < 20_000_000:         # < 20 ms
        return 50, 500
    if med < 200_000_000:        # < 200 ms
        return 20, 200
    return 5, 50


def _one(fn) -> int:
    t0 = time.perf_counter_ns()
    fn()
    return time.perf_counter_ns() - t0


# --------------------------------------------------------------- inputs ----

def synthetic(rng: np.random.Generator):
    """Deterministic observation-valid synthetic state under seed 881."""
    h = (rng.standard_normal(64) + 1j * rng.standard_normal(64)) / math.sqrt(2)
    a = np.fft.fft(h) / 8.0                       # DFT codebook responses
    eps = (rng.standard_normal(64) + 1j * rng.standard_normal(64))
    eps *= math.sqrt(SIGMA_P2 / 2)
    z = a + eps                                   # z_j = sqrt(P) a_j + eps_j
    return z.astype(np.complex128)


# ------------------------------------------------------------ components ----

_BUILD_CACHE = {}


def build(device: str, seed: int):
    """Return [(name, callable, provenance, note)] for one weight seed."""
    key = (device, seed)
    if key not in _BUILD_CACHE:
        _BUILD_CACHE[key] = _build(device, seed)
    return _BUILD_CACHE[key]


def _build(device: str, seed: int):
    import torch
    import torch.nn as nn
    import models as M

    torch.set_grad_enabled(False)
    dev = torch.device(device)
    rng = np.random.default_rng(INPUT_SEED)

    z = synthetic(rng)                                  # 64 measured phasors
    zs = z / math.sqrt(SIGMA_D2)                        # G1 input scaling
    z_t = torch.tensor(np.stack([zs.real, zs.imag], -1),
                       dtype=torch.float32).unsqueeze(0).to(dev)   # (1,64,2)
    aps = torch.tensor(np.abs(zs) ** 2, dtype=torch.float32
                       ).reshape(1, 1, 64).to(dev)
    csi2 = z_t.permute(0, 2, 1).contiguous()            # (1,2,64)
    hist = torch.tensor(rng.standard_normal((1, 8, 64)),
                        dtype=torch.float32).to(dev)
    img = torch.tensor(rng.standard_normal((1, 3, 224, 224)),
                       dtype=torch.float32).to(dev)

    # DFT codebook, for turning a predicted response into a beam index
    n = np.arange(64)
    F = np.exp(2j * np.pi * np.outer(n, n) / 64) / 8.0
    F_t = torch.tensor(np.stack([F.real, F.imag], -1),
                       dtype=torch.float32).to(dev)

    def net(m):
        return glorot_eval(m, seed, dev)

    def glorot_eval(m, s, d):
        M.glorot_(m, s)
        return m.eval().to(d)

    eig, col, apsn = net(M.GraffEig()), net(M.GraffCol()), net(M.GraffAPS())
    mlp, cnn = net(M.BeamMLP()), net(M.BeamCNN())
    lstm, tr = net(M.BeamLSTM()), net(M.SmallTransformer())
    s_net, c_net = net(M.StageValueNet()), net(M.StageValueNet())
    ranker, ee = net(M.RankerNet()), net(M.EarlyExitMLP())

    import torchvision
    rn18 = torchvision.models.resnet18(weights=None, num_classes=64)
    rn18 = glorot_eval(rn18, seed, dev)
    vit = torchvision.models.vit_b_16(weights=None, num_classes=64)
    vit = glorot_eval(vit, seed, dev)

    _, lstm_state = lstm(hist[:, :7])
    lstm_step = hist[:, 7:8]

    # numpy-side receiver state for the non-network components
    z_np = zs.copy()
    k = 8
    meas = np.zeros(64, dtype=bool)
    meas[:k] = True
    buf = np.zeros(64, dtype=np.complex128)
    buf[meas] = z_np[meas]
    feat = torch.empty(1, 194, dtype=torch.float32)
    z4 = np.stack([z_np[[0, 16, 32, 48]].real,
                   z_np[[0, 16, 32, 48]].imag], -1).ravel()
    r_in = torch.tensor(z4, dtype=torch.float32).unsqueeze(0).to(dev)

    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    # -- exhaustive sweep: the zero-compute reference still selects a beam --
    def sweep_select():
        p = buf.real ** 2 + buf.imag ** 2
        return int(np.argmax(p))

    # -- this project's frozen profile components r, q_k, u_k --------------
    def s1_r():
        y = ranker(r_in)
        order = torch.argsort(y[0], descending=True)
        sync()
        return order

    def s1_u():
        p = buf.real ** 2 + buf.imag ** 2
        p[~meas] = -np.inf
        return int(np.argmax(p))

    def s1_q():
        p = buf.real ** 2 + buf.imag ** 2
        p[~meas] = -np.inf
        b = int(np.argmax(p))                     # select B_k
        f = feat[0].numpy()
        f[:64] = buf.real
        f[64:128] = buf.imag
        f[128:192] = meas
        f[192] = SIGMA_P2 / SIGMA_D2
        f[193] = (16 - k) / 12.0
        x = feat.to(dev, non_blocking=True)
        s, c = s_net(x), c_net(x)                 # both charged every decision
        sync()
        return b if s.item() >= c.item() else -1  # stop-on-equality

    # -- published predictors, each with the codebook step it needs --------
    def codebook_argmax(v):
        """|F^H v|^2 over the 64-beam codebook, then argmax."""
        re = v[..., :64] @ F_t[..., 0] + v[..., 64:] @ F_t[..., 1]
        im = v[..., 64:] @ F_t[..., 0] - v[..., :64] @ F_t[..., 1]
        return torch.argmax(re ** 2 + im ** 2, dim=-1)

    def graff_eig():
        b = codebook_argmax(eig(z_t))
        sync()
        return b

    def graff_col():
        b = codebook_argmax(col(z_t))
        sync()
        return b

    def graff_aps():
        b = torch.argmax(apsn(aps)[0, 0])
        sync()
        return b

    def beam_mlp():
        b = torch.argmax(mlp(r_in)[0])
        sync()
        return b

    def beam_cnn():
        b = torch.argmax(cnn(csi2)[0])
        sync()
        return b

    def lstm_seq():
        y, _ = lstm(hist)
        b = torch.argmax(y[0])
        sync()
        return b

    def lstm_cached():
        y, _ = lstm(lstm_step, lstm_state)
        b = torch.argmax(y[0])
        sync()
        return b

    def transformer():
        b = torch.argmax(tr(z_t)[0])
        sync()
        return b

    def resnet18():
        b = torch.argmax(rn18(img)[0])
        sync()
        return b

    def vit_b16():
        b = torch.argmax(vit(img)[0])
        sync()
        return b

    def ee_n(nb):
        def f():
            b = torch.argmax(ee(feat.to(dev), nb)[0])
            sync()
            return b
        return f

    E, R, S = "exact", "reconstruct", "spec"
    comps = [
        ("sweep_select", sweep_select, S,
         "exhaustive 64-beam sweep: |z|^2 and argmax, no inference"),
        ("s1_r", s1_r, S, "ranker inference + sort of unmeasured order"),
        ("s1_u", s1_u, S, "mandatory stop: buffer update + select, no nets"),
        ("s1_q", s1_q, S,
         "optional stage decision: encode 194 feats + BOTH value nets + compare"),
        ("graff_eig", graff_eig, E,
         "arXiv:2201.04657 Fig.6 eigenvector net + codebook argmax"),
        ("graff_col", graff_col, E,
         "arXiv:2201.04657 covariance-vector net + codebook argmax"),
        ("graff_aps", graff_aps, R,
         "arXiv:2201.04657 Fig.5 APS conv net (widths reconstructed)"),
        ("beam_mlp", beam_mlp, R, "FC beam classifier 8-256-256-64"),
        ("beam_cnn", beam_cnn, R, "CNN beam predictor over wideband CSI"),
        ("lstm_cached", lstm_cached, R,
         "LSTM tracker, one step with cached state (honest per-decision cost)"),
        ("lstm_seq", lstm_seq, R, "LSTM tracker, full 8-step history replay"),
        ("transformer_small", transformer, R,
         "4-layer transformer encoder, d_model 128, 64 tokens"),
        ("resnet18_vision", resnet18, R,
         "ResNet-18 vision-aided beam predictor, 3x224x224, network only"),
        ("vit_b16_vision", vit_b16, R,
         "ViT-B/16 vision-aided beam predictor, 3x224x224, network only"),
        ("earlyexit_1", ee_n(1), S, "anytime predictor, 1 of 3 blocks"),
        ("earlyexit_2", ee_n(2), S, "anytime predictor, 2 of 3 blocks"),
        ("earlyexit_3", ee_n(3), S, "anytime predictor, all 3 blocks"),
    ]
    return comps


# ------------------------------------------------------------------ main ----

def env_record(profile: str, threads: int) -> dict:
    import torch
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gov = ""
    try:
        gov = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except OSError:
        pass
    return {
        "profile": profile, "cpu": cpu, "governor": gov,
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "requested_threads": threads,
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--full", action="store_true",
                    help="force the full G3 counts (1,000 warmup / 10,000 timed) even for "
                         "components whose call exceeds 200 us")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1) if a.threads == 1 else None
    if a.device == "cuda":
        torch.backends.cudnn.benchmark = True

    only = set(filter(None, a.only.split(",")))
    skip = set(filter(None, a.skip.split(",")))

    rows, raw = [], {}
    names = [c[0] for c in build(a.device, SEEDS[0])]
    for name in names:
        if (only and name not in only) or name in skip:
            continue
        per_seed = []
        for seed in SEEDS:
            comps = dict((c[0], c) for c in build(a.device, seed))
            _, fn, prov, note = comps[name]
            nw, nt = (WARMUP_FULL, TIMED_FULL) if a.full else budget(fn)
            s = timed(fn, nw, nt)
            raw[f"{name}.s{seed}"] = s
            per_seed.append({
                "seed": seed, "n_warmup": nw, "n_timed": nt,
                "p95_ns": nearest_rank_p95(s),
                "median_ns": int(np.median(s)),
                "min_ns": int(s.min()), "max_ns": int(s.max()),
                "mean_ns": float(s.mean()),
            })
        charge = max(r["p95_ns"] for r in per_seed)
        rows.append({
            "component": name, "provenance": prov, "note": note,
            "charge_ns": charge, "charge_us": round(charge / 1000.0, 3),
            "per_seed": per_seed,
            "reduced_n": per_seed[0]["n_timed"] != TIMED_FULL,
        })
        print(f"  {name:20s} {charge/1000.0:10.2f} us  "
              f"(n={per_seed[0]['n_timed']})", flush=True)

    out = {"env": env_record(a.profile, a.threads),
           "protocol": {"warmup_full": WARMUP_FULL, "timed_full": TIMED_FULL,
                        "seeds": list(SEEDS), "input_seed": INPUT_SEED,
                        "percentile": "nearest-rank p95",
                        "charge_rule": "max over weight seeds"},
           "rows": rows}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(a.out.replace(".json", "_raw.npz"), **raw)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
