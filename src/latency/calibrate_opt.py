#!/usr/bin/env python3
"""Addendum: the same charges under an optimised inference runtime.

One can object that the eager-mode figures measure PyTorch dispatch overhead
rather than the model.  That objection is fair and this answers it: identical
protocol, identical weights and inputs, but each network is traced, frozen and
run through torch.jit.optimize_for_inference first -- which folds batch norm,
removes dropout, and eliminates most of the per-op dispatch.  The gap between
the two columns is the dispatch tax; what remains is the model.
"""

import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import (SEEDS, INPUT_SEED, TIMED_FULL, nearest_rank_p95, timed,
                       budget, synthetic, env_record, SIGMA_D2)
import models as M

TARGETS = ["graff_eig", "graff_col", "graff_aps", "beam_mlp", "beam_cnn",
           "transformer_small", "resnet18_vision", "vit_b16_vision",
           "s1_q_nets"]


def frozen(module, example):
    module.eval()
    with torch.inference_mode():
        t = torch.jit.trace(module, example, check_trace=False)
        t = torch.jit.freeze(t)
        t = torch.jit.optimize_for_inference(t)
        for _ in range(3):
            t(*example) if isinstance(example, tuple) else t(example)
    return t


def build_opt(device, seed):
    dev = torch.device(device)
    rng = np.random.default_rng(INPUT_SEED)
    z = synthetic(rng) / np.sqrt(SIGMA_D2)
    z_t = torch.tensor(np.stack([z.real, z.imag], -1), dtype=torch.float32
                       ).unsqueeze(0).to(dev)
    aps = torch.tensor(np.abs(z) ** 2, dtype=torch.float32).reshape(1, 1, 64).to(dev)
    csi2 = z_t.permute(0, 2, 1).contiguous()
    img = torch.tensor(rng.standard_normal((1, 3, 224, 224)), dtype=torch.float32).to(dev)
    x194 = torch.tensor(rng.standard_normal((1, 194)), dtype=torch.float32).to(dev)
    x8 = torch.tensor(rng.standard_normal((1, 8)), dtype=torch.float32).to(dev)

    def mk(m):
        return M.glorot_(m, seed).eval().to(dev)

    import torchvision

    class TwoNets(torch.nn.Module):
        """Both stage value networks in one traced graph, as G3 charges them."""
        def __init__(s):
            super().__init__()
            s.a, s.b = M.StageValueNet(), M.StageValueNet()
        def forward(s, x):
            return s.a(x) - s.b(x)

    specs = {
        "graff_eig": (mk(M.GraffEig()), z_t),
        "graff_col": (mk(M.GraffCol()), z_t),
        "graff_aps": (mk(M.GraffAPS()), aps),
        "beam_mlp": (mk(M.BeamMLP()), x8),
        "beam_cnn": (mk(M.BeamCNN()), csi2),
        "transformer_small": (mk(M.SmallTransformer()), z_t),
        "resnet18_vision": (mk(torchvision.models.resnet18(weights=None, num_classes=64)), img),
        "vit_b16_vision": (mk(torchvision.models.vit_b_16(weights=None, num_classes=64)), img),
        "s1_q_nets": (mk(TwoNets()), x194),
    }
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    out = {}
    for name, (mod, ex) in specs.items():
        t = frozen(mod, ex)
        def f(t=t, ex=ex):
            y = t(ex); sync(); return y
        out[name] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    if a.device == "cuda":
        torch.backends.cudnn.benchmark = True

    rows = []
    for name in TARGETS:
        per_seed = []
        for seed in SEEDS:
            fn = build_opt(a.device, seed)[name]
            nw, nt = budget(fn)
            s = timed(fn, nw, nt)
            per_seed.append({"seed": seed, "n_warmup": nw, "n_timed": nt,
                             "p95_ns": nearest_rank_p95(s),
                             "median_ns": int(np.median(s))})
        charge = max(r["p95_ns"] for r in per_seed)
        rows.append({"component": name, "charge_ns": charge,
                     "charge_us": round(charge / 1000.0, 3),
                     "per_seed": per_seed,
                     "reduced_n": per_seed[0]["n_timed"] != TIMED_FULL})
        print(f"  {name:20s} {charge/1000.0:10.2f} us  (n={per_seed[0]['n_timed']})",
              flush=True)

    json.dump({"env": env_record(a.profile, a.threads), "runtime": "torchscript-frozen",
               "rows": rows}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
