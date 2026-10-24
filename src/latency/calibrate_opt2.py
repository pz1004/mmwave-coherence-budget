#!/usr/bin/env python3
"""Corrected optimised-runtime calibration: the WHOLE decision is traced.

calibrate_opt.py traced only the network, while calibrate.py charged the whole
decision -- feature work, the network, beam selection, action emission.  Taking
min() across those two columns silently dropped the selection step.  Here every
traced module ends in a beam index, so both columns charge the same operation.
"""
import json, math, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import (timed, budget, nearest_rank_p95, synthetic, SEEDS,
                       INPUT_SEED, SIGMA_D2, SIGMA_P2, TIMED_FULL, env_record)
import models as M


class Codebook(nn.Module):
    """|F^H v|^2 over the 64-beam DFT codebook, then argmax -> a beam index."""
    def __init__(self, F):
        super().__init__(); self.register_buffer("F", F)
    def forward(self, v):
        re = v[..., :64] @ self.F[..., 0] + v[..., 64:] @ self.F[..., 1]
        im = v[..., 64:] @ self.F[..., 0] - v[..., :64] @ self.F[..., 1]
        return torch.argmax(re ** 2 + im ** 2, dim=-1)


class Decide(nn.Module):
    """net -> selection.  `head` is 'codebook' or 'argmax'."""
    def __init__(self, net, head, F=None):
        super().__init__(); self.net = net
        self.cb = Codebook(F) if head == "codebook" else None
    def forward(self, x):
        y = self.net(x)
        if self.cb is not None:
            return self.cb(y)
        return torch.argmax(y.flatten(1), dim=-1)


class S1Q(nn.Module):
    """The full stage decision in torch: select, encode 194 feats, both value
    networks, compare, emit.  Same operation the eager s1_q charges."""
    def __init__(self):
        super().__init__()
        self.a, self.b = M.StageValueNet(), M.StageValueNet()
        self.register_buffer("ratio", torch.tensor(SIGMA_P2 / SIGMA_D2))
    def forward(self, buf, mask, frac):
        p = buf[..., 0] ** 2 + buf[..., 1] ** 2
        p = torch.where(mask > 0, p, torch.full_like(p, -1e30))
        beam = torch.argmax(p, dim=-1)
        x = torch.cat([buf[..., 0], buf[..., 1], mask,
                       self.ratio.expand(1, 1), frac], dim=-1)
        return beam, (self.a(x) - self.b(x))


def frozen(mod, ex):
    mod.eval()
    with torch.inference_mode():
        t = torch.jit.trace(mod, ex, check_trace=False)
        t = torch.jit.optimize_for_inference(torch.jit.freeze(t))
        for _ in range(3):
            t(*ex)
    return t


def build(seed, dev='cpu'):
    rng = np.random.default_rng(INPUT_SEED)
    z = synthetic(rng) / math.sqrt(SIGMA_D2)
    z_t = torch.tensor(np.stack([z.real, z.imag], -1), dtype=torch.float32).unsqueeze(0)
    aps = torch.tensor(np.abs(z) ** 2, dtype=torch.float32).reshape(1, 1, 64)
    csi2 = z_t.permute(0, 2, 1).contiguous()
    img = torch.tensor(rng.standard_normal((1, 3, 224, 224)), dtype=torch.float32)
    x8 = torch.tensor(rng.standard_normal((1, 8)), dtype=torch.float32)
    n = np.arange(64); Fm = np.exp(2j * np.pi * np.outer(n, n) / 64) / 8.0
    F = torch.tensor(np.stack([Fm.real, Fm.imag], -1), dtype=torch.float32)
    mask = torch.zeros(1, 64); mask[0, :8] = 1
    frac = torch.tensor([[(16 - 8) / 12.0]])

    def g(m):
        return M.glorot_(m, seed).eval()

    specs = {
        "graff_eig": (Decide(g(M.GraffEig()), "codebook", F), (z_t,)),
        "graff_col": (Decide(g(M.GraffCol()), "codebook", F), (z_t,)),
        "graff_aps": (Decide(g(M.GraffAPS()), "argmax"), (aps,)),
        "beam_mlp": (Decide(g(M.BeamMLP()), "argmax"), (x8,)),
        "beam_cnn": (Decide(g(M.BeamCNN()), "argmax"), (csi2,)),
        "transformer_small": (Decide(g(M.SmallTransformer()), "argmax"), (z_t,)),
        "resnet18_vision": (Decide(g(torchvision.models.resnet18(
            weights=None, num_classes=64)), "argmax"), (img,)),
        "vit_b16_vision": (Decide(g(torchvision.models.vit_b_16(
            weights=None, num_classes=64)), "argmax"), (img,)),
        "s1_q": (g(S1Q()), (z_t, mask, frac)),
    }
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    out = {}
    for name, (mod, ex) in specs.items():
        mod = mod.to(dev)
        ex = tuple(e.to(dev) for e in ex)
        t = frozen(mod, ex)
        def f(t=t, ex=ex, sync=sync):
            y = t(*ex); sync(); return y
        out[name] = f
    return out


if __name__ == "__main__":
    torch.set_grad_enabled(False); torch.set_num_threads(1)
    DEV = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    if DEV == "cuda":
        torch.backends.cudnn.benchmark = True
    rows = []
    names = list(build(SEEDS[0], DEV).keys())
    for name in names:
        per = []
        for seed in SEEDS:
            fn = build(seed, DEV)[name]
            nw, nt = budget(fn)
            s = timed(fn, nw, nt)
            per.append({"seed": seed, "n_warmup": nw, "n_timed": nt,
                        "p95_ns": nearest_rank_p95(s)})
        ch = max(r["p95_ns"] for r in per)
        rows.append({"component": name, "charge_ns": ch,
                     "charge_us": round(ch / 1000.0, 3), "per_seed": per,
                     "reduced_n": per[0]["n_timed"] != TIMED_FULL})
        print(f"  {name:20s} {ch/1000.0:10.2f} us  (n={per[0]['n_timed']})", flush=True)
    json.dump({"env": env_record(sys.argv[1], 1), "runtime": "torchscript-full-decision",
               "rows": rows}, open(sys.argv[2], "w"), indent=2)
    print("wrote", sys.argv[2])
