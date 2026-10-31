#!/usr/bin/env python3
"""Train the anytime early-exit beam predictor.  Fit city only; epoch chosen on
the validation city; test cities never touched here.

Two rounds, as the protocol declares.  Round 0 sees prefixes of the frozen
bit-reversal probe order.  Round 1 -- the model that is actually used -- sees
prefixes the round-0 model itself ordered after a 4-probe frozen opening, so
the training distribution is the one the deployed controller creates.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
import obs, feats
import models as M

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIT, VAL = 'city_0_newyork_28', 'city_1_losangeles_28'
K0 = 4                      # frozen opening probes before any compute
K_TRAIN = 48                # training probe counts ~ U{1..48} (A3)
EPOCHS = 80
BATCH = 1024
VAL_KS = (2, 4, 8, 16)
import protocol
# The seed list in force (ten seeds); `--seeds ...` overrides it, e.g. to
# re-run the prespecified three.  Written in exactly one place: protocol.json.
SEEDS = protocol.seed_arg()


def load(name):
    z = np.load(os.path.join(HERE, 'data', f'{name}.npz'))
    pbar = torch.tensor(z['pbar'][:, obs.LAYER6_OFFSET:].astype(np.float64), device=DEV)
    se = torch.tensor(z['se'][:, obs.LAYER6_OFFSET:].astype(np.float32), device=DEV)
    return pbar, se, se.argmax(1)


def measure_t(pbar, gen):
    """G2 measurement, torch: (sigma_p^2/2K) chi'^2(2K, 2K S/sigma_p^2)."""
    lam = 2.0 * obs.K_F * obs.P_TX * pbar / obs.SIGMA_P2
    z = torch.randn(pbar.shape, generator=gen, device=DEV, dtype=torch.float64)
    x = (z + torch.sqrt(lam)) ** 2
    # chi^2_{2K-1} = sum of 2K-1 squared standard normals
    r = torch.randn((*pbar.shape, 2 * obs.K_F - 1), generator=gen,
                    device=DEV, dtype=torch.float64)
    x = x + (r * r).sum(-1)
    y = obs.SIGMA_P2 / (2.0 * obs.K_F) * x
    return torch.clamp(y - obs.SIGMA_P2, min=1e-3 * obs.SIGMA_P2)


def mask_from_order(order, k):
    """order (B,L) int64, k (B,) int64 -> (B,64) bool of that order's first k."""
    b, l = order.shape
    take = torch.arange(l, device=DEV)[None, :] < k[:, None]
    m = torch.zeros((b, 64), dtype=torch.bool, device=DEV)
    m.scatter_(1, order, take)
    return m


def order_from_logits(logits, opening):
    """Frozen opening first, then the model's ranking of what is left."""
    B = logits.shape[0]
    lg = logits.clone()
    lg.scatter_(1, opening.expand(B, -1), -1e30)
    rest = lg.argsort(1, descending=True)
    return torch.cat([opening.expand(B, -1), rest[:, :64 - opening.shape[1]]], 1)


def evaluate(model, pbar, se, tgt, gen, ks=VAL_KS, order=None, opening=None,
             round0=None):
    """Mean delivered SE of each exit's recommendation, averaged over ks."""
    model.eval()
    out = np.zeros(3)
    with torch.no_grad():
        for k in ks:
            y = measure_t(pbar, gen)
            if round0 is None:
                o = order.expand(pbar.shape[0], -1)
            else:
                m0 = mask_from_order(opening.expand(pbar.shape[0], -1),
                                     torch.full((pbar.shape[0],), K0, device=DEV))
                f0, _ = feats.build(y, m0)
                o = order_from_logits(round0(f0, 3), opening)
            m = mask_from_order(o, torch.full((pbar.shape[0],), k, device=DEV))
            f, _ = feats.build(y, m)
            for e in (1, 2, 3):
                rec = model(f, e).argmax(1)
                out[e - 1] += se.gather(1, rec[:, None]).mean().item()
    return out / len(ks)


def fit(seed, pbar, se, tgt, vp, vs, vt, round0, opening, spread, log):
    torch.manual_seed(seed)
    model = M.EarlyExitMLP().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    vgen = torch.Generator(device=DEV).manual_seed(seed + 1000)
    n = pbar.shape[0]
    best, best_ep, best_state = -1.0, -1, None
    for ep in range(EPOCHS):
        model.train()
        y = measure_t(pbar, gen)
        if round0 is None:
            o = spread.expand(n, -1)
        else:
            with torch.no_grad():
                m0 = mask_from_order(opening.expand(n, -1),
                                     torch.full((n,), K0, device=DEV))
                f0, _ = feats.build(y, m0)
                o = order_from_logits(round0(f0, 3), opening)
            # A3: half the batch keeps the frozen order, which is what every
            # fixed-K baseline feeds the predictor at evaluation time.
            keep = torch.rand(n, generator=gen, device=DEV) < 0.5
            o = torch.where(keep[:, None], spread.expand(n, -1), o)
        k = torch.randint(1, K_TRAIN + 1, (n,), generator=gen, device=DEV)
        m = mask_from_order(o, k)
        f, _ = feats.build(y, m)
        perm = torch.randperm(n, generator=gen, device=DEV)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            loss = sum(nn.functional.cross_entropy(model(f[idx], e), tgt[idx])
                       for e in (1, 2, 3))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 4 == 3 or ep == EPOCHS - 1:
            v = evaluate(model, vp, vs, vt, vgen, order=spread,
                         opening=opening, round0=round0)
            score = float(v.mean())
            if score > best:
                best, best_ep = score, ep
                best_state = {k2: v2.detach().clone() for k2, v2 in model.state_dict().items()}
            log.append({"seed": seed, "epoch": ep, "val_se_by_exit": [round(x, 4) for x in v],
                        "val_score": round(score, 4), "loss": round(float(loss), 4)})
            print(f"    seed {seed} ep {ep:3d} loss {float(loss):7.3f} "
                  f"val SE {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}  best@{best_ep}", flush=True)
    model.load_state_dict(best_state)
    return model, best, best_ep


if __name__ == '__main__':
    t0 = time.time()
    pbar, se, tgt = load(FIT)
    vp, vs, vt = load(VAL)
    print(f"fit {FIT} n={pbar.shape[0]}  val {VAL} n={vp.shape[0]}  dev={DEV}")
    spread = torch.tensor(obs.spread_order().copy(), device=DEV)[None, :]
    opening = spread[:, :K0]
    out = {"fit": FIT, "val": VAL, "k0": K0, "epochs": EPOCHS, "seeds": list(SEEDS),
           "oracle_val_se": round(float(vs.max(1).values.mean()), 4),
           "oracle_fit_se": round(float(se.max(1).values.mean()), 4), "runs": []}
    os.makedirs(os.path.join(HERE, 'models'), exist_ok=True)
    # Never retrain a seed that already has checkpoints: nothing here makes CUDA
    # training bit-deterministic, so a retrained seed would silently replace
    # the weights every committed number was computed with.  Its record is kept.
    tpath = os.path.join(HERE, 'training.json')
    prior = {r['seed']: r for r in json.load(open(tpath))['runs']} if os.path.exists(tpath) else {}
    for seed in SEEDS:
        ckpt = [os.path.join(HERE, 'models', f'{p}_{seed}.pt') for p in ('ee', 'ee_round0')]
        if all(os.path.exists(c) for c in ckpt) and seed in prior:
            out['runs'].append(prior[seed])
            print(f"  [seed {seed}] checkpoints exist; kept, not retrained")
            continue
        log0, log1 = [], []
        print(f"  [seed {seed}] round 0 (frozen-order prefixes)")
        m0, s0, e0 = fit(seed, pbar, se, tgt, vp, vs, vt, None, opening, spread, log0)
        m0.eval()
        for p in m0.parameters():
            p.requires_grad_(False)
        print(f"  [seed {seed}] round 1 (model-ordered prefixes)")
        m1, s1, e1 = fit(seed, pbar, se, tgt, vp, vs, vt, m0, opening, spread, log1)
        torch.save(m1.state_dict(), os.path.join(HERE, 'models', f'ee_{seed}.pt'))
        torch.save(m0.state_dict(), os.path.join(HERE, 'models', f'ee_round0_{seed}.pt'))
        out['runs'].append({"seed": seed, "round0_val": round(s0, 4), "round0_epoch": e0,
                            "round1_val": round(s1, 4), "round1_epoch": e1,
                            "log_round0": log0, "log_round1": log1})
        print(f"  [seed {seed}] round0 {s0:.4f} @ep{e0}   round1 {s1:.4f} @ep{e1}")
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, 'training.json'), 'w'), indent=1)
    print(f"\ndone in {out['seconds']}s  oracle val SE {out['oracle_val_se']}")
