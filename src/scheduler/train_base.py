#!/usr/bin/env python3
"""Train the two published architectures the G2 observation can carry, under
exactly the protocol the anytime predictor got: same fit city, same validation
city, same seeds, same noise law.

  fc   BeamMLP  (Phase 1 'beam_mlp')  -- K measured log-SNRs in, 64 logits out
  cnn  BeamCNN  (Phase 1 'beam_cnn')  -- (2,64) masked log-SNR + mask in

Each is trained at K = 4, 8, 16, 32 and 64 probes and the best K is selected per
architecture on the VALIDATION city, so neither comparator is strawmanned by a
probe count that happens to suit ours.  The vision architectures (ResNet-18,
ViT-B/16) have no image observation in these scenarios and enter only through
the Phase 2 tau* bound, which excludes them at any accuracy.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
import obs, feats
import models as M
from train import load, measure_t, mask_from_order, DEV, FIT, VAL, SEEDS, EPOCHS, BATCH

KS = (4, 8, 16, 32, 64)   # K=32,64 included: both networks are still rising at 16


def make(kind, k, seed):
    torch.manual_seed(seed)
    return (M.BeamMLP(n_in=k) if kind == 'fc' else M.BeamCNN()).to(DEV)


def inputs(kind, y, mask, order, k):
    s = torch.clamp(feats.snr_db_t(y), feats.CLIP_LO, feats.CLIP_HI) / 20.0
    if kind == 'fc':
        return s.gather(1, order[:, :k].expand(s.shape[0], -1)).float()
    return torch.stack([torch.where(mask, s, torch.full_like(s, feats.UNPROBED / 20.0)),
                        mask.to(s.dtype)], 1).float()


def run(kind, k, seed, pbar, se, tgt, vp, vs, vt, spread):
    model = make(kind, k, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    vgen = torch.Generator(device=DEV).manual_seed(seed + 1000)
    n, nv = pbar.shape[0], vp.shape[0]
    kv = torch.full((n,), k, device=DEV)
    kvv = torch.full((nv,), k, device=DEV)
    best, best_ep, best_state = -1.0, -1, None
    for ep in range(EPOCHS):
        model.train()
        y = measure_t(pbar, gen)
        m = mask_from_order(spread.expand(n, -1), kv)
        x = inputs(kind, y, m, spread, k)
        perm = torch.randperm(n, generator=gen, device=DEV)
        for s0 in range(0, n, BATCH):
            idx = perm[s0:s0 + BATCH]
            loss = nn.functional.cross_entropy(model(x[idx]), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        if ep % 4 == 3 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                yv = measure_t(vp, vgen)
                mv = mask_from_order(spread.expand(nv, -1), kvv)
                rec = model(inputs(kind, yv, mv, spread, k)).argmax(1)
                v = float(vs.gather(1, rec[:, None]).mean())
            if v > best:
                best, best_ep = v, ep
                best_state = {a: b.detach().clone() for a, b in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_ep


if __name__ == '__main__':
    t0 = time.time()
    pbar, se, tgt = load(FIT); vp, vs, vt = load(VAL)
    spread = torch.tensor(obs.spread_order().copy(), device=DEV)[None, :]
    os.makedirs(os.path.join(HERE, 'models'), exist_ok=True)
    out = {"ks": list(KS), "runs": []}
    # As in train.py: a (kind, K, seed) whose checkpoint exists is kept, never
    # retrained, together with its recorded validation score.
    bpath = os.path.join(HERE, 'baselines.json')
    prior = ({(r['kind'], r['k'], r['seed']): r for r in json.load(open(bpath))['runs']}
             if os.path.exists(bpath) else {})
    for kind in ('fc', 'cnn'):
        for k in KS:
            for seed in SEEDS:
                ckpt = os.path.join(HERE, 'models', f'{kind}_k{k}_{seed}.pt')
                if os.path.exists(ckpt) and (kind, k, seed) in prior:
                    out['runs'].append(prior[(kind, k, seed)])
                    continue
                m, v, ep = run(kind, k, seed, pbar, se, tgt, vp, vs, vt, spread)
                torch.save(m.state_dict(), os.path.join(HERE, 'models', f'{kind}_k{k}_{seed}.pt'))
                out['runs'].append({"kind": kind, "k": k, "seed": seed,
                                    "val_se": round(v, 4), "epoch": ep})
                print(f"  {kind:3s} K={k:2d} seed {seed}: val SE {v:.4f} @ep{ep}", flush=True)
    for kind in ('fc', 'cnn'):
        by = {k: np.mean([r['val_se'] for r in out['runs']
                          if r['kind'] == kind and r['k'] == k]) for k in KS}
        out[f'best_k_{kind}'] = int(max(by, key=by.get))
        out[f'val_by_k_{kind}'] = {str(k): round(float(v), 4) for k, v in by.items()}
        print(f"{kind}: val by K {out[f'val_by_k_{kind}']} -> best K={out[f'best_k_{kind}']}")
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, 'baselines.json'), 'w'), indent=1)
    print(f"done in {out['seconds']}s")
