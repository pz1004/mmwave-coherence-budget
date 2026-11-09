#!/usr/bin/env python3
"""Rule R43: the trained predictors on the covered population (protocol.json analysis 'covtrain',
declared before it ran; R43 is post hoc).

Section VII names a possible cause of the trained networks' low top-1 accuracy, untested: the users
short of CQI 1, whose best-beam labels measurement noise sets.  R43 tests it on the validation
scenario (Los Angeles), where the paper states the accuracies (profile_pred.py ->
predictor_profile.json), in two ways.

(a) The stored networks (models/{fc,cnn}_k{K}_{seed}.pt, K = 4, 8, 16, 32 and 64, the ten seeds)
    and Top-K, with profile_pred.py's streams and calls (sim.measure on Site.pbar seeded with the
    seed, methods.fixed_net, methods.topk): top-1 accuracy, the share of users whose selected beam is
    the genie-aided one, and efficiency, the mean selected rate over the mean genie-aided rate, on
    all users and on rule R37's covered population (genie-aided rate at least CQI 1, 2 x 78/1024
    bit/s/Hz); the mean over seeds.
(b) The same architectures retrained on the training scenario's covered users only, each epoch
    choice made on the validation scenario's covered users, with train_base.run unchanged (AdamW,
    80 epochs, its seeds and streams), evaluated as in (a).  The retrained weights stay in models/
    (untracked, like every checkpoint) as cov_{kind}_k{K}_{seed}.pt.

Gates, before anything is written: T1 (a) on all users reproduces predictor_profile.json's top1_acc
and eta, as rounded there, for every network and Top-K it lists; T2 the covered users of both
scenarios are covpop.json's (their shares times the scenarios' sizes).

usage:  python3 covtrain.py [--gate]      -> covtrain.json   (--gate runs T1-T2 and writes nothing)
"""
import hashlib, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
KEY = 'covtrain'
OUT = os.path.join(HERE, 'covtrain.json')
CQI1 = 2 * 78 / 1024
KS = (4, 8, 16, 32, 64)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def evaluate_all(site, ys, nets, keep, ch, tp):
    """Per method, per seed: (acc_all, eta-numerator_all, acc_cov, eta-numerator_cov)."""
    import torch, obs, methods
    L6 = obs.LAYER6_OFFSET
    out = {}
    for si, y in enumerate(ys):
        for k in KS:
            if k < 64:
                d, c = methods.topk(site, y, k, tp)
                out.setdefault(f'topk{k}', []).append(c)
            for kind in ('fc', 'cnn'):
                for tag, m in nets(kind, k, si):
                    tau = ch[f'fc_k{k}'] if kind == 'fc' else ch['cnn']
                    d, c = methods.fixed_net(site, y, k, kind, m, tau, tp)
                    out.setdefault(f'{tag}{kind}{k}', []).append(c)
    kt = torch.tensor(keep, device=site.se.device)
    orc_all, orc_cov = float(site.oracle_se.mean()), float(site.oracle_se[kt].mean())
    res = {}
    for name, cs in out.items():
        acc_a, acc_c, se_a, se_c = [], [], [], []
        for c in cs:
            hit = (c - L6 == site.oracle_beam).double()
            se = site.se.gather(1, c[:, None])[:, 0].double()
            acc_a.append(float(hit.mean())); acc_c.append(float(hit[kt].mean()))
            se_a.append(float(se.mean())); se_c.append(float(se[kt].mean()))
        res[name] = {'top1_acc_all': float(np.mean(acc_a)), 'top1_acc_covered': float(np.mean(acc_c)),
                     'eta_all': float(np.mean(se_a)) / orc_all,
                     'eta_covered': float(np.mean(se_c)) / orc_cov,
                     'top1_acc_covered_seeds': acc_c, 'eta_covered_seeds': [s / orc_cov for s in se_c]}
    return res


def main():
    import torch, obs, sim, protocol
    import models as M
    gate_only = '--gate' in sys.argv
    if not gate_only:
        P = json.load(open(os.path.join(HERE, 'protocol.json')))
        A = [a for a in P['analyses'] if a['key'] == KEY]
        assert len(A) == 1 and A[0]['scripts_sha256']['covtrain.py'] == sha(os.path.abspath(__file__)), \
            'covtrain.py is not the declared script'
        assert not os.path.exists(OUT), f'{OUT} exists: written once'
    from train import load, FIT, VAL, DEV
    seeds = protocol.seed_arg()
    lat = json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json')))
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    tp = obs.T_P_US + ch['probe_update']
    t0 = time.time()
    site = sim.Site(VAL)
    keep = (site.oracle_se >= CQI1).cpu().numpy()
    ys = [sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(s)) for s in seeds]

    def stored(kind, k, si):
        m = (M.BeamMLP(n_in=k) if kind == 'fc' else M.BeamCNN()).to(sim.DEV)
        m.load_state_dict(torch.load(os.path.join(HERE, 'models', f'{kind}_k{k}_{seeds[si]}.pt')))
        return [('', m.eval())]

    res = evaluate_all(site, ys, stored, keep, ch, tp)
    PP = json.load(open(os.path.join(HERE, 'predictor_profile.json')))
    assert PP['city'] == VAL
    t1 = {m: (round(res[m]['top1_acc_all'], 4) == PP['methods'][m]['top1_acc'] and
              round(res[m]['eta_all'], 4) == PP['methods'][m]['eta'])
          for m in res if m in PP['methods']}
    CP = json.load(open(os.path.join(HERE, 'covpop.json')))
    t2 = {}
    for city in (FIT, VAL):
        _, se_c, _ = load(city)
        kc = int((se_c.max(1).values >= CQI1).sum())
        t2[city] = kc == round(CP['shares'][city]['cqi1']['users_share'] * se_c.shape[0])
    gates = {'T1': t1, 'T2': t2}
    ok = {g: all(v.values()) for g, v in gates.items()}
    print(json.dumps(ok), f'({len(t1)} methods checked)', flush=True)
    if not all(ok.values()):
        raise SystemExit(f'gate failed: {gates}; nothing written')
    if gate_only:
        print('gates T1-T2 hold; nothing written')
        return
    # (b) retrain on the covered users of the training scenario, epochs chosen on the validation
    # scenario's covered users, with train_base.run unchanged
    import train_base
    pbar, se, tgt = load(FIT)
    vp, vs, vt = load(VAL)
    kf, kv = se.max(1).values >= CQI1, vs.max(1).values >= CQI1
    spread = torch.tensor(obs.spread_order().copy(), device=DEV)[None, :]
    runs = []
    for kind in ('fc', 'cnn'):
        for k in KS:
            for seed in seeds:
                m, v, ep = train_base.run(kind, k, seed, pbar[kf], se[kf], tgt[kf], vp[kv], vs[kv],
                                          vt[kv], spread)
                torch.save(m.state_dict(), os.path.join(HERE, 'models', f'cov_{kind}_k{k}_{seed}.pt'))
                runs.append({'kind': kind, 'k': k, 'seed': seed, 'val_se_covered': v, 'epoch': ep})
                print(f'  retrained {kind} K={k} seed {seed}: val SE {v:.4f} @ep{ep} '
                      f'({time.time() - t0:.0f} s)', flush=True)

    def retrained(kind, k, si):
        m = (M.BeamMLP(n_in=k) if kind == 'fc' else M.BeamCNN()).to(sim.DEV)
        m.load_state_dict(torch.load(os.path.join(HERE, 'models', f'cov_{kind}_k{k}_{seeds[si]}.pt')))
        return [('cov_', m.eval())]

    res_cov = evaluate_all(site, ys, retrained, keep, ch, tp)
    res.update({m: v for m, v in res_cov.items() if m.startswith('cov_')})
    out = {'rule': 'R43', 'script_sha256': sha(os.path.abspath(__file__)), 'city': VAL, 'fit': FIT,
           'seeds': list(seeds), 'cqi1': CQI1, 'ks': list(KS), 'covered_users': int(keep.sum()),
           'users': int(keep.size), 'gate': gates, 'methods': res, 'retraining': runs,
           'inputs_sha256': {'predictor_profile.json': sha(os.path.join(HERE, 'predictor_profile.json')),
                             'covpop.json': sha(os.path.join(HERE, 'covpop.json'))},
           'seconds': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for m in sorted(res, key=lambda x: (x.lstrip('cov_').lstrip('fcntopk'), x)):
        r = res[m]
        print(f"  {m:>10}: acc all {r['top1_acc_all']:.3f} covered {r['top1_acc_covered']:.3f}; "
              f"eta all {r['eta_all']:.4f} covered {r['eta_covered']:.4f}")
    print(f'wrote {OUT} in {out["seconds"]} s')


if __name__ == '__main__':
    main()
