#!/usr/bin/env python3
"""The exhaustive sweep's efficiency under the measurement of the aperiodic CSI-RS procedure of
Section VI-D.  Declared in protocol.json (analysis 'csirs_margins_calib') before it ran; its lower
bound and R12's tests follow the amendment recorded there.

The probe model.  sim.measure (G2) reports a codeword's band-averaged received power from a pilot
of N = 1000 complex samples (10 us over 100 MHz), combined coherently within each of the K_F = 16
sub-bands, whose channel the model holds at its centre value:
    y = (s2 / 2K_F) chi'^2(2K_F, 2 N g) - s2,  clipped at 1e-3 s2,  s2 = sigma_d^2 K_F / N,
with g = P_TX pbar / sigma_d^2 the band-averaged SNR.  Every sample carries the data's SNR, so the
number of samples N alone sets the measurement noise.

The procedure.  One NZP CSI-RS resource of one port and one OFDM symbol (TS 38.211 Table
7.4.1.5.3-1: density 3, 1 or 0.5 resource elements per resource block) over the 66 resource
blocks of a 100 MHz carrier at 120 kHz (TS 38.101-2 Table 5.3.2-1; a CSI-RS may span the whole
bandwidth part, TS 38.331 CSI-FrequencyOccupation) carries N = 198, 66 or 33 samples at the data's
energy per resource element (no power offset).  Two estimators:
  coherent  the model's own law with N in place of 1000, on the probe model's own normal draws
            (evaluate's generator), so that N = 1000 is the recorded run and every N is paired
            with it (common random numbers); defined for N >= K_F, one sample per sub-band, where
            it coincides with the per-element law;
  per_re    energy detection on each resource element without coherent combining,
            (sigma_d^2 / 2N) chi'^2(2N, 2 N g) - sigma_d^2, clipped at 1e-3 sigma_d^2, on a numpy
            stream of its own: a conservative variant.
The sweep reports the CRI of its largest measured L1-RSRP (TS 38.214 5.2.1.4.3), so quantizing the
reported value cannot change its choice; the base station takes the reported beam, chosen by
methods.topk's rule (the argmax in the frozen spread order).

Statistics, per configuration and scenario (the seven non-training scenarios, all eligible users,
the 10 seeds):
  eta     the mean over seeds and users of the selected beam's rate over that of the genie-aided
          beam (evaluate's definition of eta);
  se_lin  the delta-method standard error of eta: the t interval's ingredients, s_seed^2/S +
          SE_block^2, of the linearized residual (se_selected - eta se_genie)/mean(se_genie),
          from evaluate.paired_bootstrap_hier (the t interval of the contrast
          mean(se_selected - e se_genie) at a fixed e has a standard error growing with (eta - e)
          times the between-block uncertainty of the mean genie-aided rate, so its t version
          cannot pass at any e, a Fieller-type set without a bound);
  eta_lb  one-sided, at level 0.05/7 (Bonferroni over the seven scenarios), the smaller of
          e_boot  the largest e at which the crossed bootstrap of the same paired contrast puts
                  at most 0.05/7 of its replicates at or below zero, which is the bootstrap
                  quantile of the efficiency itself; a SCAN-point scan of [eta - 0.25, eta] that
                  must show a single pass-to-fail transition, then bisection to 1e-6, on one
                  bootstrap stream per scenario and configuration; and
          e_t     eta - t_(S-1)(1 - 0.05/7) se_lin;
  R12     at e = the CSI-RS threshold of ResNet-18 on the GPU (make_draft's formula, recomputed
          here from rules_rx.json): the bootstrap p-value of the contrast and the t p-value of
          (eta - e)/se_lin with S - 1 degrees of freedom; established in a scenario when the
          larger survives Holm across the seven scenarios and both 95% intervals lie above e; the
          exclusion is supported at a configuration when it is established in all seven.

Gates, before anything is written: (i) the coherent law at N = 1000 reproduces sim.measure bit for
bit for every scenario and seed; (ii) the recomputed sweep at N = 1000 reproduces the stored
per-period means (within 1e-5) and eta (to four decimals) in all seven scenarios; (iii) both
estimators are unbiased before the clip (10^6 draws at three SNRs and two N, within four
standard errors).

usage:  python3 csirs.py [--check-only]       -> csirs.json
"""
import hashlib, json, os, sys, time
import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, obs, methods, evaluate, protocol
from reading_rules import holm

SCEN = ['city_1_losangeles_28', 'city_2_chicago_28', 'city_3_houston_28', 'city_4_phoenix_28',
        'city_5_philadelphia_28', 'city_6_miami_28', 'city_7_sandiego_28']
STORED = {SCEN[0]: 'validation.json', **{n: 'results.json' for n in SCEN[1:3]},
          **{n: 'heldout.json' for n in SCEN[3:]}}
N_PROBE = 1000            # the probe model's samples: obs asserts K_F sigma_d^2 / sigma_p^2 == 1000
N_RB = 66                 # TS 38.101-2 Table 5.3.2-1: 100 MHz at 120 kHz
DENSITY = {'3': 3 * N_RB, '1': 1 * N_RB, '0.5': N_RB // 2}      # one port, one symbol
# coherent combining needs at least one sample per sub-band; at N = K_F the two laws coincide
COHERENT_N = [16, 24, 33, 48, 66, 99, 132, 198, 330, 500, 1000]
PER_RE_N = [33, 66, 198, 1000]
ALPHA_LB = 0.05 / len(SCEN)
TOL = 1e-5                # the stored means are rounded to five decimals
SCAN = 41                 # grid points of the scan that brackets eta_lb
assert set(DENSITY.values()) <= set(COHERENT_N) and set(DENSITY.values()) <= set(PER_RE_N)
assert min(COHERENT_N) == obs.K_F
assert obs.SIGMA_D2 / (N_PROBE / obs.K_F) == obs.SIGMA_P2, "N = 1000 is sim.measure's law"


def nr_threshold(grid):
    """make_draft._nr_eta for ResNet-18 on the GPU at its fastest runtime: the lowest lower bound
    on the sweep's efficiency at which Proposition 1(b) excludes it under the procedure, with 12
    one-symbol resources per 125 us slot, the common delays zero and one slot in its favor."""
    r5 = json.load(open(os.path.join(HERE, 'rules_rx.json')))['R5']
    row = next(r for r in r5['rows']
               if r['component'] == 'resnet18_vision' and r['profile'] == 'gpu-resident')
    tau, slot, n, K = min(row['charges_us'].values()), 125.0, 12, r5['K']
    end = lambda j: (j // n) * slot + (14 - n + j % n + 1) * slot / 14
    t_max = max(grid)
    return 1 - (tau - (end(63) - end(K - 1)) - slot) / (t_max - end(63)), tau


def draws(site, gen):
    """sim.measure's normal draws in its order: z (n, 126), then r (n, 126, 2K_F - 1), summed."""
    z = torch.randn(site.pbar.shape, generator=gen, device=sim.DEV, dtype=torch.float64)
    r = torch.randn((*site.pbar.shape, 2 * obs.K_F - 1), generator=gen, device=sim.DEV,
                    dtype=torch.float64)
    return z, (r * r).sum(-1)


def coherent(pbar, z, q, n_samp):
    """sim.measure's arithmetic with N samples in place of 1000."""
    s2 = obs.SIGMA_D2 / (n_samp / obs.K_F)
    lam = 2.0 * obs.K_F * obs.P_TX * pbar / s2
    x = (z + torch.sqrt(lam)) ** 2
    return torch.clamp(s2 / (2 * obs.K_F) * (x + q) - s2, min=1e-3 * s2)


def per_re(pbar64, rng, n_samp):
    """Energy detection on each of N resource elements, no coherent combining."""
    x = rng.noncentral_chisquare(2 * n_samp, 2 * n_samp * obs.P_TX * pbar64 / obs.SIGMA_D2)
    return np.maximum(obs.SIGMA_D2 / (2 * n_samp) * x - obs.SIGMA_D2, 1e-3 * obs.SIGMA_D2)


def unbiased():
    """Gate (iii): before the clip, both estimators average the band-averaged power."""
    rng, out = np.random.default_rng(evaluate.stable_seed('csirs', 'unbiased')), []
    for n_samp in (33, 1000):
        for g in (0.01, 1.0, 100.0):
            pbar = g * obs.SIGMA_D2 / obs.P_TX
            s2 = obs.SIGMA_D2 / (n_samp / obs.K_F)
            xc = rng.noncentral_chisquare(2 * obs.K_F, 2 * n_samp * g, 10 ** 6)
            yc = s2 / (2 * obs.K_F) * xc - s2
            xr = rng.noncentral_chisquare(2 * n_samp, 2 * n_samp * g, 10 ** 6)
            yr = obs.SIGMA_D2 / (2 * n_samp) * xr - obs.SIGMA_D2
            for nm, y in (('coherent', yc), ('per_re', yr)):
                zs = (y.mean() - obs.P_TX * pbar) / (y.std() / np.sqrt(y.size))
                out.append({"estimator": nm, "n": n_samp, "snr": g, "z": float(zs)})
    return out, all(abs(o['z']) < 4 for o in out)


def scenario(name, seeds, t_probe, grid):
    """Per seed and configuration, the rate of the beam the sweep selects, (S, n) each; the
    genie-aided rate (n,); the blocks; and gates (i) and (ii)."""
    site = sim.Site(name)
    L6 = obs.LAYER6_OFFSET
    sel, same, gen_beam, bit, rates, se_seed = {}, {}, {}, True, {}, []
    for s in seeds:
        y_ref = sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        z, q = draws(site, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        rng = np.random.default_rng(evaluate.stable_seed('csirs', 'per_re', name, s))
        pbar64 = site.pbar[:, L6:].cpu().numpy()
        c_ref = None
        for est, ns in [('coherent', n) for n in COHERENT_N] + [('per_re', n) for n in PER_RE_N]:
            if est == 'coherent':
                y = coherent(site.pbar, z, q, ns)
                if ns == N_PROBE:
                    bit &= bool(torch.equal(y, y_ref))
            else:
                y = torch.zeros_like(site.pbar)
                y[:, L6:] = torch.tensor(per_re(pbar64, rng, ns), device=sim.DEV)
            d, c = methods.topk(site, y, 64, t_probe)
            se = site.se.gather(1, c[:, None])[:, 0].double()
            key = f"{est}|{ns}"
            sel.setdefault(key, []).append(se.cpu().numpy())
            if est == 'coherent' and ns == N_PROBE:
                c_ref = c
                se_seed.append(float(se.mean()))
                for T in grid:
                    rates.setdefault(T, []).append(
                        float((torch.clamp(1 - d / T, min=0.0) * se).mean()))
            same.setdefault(key, []).append(c)
            gen_beam.setdefault(key, []).append(
                float((c == site.oracle_beam + L6).double().mean()))
        for key, cs in same.items():
            same[key][-1] = float((cs[-1] == c_ref).double().mean())
        print(f"  [{name} seed {s}] done", flush=True)
    st = next(x for x in json.load(open(os.path.join(HERE, STORED[name])))['sites']
              if x['site'] == name)['methods']['sweep64']
    oracle_mean = float(site.oracle_se.mean())
    worst = max(abs(float(np.mean(rates[T])) - st['per_T'][str(T)]['mean']) for T in grid)
    eta_rec = round(float(np.mean(se_seed)) / oracle_mean, 4)
    gates = {"bit_for_bit": bit, "per_T_max_abs": worst, "eta": eta_rec, "eta_stored": st['eta'],
             "holds": bool(bit and worst <= TOL and eta_rec == st['eta'])}
    blocks = evaluate.block_ids(site.rx)
    return ({k: np.stack(v) for k, v in sel.items()}, site.oracle_se.double().cpu().numpy(),
            blocks, int(blocks.max()) + 1, gates,
            {k: float(np.mean(v)) for k, v in same.items()},
            {k: float(np.mean(v)) for k, v in gen_beam.items()},
            float((site.strat_snr < 0).mean()))


def test(selk, gse, blocks, nb, e, tag):
    """The paper's paired test of mean(se_selected - e se_genie) > 0, on one bootstrap stream."""
    rng = np.random.default_rng(evaluate.stable_seed('csirs', 'boot', tag))
    _, agg, D, tint = evaluate.paired_bootstrap_hier([selk - e * gse[None, :]], blocks, nb, rng)
    return {"diff": D, "ci95": [agg[0], agg[1]], "p": agg[2], "ci95_t": tint['ci95_t'],
            "p_t": tint['p_t']}


def passes_boot(r):
    """At most ALPHA_LB of the bootstrap replicates of the contrast at or below zero."""
    return r['diff'] > 0 and r['ci95'][0] > 0 and r['p'] / 2 <= ALPHA_LB


def lin_se(selk, gse, blocks, nb, eta, tag):
    """The delta-method standard error of eta, from the t interval's ingredients of the
    linearized residual, whose seed means are the per-seed efficiencies less eta."""
    rng = np.random.default_rng(evaluate.stable_seed('csirs', 'lin', tag))
    res = (selk - eta * gse[None, :]) / gse.mean()
    _, _, D, t = evaluate.paired_bootstrap_hier([res], blocks, nb, rng)
    assert abs(D) < 1e-9, (tag, D)
    return float(np.sqrt(t['seed_sd'] ** 2 / t['n_seeds'] + t['se_block'] ** 2))


def lower_bound(selk, gse, blocks, nb, eta, se_lin, tag):
    """eta_lb = min(e_boot, e_t), both one-sided at ALPHA_LB (see the module docstring)."""
    lo = max(0.0, eta - 0.25)
    grid = np.linspace(lo, eta, SCAN)
    ok = [passes_boot(test(selk, gse, blocks, nb, e, tag)) for e in grid]
    assert ok[0], (tag, "the scan's lowest point does not pass; widen the scan")
    flips = sum(a != b for a, b in zip(ok, ok[1:]))
    assert flips <= 1, (tag, "more than one pass-to-fail transition", ok)
    if all(ok):
        e_b = float(grid[-1])
    else:
        i = ok.index(False)
        a, b = float(grid[i - 1]), float(grid[i])
        while b - a > 1e-6:
            m = 0.5 * (a + b)
            a, b = (m, b) if passes_boot(test(selk, gse, blocks, nb, m, tag)) else (a, m)
        e_b = a
    e_t = eta - float(stats.t.ppf(1 - ALPHA_LB, selk.shape[0] - 1)) * se_lin
    return min(e_b, e_t), e_b, e_t, flips


def at_threshold(selk, gse, blocks, nb, eta, se_lin, thr, tag):
    """R12's two tests of eta > thr: the contrast's bootstrap and the delta-method t."""
    r = test(selk, gse, blocks, nb, thr, tag)
    dof = selk.shape[0] - 1
    r.update(eta_minus_thr=eta - thr, se_lin=se_lin,
             p_t=float(2 * stats.t.sf(abs(eta - thr) / se_lin, dof)),
             t_lo=eta - float(stats.t.ppf(0.975, dof)) * se_lin - thr)
    r.pop('ci95_t')
    return r


def main():
    check = '--check-only' in sys.argv
    seeds = protocol.seeds()
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    thr, tau = nr_threshold(grid)
    t0 = time.time()
    ub, ub_ok = unbiased()
    data = {n: scenario(n, seeds, t_probe, grid) for n in SCEN}
    gates = {"unbiased": {"holds": ub_ok, "draws": ub},
             **{n: data[n][4] for n in SCEN}}
    print(json.dumps({k: (v if k != 'unbiased' else v['holds']) for k, v in gates.items()},
                     indent=1))
    if not (ub_ok and all(data[n][4]['holds'] for n in SCEN)):
        raise SystemExit("gate failed: nothing tested, nothing written")
    if check:
        print("gates hold; nothing tested, nothing written")
        return
    out = {"script_sha256": hashlib.sha256(
               open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "seeds": list(seeds), "scenarios": SCEN, "n_rb": N_RB, "density_samples": DENSITY,
           "n_probe_model": N_PROBE, "t_probe_us": t_probe, "alpha_lb": ALPHA_LB,
           "threshold": {"implementation": "resnet18_vision|gpu-resident", "tau_us": tau,
                         "eta": thr},
           "rule": ("R12: at a configuration, the exclusion "
                    "of ResNet-18 on the GPU under the procedure is supported when eta > the "
                    "threshold is established in all seven scenarios: the contrast's bootstrap "
                    "interval and the delta-method t interval above it and the larger p-value "
                    "surviving Holm across the seven"),
           "gate": gates, "configs": {}}
    for key in [f"coherent|{n}" for n in COHERENT_N] + [f"per_re|{n}" for n in PER_RE_N]:
        rec, rows = {"per_scenario": {}}, []
        for n in SCEN:
            sel, gse, blocks, nb, _, same, gbeam, low = data[n]
            eta = float(sel[key].mean(1).mean() / gse.mean())
            se_lin = lin_se(sel[key], gse, blocks, nb, eta, f"{n}|{key}|lin")
            lb, e_b, e_t, flips = lower_bound(sel[key], gse, blocks, nb, eta, se_lin,
                                              f"{n}|{key}|lb")
            at = at_threshold(sel[key], gse, blocks, nb, eta, se_lin, thr, f"{n}|{key}|thr")
            r = {"eta": eta, "eta_seeds": (sel[key].mean(1) / gse.mean()).tolist(),
                 "se_lin": se_lin, "eta_lb": lb, "eta_lb_boot": e_b, "eta_lb_t": e_t,
                 "scan_transitions": flips, "same_beam_as_probe_model": same[key],
                 "genie_beam_share": gbeam[key], "share_best_snr_below_0dB": low,
                 "at_threshold": at}
            rec["per_scenario"][n] = r
            rows.append(at)
        rej = holm([max(r['p'], r['p_t']) for r in rows])
        for n, r, h in zip(SCEN, rows, rej):
            r["holm_rejects"] = bool(h)
            r["established"] = bool(h and r['ci95'][0] > 0 and r['t_lo'] > 0)
        ps = rec["per_scenario"].values()
        rec.update(eta_min=min(r['eta'] for r in ps), eta_lb_min=min(r['eta_lb'] for r in ps),
                   n_established=sum(r['at_threshold']['established'] for r in ps),
                   supported=all(r['at_threshold']['established'] for r in ps))
        out["configs"][key] = rec
        print(f"{key}: eta {rec['eta_min']:.4f}+, bound {rec['eta_lb_min']:.4f}, established "
              f"{rec['n_established']}/7", flush=True)
    sup = [n for n in COHERENT_N if out["configs"][f"coherent|{n}"]["supported"]]
    out["fewest_samples_supported"] = (min(sup) if sup and all(
        out["configs"][f"coherent|{n}"]["supported"] for n in COHERENT_N if n >= min(sup))
        else None)
    out["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, 'csirs.json')
    assert not os.path.exists(p), "csirs.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote csirs.json in {out['seconds']}s")


if __name__ == '__main__':
    main()
