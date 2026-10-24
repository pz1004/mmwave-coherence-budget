#!/usr/bin/env python3
"""The exhaustive sweep's efficiency under the aperiodic CSI-RS procedure as NR configures it.
Declared in protocol.json (analysis 'csirs_cfg_refbounds') before it ran.

The configuration (TS 38.214 V18.11.0 5.2.1.5.1, 5.2.1.6, 5.2.3, 5.4 and 5.2.1.4.3; TS 38.331
V18.11.0; read 2026-10-18).  An aperiodic NZP-CSI-RS resource set has one triggering offset, so its
resources share a slot; a trigger state selects one set per associated report setting; and a UE
supports at most four aperiodic beam-report settings per bandwidth part.  The 64 one-symbol
resources of the sweep, 12 per 125 us slot after two control symbols (make_draft's _nr_end), are
therefore six one-slot sets (12, 12, 12, 12, 12 and 4 resources) under two triggers, each set
measured by a cri-RSRP report setting.  The UE reports, per set, the resource with its largest
measured L1-RSRP, quantized to a 7-bit value in [-140, -44] dBm in 1 dB steps; the base station
compares the six reports.  Across reports the 1 dB steps can tie, which csirs.py does not
model: it compares the 64 measurements exactly.  This script adds that comparison and nothing else:
the measurement laws, streams and statistics are csirs.py's.

  sets        'spread': consecutive chunks (12, 12, 12, 12, 12, 4) of obs.spread_order(), the order
              the sweep probes, the only order in which the reduced set's four resources are the
              sweep's first four; 'contiguous': beams 0-11, 12-23, ..., 60-63, which a deployment
              running the sweep alone could use
  UE          per set, the argmax of the measurement y over the set, ties to the earliest spread
              position (methods.topk's rule)
  L1-RSRP     10 log10(y / N_SC) + 30 dBm, y in watts estimating P_TX pbar (sim.measure), the data's
              energy per element over N_SC = 12 x 66 subcarriers
  quantizer   q = clamp(floor(L1-RSRP), -141, -44): 1 dB bins [k, k+1), one bin below -140 and one
              at -44 and above (the model's grid: the mapping table of TS 38.133 was not read)
  base station  the largest q; ties to the set winner with the lowest genie-aided rate ('worst', the
              lower envelope of every rule the base station could use) or to the earliest set
              ('first', descriptive); 'exact' is csirs.py's choice, recomputed as gate G2

Per variant ('spread|worst', 'contiguous|worst'), configuration and scenario, in csirs.json's schema:
eta, the per-seed efficiencies, the delta-method standard error (csirs.lin_se), the one-sided lower
bound at csirs.ALPHA_LB = 0.05/7 (effbound.lower_bound, its scan from zero), rule R12's test at the CSI-RS
threshold (csirs.at_threshold, Holm across the seven scenarios), and descriptive shares: users whose
top report is tied in range, users whose six reports all fall in the bottom bin, users served on the
exact choice's beam, the seed share of variance of R12's contrast (calib_holm.gate3's formula).
Bootstrap streams are csirs.py's (tags '{scenario}|{config}|lin', '|lb', '|thr'), so every variant
is paired with csirs.py's exact choice.  Descriptive only: the 'first' rule, and 'spread|worst' with the
1 dB grid shifted by 0.25, 0.5 and 0.75 dB (where the grid falls relative to the users depends on
the assumed transmit power).

Gates, before anything is written: G1 csirs.py's gates (i) and (ii); G2 the exact choice reproduces
csirs.json in every configuration and scenario (floats to 1e-12, flags exactly); G3 each assignment
partitions the 64 beams as declared and the unquantized six-set choice equals methods.topk's; G4
the quantizer's unit cases; G5 per user the 'worst' rate is at most the exact choice's, the exact
winner is among the tied sets, and the 'worst' bootstrap bound is at most the exact one plus 1e-6;
G6 every R12 contrast's seed share below 10%, the formula reproducing calib_holm.json's value on
csirs.json; G7 the scan's own asserts.  --check-only runs G1-G4, G5's per-user part and G6's
reproduction, and computes no statistic of a configured variant.

usage:  python3 csirs_cfg.py [--check-only]       -> csirs_cfg.json
"""
import hashlib, json, os, sys, time
import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim, obs, methods, evaluate, protocol, csirs, effbound
from reading_rules import holm

SCEN = csirs.SCEN
L6 = obs.LAYER6_OFFSET
N_SC = 12 * csirs.N_RB            # 792 subcarriers of a 100 MHz carrier at 120 kHz
RSRP_LO, RSRP_HI = -140, -44      # TS 38.214 5.2.1.4.3: 7 bits over [-140, -44] dBm, 1 dB steps
SETS = (12, 12, 12, 12, 12, 4)    # one aperiodic set per slot
ASSIGN = ('spread', 'contiguous')
VARIANTS = tuple(f"{a}|worst" for a in ASSIGN)
WINDOW = 1.0                      # the scan starts at zero, where every efficiency's test passes
OFFSETS = (0.25, 0.5, 0.75)
KEYS = [f"coherent|{n}" for n in csirs.COHERENT_N] + [f"per_re|{n}" for n in csirs.PER_RE_N]
TOL_EQ = 1e-12
POS = np.empty(64, np.int64)
POS[obs.spread_order()] = np.arange(64)     # each beam's position in the frozen spread order


def sets_of(assign):
    """The six sets of beam indices, each in the frozen spread order."""
    order = np.asarray(obs.spread_order()) if assign == 'spread' else np.arange(64)
    out, k = [], 0
    for m in SETS:
        st = np.asarray(order[k:k + m])
        out.append(st[np.argsort(POS[st], kind='stable')])
        k += m
    assert sorted(np.concatenate(out).tolist()) == list(range(64)), assign
    return out


def rsrp_dbm(y):
    """Per-element L1-RSRP in dBm of a band power y in watts."""
    return 10.0 * torch.log10(y / N_SC) + 30.0


def quantize(r, offset=0.0):
    """1 dB bins [k, k+1) of the grid shifted by offset dB; one bin below RSRP_LO, one at RSRP_HI
    and above."""
    return torch.clamp(torch.floor(r - offset), RSRP_LO - 1, RSRP_HI)


def quantizer_cases():
    """Gate G4."""
    t = lambda v: torch.tensor(v, dtype=torch.float64)
    cases = [(-140.0, -140), (-140.0 - 1e-9, -141), (-44.0, -44), (-44.5, -45), (-43.0, -44),
             (-200.0, -141), (-90.2, -91)]
    ok = all(int(quantize(t(v))) == w for v, w in cases)
    ok &= abs(float(rsrp_dbm(t(0.792)))) < 1e-12           # 0.792 W over 792 elements: 0 dBm
    r = torch.linspace(-160, -30, 100001, dtype=torch.float64)
    ok &= bool((quantize(r)[1:] >= quantize(r)[:-1]).all())   # monotone
    return bool(ok)


def winners(yn, sets):
    """Per set, the UE's strongest resource and its measurement: (n, 6) each."""
    W, Yw = [], []
    for st in sets:
        idx = torch.tensor(st, device=sim.DEV)
        a = yn[:, idx].argmax(1)                  # the first maximum: the earliest spread position
        W.append(idx[a])
        Yw.append(yn[:, idx].gather(1, a[:, None])[:, 0])
    return torch.stack(W, 1), torch.stack(Yw, 1)


def choose(yn, sn, sets, rule, offset=0.0):
    """The base station's choice of beam from the six reports.  Returns (beam, set, tied, in range):
    the chosen beam (n,), the set it came from, the number of sets sharing the top code and whether
    that code lies in range (None for 'exact')."""
    W, Yw = winners(yn, sets)
    pos = torch.tensor(POS, device=sim.DEV)[W]
    if rule == 'exact':                           # the largest measurement, ties to spread order
        cand = Yw == Yw.max(1, keepdim=True).values
        k = torch.where(cand, pos, torch.full_like(pos, 10 ** 6)).argmin(1)
        return W.gather(1, k[:, None])[:, 0], k, None, None
    Q = quantize(rsrp_dbm(Yw), offset)
    qmax = Q.max(1, keepdim=True).values
    tied = Q == qmax
    if rule == 'worst':                           # the lowest genie-aided rate among the tied
        sew = sn.gather(1, W).double()
        k = torch.where(tied, sew, torch.full_like(sew, float('inf'))).argmin(1)
    elif rule == 'first':                         # the earliest tied set
        k = tied.to(torch.int64).argmax(1)
    else:
        raise ValueError(rule)
    return W.gather(1, k[:, None])[:, 0], k, tied.sum(1), qmax[:, 0] >= RSRP_LO


def seed_share(ps, S, chi):
    """calib_holm.gate3's seed share of variance of one R12 contrast (99.9% upper bound)."""
    at = ps['at_threshold']
    gbar = abs(at['diff'] / at['eta_minus_thr'])
    sd = gbar * float(np.std(ps['eta_seeds'], ddof=1))
    v_seed = (S - 1) * sd ** 2 / chi / S
    v_boot = ((at['ci95'][1] - at['ci95'][0]) / 3.92) ** 2
    return v_seed / (v_seed + v_boot)


def scenario(name, seeds, t_probe, grid):
    """Per seed and configuration, the selected rates of the exact choice and of each variant; the
    descriptive efficiencies; and gates G1, G3 and G5's per-user part."""
    site = sim.Site(name)
    SETS_OF = {a: sets_of(a) for a in ASSIGN}
    sel = {v: {} for v in ('exact',) + VARIANTS}
    desc = {}
    tie, bottom, same_ex, same_pr, gbeam = {}, {}, {}, {}, {}
    bit, g3, g5, rates, se_seed = True, True, True, {}, []
    for s in seeds:
        y_ref = sim.measure(site.pbar, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        z, q = csirs.draws(site, torch.Generator(device=sim.DEV).manual_seed(
            evaluate.stable_seed(name, s)))
        rng = np.random.default_rng(evaluate.stable_seed('csirs', 'per_re', name, s))
        pbar64 = site.pbar[:, L6:].cpu().numpy()
        c_ref, chosen = None, {}
        for est, ns in ([('coherent', n) for n in csirs.COHERENT_N]
                        + [('per_re', n) for n in csirs.PER_RE_N]):
            if est == 'coherent':
                y = csirs.coherent(site.pbar, z, q, ns)
                if ns == csirs.N_PROBE:
                    bit &= bool(torch.equal(y, y_ref))
            else:
                y = torch.zeros_like(site.pbar)
                y[:, L6:] = torch.tensor(csirs.per_re(pbar64, rng, ns), device=sim.DEV)
            key = f"{est}|{ns}"
            d, c = methods.topk(site, y, 64, t_probe)          # csirs.py's choice
            yn = y[:, L6:]
            se_x = site.se.gather(1, c[:, None])[:, 0].double()
            sel['exact'].setdefault(key, []).append(se_x.cpu().numpy())
            if est == 'coherent' and ns == csirs.N_PROBE:
                c_ref = c
                se_seed.append(float(se_x.mean()))
                for T in grid:
                    rates.setdefault(T, []).append(
                        float((torch.clamp(1 - d / T, min=0.0) * se_x).mean()))
            chosen[key] = {}
            for a in ASSIGN:
                b_ex, k_ex, _, _ = choose(yn, site.sn, SETS_OF[a], 'exact')
                g3 &= bool(torch.equal(L6 + b_ex, c))
                b_w, _, ntie, inr = choose(yn, site.sn, SETS_OF[a], 'worst')
                se_w = site.se.gather(1, (L6 + b_w)[:, None])[:, 0].double()
                # G5 (per user): never better than the exact choice, whose set is among the tied
                Q = quantize(rsrp_dbm(winners(yn, SETS_OF[a])[1]))
                g5 &= bool((se_w <= se_x + 1e-12).all()) and bool(
                    (Q.gather(1, k_ex[:, None])[:, 0] == Q.max(1).values).all())
                v = f"{a}|worst"
                sel[v].setdefault(key, []).append(se_w.cpu().numpy())
                chosen[key][v] = L6 + b_w
                tie.setdefault((v, key), []).append(float(((ntie >= 2) & inr).double().mean()))
                bottom.setdefault((v, key), []).append(float((~inr).double().mean()))
                same_ex.setdefault((v, key), []).append(float(((L6 + b_w) == c).double().mean()))
                gbeam.setdefault((v, key), []).append(
                    float(((L6 + b_w) == site.oracle_beam + L6).double().mean()))
                b_f, _, _, _ = choose(yn, site.sn, SETS_OF[a], 'first')
                desc.setdefault((f"{a}|first", key), []).append(
                    float(site.se.gather(1, (L6 + b_f)[:, None])[:, 0].double().mean()))
                if a == 'spread':
                    for off in OFFSETS:
                        b_o, _, _, _ = choose(yn, site.sn, SETS_OF[a], 'worst', off)
                        desc.setdefault((f"spread|worst|offset{off:g}", key), []).append(
                            float(site.se.gather(1, (L6 + b_o)[:, None])[:, 0].double().mean()))
        for key, by_v in chosen.items():
            for v, cv in by_v.items():
                same_pr.setdefault((v, key), []).append(float((cv == c_ref).double().mean()))
        print(f"  [{name} seed {s}] done", flush=True)
    st = next(x for x in json.load(open(os.path.join(HERE, csirs.STORED[name])))['sites']
              if x['site'] == name)['methods']['sweep64']
    oracle_mean = float(site.oracle_se.mean())
    worst_T = max(abs(float(np.mean(rates[T])) - st['per_T'][str(T)]['mean']) for T in grid)
    eta_rec = round(float(np.mean(se_seed)) / oracle_mean, 4)
    gates = {"bit_for_bit": bit, "per_T_max_abs": worst_T, "eta": eta_rec, "eta_stored": st['eta'],
             "csirs_gates": bool(bit and worst_T <= csirs.TOL and eta_rec == st['eta']),
             "sets_and_exact_choice": g3, "worst_never_better": g5}
    blocks = evaluate.block_ids(site.rx)
    gse = site.oracle_se.double().cpu().numpy()
    shares = {k: float(np.mean(v)) for k, v in
              [(f"{v}|{key}|tie", tie[(v, key)]) for v, key in tie]
              + [(f"{v}|{key}|bottom", bottom[(v, key)]) for v, key in bottom]
              + [(f"{v}|{key}|same_exact", same_ex[(v, key)]) for v, key in same_ex]
              + [(f"{v}|{key}|same_probe", same_pr[(v, key)]) for v, key in same_pr]
              + [(f"{v}|{key}|genie", gbeam[(v, key)]) for v, key in gbeam]}
    d_eta = {f"{k}|{key}": float(np.mean(v)) / float(gse.mean()) for (k, key), v in desc.items()}
    return ({v: {k: np.stack(a) for k, a in by.items()} for v, by in sel.items()}, gse, blocks,
            int(blocks.max()) + 1, gates, shares, d_eta)


def stats_of(selk, gse, blocks, nb, n, key, thr, window):
    """csirs.py's statistics for one configuration and scenario, on its streams."""
    eta = float(selk.mean(1).mean() / gse.mean())
    se_lin = csirs.lin_se(selk, gse, blocks, nb, eta, f"{n}|{key}|lin")
    lb, e_b, e_t, flips = effbound.lower_bound(selk, gse, blocks, nb, eta, se_lin,
                                               f"{n}|{key}|lb", csirs.ALPHA_LB, window)
    at = csirs.at_threshold(selk, gse, blocks, nb, eta, se_lin, thr, f"{n}|{key}|thr")
    return {"eta": eta, "eta_seeds": (selk.mean(1) / gse.mean()).tolist(), "se_lin": se_lin,
            "eta_lb": lb, "eta_lb_boot": e_b, "eta_lb_t": e_t, "scan_transitions": flips,
            "at_threshold": at}


def holm_r12(rec):
    """R12 across the seven scenarios, as csirs.py reads it."""
    rows = [r['at_threshold'] for r in rec.values()]
    rej = holm([max(r['p'], r['p_t']) for r in rows])
    for r, h in zip(rows, rej):
        r["holm_rejects"] = bool(h)
        r["established"] = bool(h and r['ci95'][0] > 0 and r['t_lo'] > 0)
    return {"per_scenario": rec, "eta_min": min(r['eta'] for r in rec.values()),
            "eta_lb_min": min(r['eta_lb'] for r in rec.values()),
            "n_established": sum(r['at_threshold']['established'] for r in rec.values()),
            "supported": all(r['at_threshold']['established'] for r in rec.values())}


def fewest(configs):
    sup = [n for n in csirs.COHERENT_N if configs[f"coherent|{n}"]["supported"]]
    return (min(sup) if sup and all(configs[f"coherent|{n}"]["supported"]
                                    for n in csirs.COHERENT_N if n >= min(sup)) else None)


def same(a, b, path=''):
    """G2: a recomputed record equals csirs.json's, floats to TOL_EQ, everything else exactly."""
    if isinstance(b, dict):
        return all(same(a[k], b[k], f"{path}.{k}") for k in b)
    if isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, path) for x, y in zip(a, b))
    if isinstance(b, float) and not isinstance(a, bool):
        return abs(a - b) <= TOL_EQ
    return a == b


def main():
    check = '--check-only' in sys.argv
    seeds = protocol.seeds()
    grid = json.load(open(os.path.join(HERE, 'protocol.json')))['coherence_grid_us']
    t_probe = evaluate.charges(json.load(open(os.path.join(HERE, 'lat_cpu1t_pcore.json'))))[
        'T_PROBE']
    CS = json.load(open(os.path.join(HERE, 'csirs.json')))
    thr, tau = csirs.nr_threshold(grid)
    assert thr == CS['threshold']['eta'] and tau == CS['threshold']['tau_us']
    assert list(CS['seeds']) == list(seeds) and CS['scenarios'] == SCEN
    t0 = time.time()
    g4 = quantizer_cases()
    S, chi = len(seeds), stats.chi2.ppf(0.001, len(seeds) - 1)
    CH = json.load(open(os.path.join(HERE, 'calib_holm.json')))
    g6_repro = max(seed_share(ps, S, chi) for cf in CS['configs'].values()
                   for ps in cf['per_scenario'].values())
    g6_ok = abs(g6_repro - CH['gates']['r12_seed_share_max_upper']) < 1e-15
    data, g2, stat = {}, True, {v: {k: {} for k in KEYS} for v in VARIANTS}
    exact = {k: {} for k in KEYS}
    for n in SCEN:
        sel, gse, blocks, nb, gates, shares, d_eta = scenario(n, seeds, t_probe, grid)
        for key in KEYS:                  # G2: csirs.json's record, recomputed
            r = stats_of(sel['exact'][key], gse, blocks, nb, n, key, thr, 0.25)
            exact[key][n] = r
            ref = CS['configs'][key]['per_scenario'][n]
            g2 &= same({k: r[k] for k in r if k != 'at_threshold'},
                       {k: ref[k] for k in r if k != 'at_threshold'})
            g2 &= same(r['at_threshold'], {k: ref['at_threshold'][k] for k in r['at_threshold']})
        if not check:
            for v in VARIANTS:
                for key in KEYS:
                    r = stats_of(sel[v][key], gse, blocks, nb, n, key, thr, WINDOW)
                    assert r['eta_lb_boot'] <= exact[key][n]['eta_lb_boot'] + 1e-6, (v, key, n)
                    r.update(same_beam_as_exact=shares[f"{v}|{key}|same_exact"],
                             same_beam_as_probe_model=shares[f"{v}|{key}|same_probe"],
                             genie_beam_share=shares[f"{v}|{key}|genie"],
                             tie_share_in_range=shares[f"{v}|{key}|tie"],
                             bottom_bin_share=shares[f"{v}|{key}|bottom"])
                    stat[v][key][n] = r
        del sel                           # the per-user arrays are not kept across scenarios
        data[n] = (None, gse, blocks, nb, gates, shares, d_eta)
    for key in KEYS:                      # G2's Holm step and the configuration summaries
        ref = CS['configs'][key]
        h = holm_r12(exact[key])
        g2 &= all(h['per_scenario'][n]['at_threshold'][f] == ref['per_scenario'][n]
                  ['at_threshold'][f] for n in SCEN for f in ('holm_rejects', 'established'))
        g2 &= (h['supported'] == ref['supported'] and h['n_established'] == ref['n_established'])
    gate = {"G1_csirs_gates": all(data[n][4]['csirs_gates'] for n in SCEN),
            "G2_exact_reproduces_csirs_json": bool(g2),
            "G3_sets_and_exact_choice": all(data[n][4]['sets_and_exact_choice'] for n in SCEN),
            "G4_quantizer": g4,
            "G5_worst_never_better": all(data[n][4]['worst_never_better'] for n in SCEN),
            "G6_formula_reproduces_calib_holm": bool(g6_ok),
            "per_scenario": {n: data[n][4] for n in SCEN}}
    print(json.dumps({k: v for k, v in gate.items() if k != 'per_scenario'}, indent=1))
    if not all(v for k, v in gate.items() if k != 'per_scenario'):
        raise SystemExit("gate failed: nothing written")
    if check:
        print(f"gates hold; no configured variant's statistic computed, nothing written "
              f"({time.time() - t0:.0f} s)")
        return
    out = {"script_sha256": hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(),
           "csirs_json_sha256": hashlib.sha256(
               open(os.path.join(HERE, 'csirs.json'), 'rb').read()).hexdigest(),
           "seeds": list(seeds), "scenarios": SCEN, "set_sizes": list(SETS),
           "assignments": {a: [s.tolist() for s in sets_of(a)] for a in ASSIGN},
           "quantizer": {"n_sc": N_SC, "lo_dbm": RSRP_LO, "hi_dbm": RSRP_HI,
                         "bins": "floor to 1 dB, one bin below lo and one at hi and above"},
           "threshold": CS['threshold'], "alpha_lb": csirs.ALPHA_LB, "window": WINDOW,
           "rule": ("R16: the paper's CSI-RS claims take, per configuration, the lowest of "
                    "csirs.py's exact comparison and the two 'worst' variants; R12 holds only if "
                    "it holds in all three"),
           "gate": gate, "variants": {}, "descriptive": {}}
    worst_share = 0.0
    for v in VARIANTS:
        cfgs = {}
        for key in KEYS:
            cfgs[key] = holm_r12(stat[v][key])
            for ps in cfgs[key]['per_scenario'].values():
                ps['seed_share_upper'] = seed_share(ps, S, chi)
                ps['p_ge_p_t'] = bool(ps['at_threshold']['p'] >= ps['at_threshold']['p_t'])
                worst_share = max(worst_share, ps['seed_share_upper'])
            print(f"{v} {key}: eta {cfgs[key]['eta_min']:.4f}+, bound "
                  f"{cfgs[key]['eta_lb_min']:.4f}, established {cfgs[key]['n_established']}/7",
                  flush=True)
        out["variants"][v] = {"configs": cfgs, "fewest_samples_supported": fewest(cfgs)}
    assert worst_share < 0.10, ("G6: an R12 contrast's seed share reaches 10%", worst_share)
    out["gate"]["G6_seed_share_max_upper"] = worst_share
    for n in SCEN:
        for k, e in data[n][6].items():
            nm, key = k.rsplit('|', 2)[0], '|'.join(k.rsplit('|', 2)[1:])
            out["descriptive"].setdefault(nm, {}).setdefault(key, {})[n] = e
    out["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, 'csirs_cfg.json')
    assert not os.path.exists(p), "csirs_cfg.json exists: it is not rewritten"
    json.dump(out, open(p, 'w'), indent=1)
    print(f"wrote csirs_cfg.json in {out['seconds']}s")


if __name__ == '__main__':
    main()
