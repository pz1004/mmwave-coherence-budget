"""Phase 3 site qualification.  Rules frozen before any count was taken.

R1 path validity.  A path is valid iff power, phase, delay, aod_el and aod_az are
   all finite.  A receiver whose finite-masks disagree across those five fields is
   structurally inconsistent; such receivers are counted and excluded.

R2 receiver eligibility.  Receiver u is eligible for transmitter t iff it is
   structurally consistent, has at least one valid path, has a finite position, and
   yields a finite narrowband channel.  No threshold on gain, SNR or beam outcome
   enters.  Receivers the ray tracer left with zero paths carry no channel to probe;
   they are counted separately and reported, never silently dropped.

R3 transmitter choice.  Each site contributes exactly one transmitter: the lowest
   tx_set_id whose eligible count reaches 5,000 (the per-role sample size in
   pilot-design.md).  If none reaches it, the site fails.

R4 site qualification.  Archive extracted; every advertised TX-RX pair loaded; the
   F2 channel identity holds on a 64-receiver sample from this site; metadata matches
   the catalog (28 GHz, Wireless InSite 3.3); and R3 returns a transmitter.

R5 order and roles.  Catalog numerical order.  Among qualifying sites: 1st training,
   2nd validation, 3rd test A, 4th test B, 5th and later additional held-out tests.

Gate: at least four sites qualify.  Failures are dropped, never substituted.
"""
from __future__ import annotations

import hashlib, json, os, sys, time
import numpy as np
import deepmimo as dm

import g1

HERE = os.path.dirname(os.path.abspath(__file__))
dm.config.set('scenarios_folder', os.path.join(HERE, 'scenarios'))
SITES = json.load(open(os.path.join(HERE, 'acquisition.json')))['frozen_order']
MIN_ELIGIBLE = 5000
RNG_SEED = 881
DF_GRID = np.linspace(-g1.BANDWIDTH / 2, g1.BANDWIDTH / 2, 9)   # 100 MHz, centred
N_BAND_SUB = 2000


def pct(a, qs=(1, 5, 25, 50, 75, 95, 99)):
    return {q: round(float(np.percentile(a, q)), 2) for q in qs}


out = {"rules": "R1-R5 in module docstring", "min_eligible": MIN_ELIGIBLE,
       "sites": []}

for num, name in enumerate(SITES):
    t0 = time.time()
    rec = {"order": num, "scenario": name}
    try:
        macro = dm.load(name)
    except Exception as e:                       # keep the failure, do not retry
        rec.update(loaded=False, error=f"{type(e).__name__}: {e}", qualified=False)
        out["sites"].append(rec); print(rec); continue
    rec["loaded"] = True
    rec["n_tx_sets"] = len(macro)
    d0 = macro[0]
    rec["carrier_Hz"] = float(d0.rt_params['frequency'])
    rec["raytracer"] = f"{d0.rt_params['raytracer_name']} {d0.rt_params['raytracer_version']}"
    rec["meta_ok"] = (abs(rec["carrier_Hz"] - 28e9) < 1e6
                      and d0.rt_params['raytracer_version'] == '3.3')

    txs = []
    for ti in range(len(macro)):
        d = macro[ti]
        pw, ph = np.asarray(d.power), np.asarray(d.phase)
        el, az = np.asarray(d.aod_el), np.asarray(d.aod_az)
        dly, inter = np.asarray(d.delay), np.asarray(d.inter)
        rx = np.asarray(d.rx_pos)
        masks = [np.isfinite(x) for x in (pw, ph, dly, el, az)]
        consistent = np.ones(pw.shape[0], bool)
        for m in masks[1:]:
            consistent &= (m == masks[0]).all(1)
        n_valid = masks[0].sum(1)
        zero_path = (n_valid == 0)
        cand = np.flatnonzero(consistent & ~zero_path & np.isfinite(rx).all(1))
        h = g1.narrowband(pw[cand], ph[cand], el[cand], az[cand])
        good = np.isfinite(h).all(1)
        elig = cand[good]
        bp = g1.beam_power(h[good])
        gam = 10 * np.log10(np.maximum(g1.snr(bp.max(1)), 1e-30))
        t = {"tx_set_id": int(d.txrx['tx_set_id']), "tx_index": ti,
             "n_rx_total": int(pw.shape[0]),
             "n_zero_path": int(zero_path.sum()),
             "n_structurally_inconsistent": int((~consistent).sum()),
             "n_nonfinite_channel": int((~good).sum()),
             "n_eligible": int(elig.size),
             "n_los": int((inter[elig, 0] == 0).sum()),
             "median_paths": int(np.median(n_valid[elig])),
             "best_beam_snr_dB_pct": pct(gam),
             "frac_best_beam_below_0dB": round(float((gam < 0).mean()), 4),
             "eligible_sha256": hashlib.sha256(elig.astype(np.int64).tobytes()).hexdigest()[:16]}
        txs.append(t)
        del h, bp
    rec["transmitters"] = txs

    # R3
    ok = [t for t in txs if t["n_eligible"] >= MIN_ELIGIBLE]
    chosen = min(ok, key=lambda t: t["tx_set_id"]) if ok else None
    rec["chosen_tx_set_id"] = chosen["tx_set_id"] if chosen else None
    rec["chosen_tx_index"] = chosen["tx_index"] if chosen else None

    # per-site decode identity (F2 on this site's own data) --------------
    d = macro[rec["chosen_tx_index"] or 0]
    pw, ph = np.asarray(d.power), np.asarray(d.phase)
    el, az, dly = np.asarray(d.aod_el), np.asarray(d.aod_az), np.asarray(d.delay)
    act = np.flatnonzero(np.isfinite(pw).sum(1) > 0)
    rng = np.random.default_rng(RNG_SEED)
    sub = rng.choice(act, size=min(64, act.size), replace=False)
    p = dm.ChannelParameters()
    p.bs_antenna.shape = np.array(g1.ANT_SHAPE); p.bs_antenna.spacing = g1.ANT_SPACING
    p.ue_antenna.shape = np.array([1, 1]); p.freq_domain = 1
    p.ofdm.subcarriers = 512; p.ofdm.selected_subcarriers = np.arange(1)
    p.ofdm.bandwidth = g1.BANDWIDTH; p.num_paths = pw.shape[1]
    H = d.compute_channels(p)
    h_dm = np.sqrt(512) * H[sub, 0, :, 0].astype(np.complex128)
    elr = np.rad2deg(np.asarray(d['_aod_el_rot'])[sub])
    azr = np.rad2deg(np.asarray(d['_aod_az_rot'])[sub])
    e = np.abs(h_dm - g1.narrowband(pw[sub], ph[sub], elr, azr)).max() / np.abs(h_dm).max()
    rec["decode_rel_err"] = float(f"{e:.3e}")
    rec["decode_ok"] = bool(e < 1e-6)
    del H, h_dm

    # narrowband adequacy over the 100 MHz band, this site ----------------
    sub2 = rng.choice(act, size=min(N_BAND_SUB, act.size), replace=False)
    hb = np.stack([g1.beam_power(g1.narrowband_at(pw[sub2], ph[sub2], el[sub2],
                                                  az[sub2], dly[sub2], df))
                   for df in DF_GRID], -1)                      # (B, M, K)
    b0 = hb[:, :, len(DF_GRID) // 2].argmax(1)                  # centre frequency
    tot = hb.sum(-1)
    loss = 10 * np.log10(tot.max(1) / tot[np.arange(tot.shape[0]), b0])
    w = np.where(np.isfinite(pw[sub2]), 10 ** (pw[sub2] / 10), 0.0)
    tt = np.where(np.isfinite(dly[sub2]), dly[sub2], 0.0)
    mu = (w * tt).sum(1) / w.sum(1)
    rms = np.sqrt((w * (tt - mu[:, None]) ** 2).sum(1) / w.sum(1))
    rec["band"] = {"argmax_agreement": round(float((hb.argmax(1) == b0[:, None]).mean()), 4),
                   "median_loss_dB": round(float(np.median(loss)), 3),
                   "p95_loss_dB": round(float(np.percentile(loss, 95)), 3),
                   "median_rms_delay_ns": round(float(np.median(rms) * 1e9), 2),
                   "coherence_bw_MHz": round(float(1 / (5 * np.median(rms)) / 1e6), 2)}
    del hb, tot

    rec["qualified"] = bool(rec["meta_ok"] and rec["decode_ok"] and chosen is not None)
    rec["seconds"] = round(time.time() - t0, 1)
    out["sites"].append(rec)
    print(f"[{num}] {name}: qualified={rec['qualified']} "
          f"tx={rec['chosen_tx_set_id']} elig={chosen['n_eligible'] if chosen else 0} "
          f"decode={rec['decode_rel_err']} {rec['seconds']}s", flush=True)
    del macro

q = [s for s in out["sites"] if s.get("qualified")]
roles = ["training", "validation", "test_A", "test_B"] + \
        [f"test_extra_{i}" for i in range(1, 9)]
for i, s in enumerate(q):
    s["role"] = roles[i]
out["n_qualified"] = len(q)
out["gate_pass"] = len(q) >= 4
out["frozen_split"] = [{"order": s["order"], "scenario": s["scenario"],
                        "tx_set_id": s["chosen_tx_set_id"], "role": s["role"]}
                       for s in q]
json.dump(out, open(os.path.join(HERE, 'qualification.json'), 'w'), indent=1)
print(f"\nqualified {len(q)}/{len(SITES)}  gate={'PASS' if out['gate_pass'] else 'FAIL'}")
