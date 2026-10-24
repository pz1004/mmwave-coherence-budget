#!/usr/bin/env python3
"""Build the G2 band-averaged dataset for every qualified site.

Eligibility is recomputed with Phase 3's rule R2 verbatim and the resulting
index array is checked against the sha256 Phase 3 recorded, so this is provably
the same receiver set -- not a re-derivation that happens to agree.

Per site and per eligible receiver we store, for all 126 codewords:
    pbar[u,c] = mean_j |h_j^T f_c|^2                 (what a 100 MHz pilot sees)
    se[u,c]   = mean_j log2(1 + P_tx |h_j^T f_c|^2 / sigma_d^2)   (what it earns)
Nothing else is needed: the G2 measurement law is a function of pbar alone.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np

# Re-runnable: if every dataset is present and matches the sha256 the manifest
# recorded when it was built, it is not rebuilt.  The build needs the DeepMIMO
# library and the licensed scenario files, neither of which a re-run of the
# later stages should require; a dataset that fails its hash is rebuilt.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, 'manifest.json')
if os.path.exists(_MAN):
    _want = json.load(open(_MAN)).get('dataset_sha256', {})
    _ok = _want and all(
        os.path.exists(os.path.join(_HERE, 'data', f)) and
        hashlib.sha256(open(os.path.join(_HERE, 'data', f), 'rb').read()).hexdigest()
        == d['sha256'] for f, d in _want.items())
    if _ok:
        print(f"{len(_want)} datasets present and match the manifest's sha256; not rebuilt")
        sys.exit(0)

import deepmimo as dm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1, obs

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.abspath(os.path.join(HERE, '..', 'sites'))
OUT = os.path.join(HERE, 'data'); os.makedirs(OUT, exist_ok=True)
dm.config.set('scenarios_folder', os.path.join(P3, 'scenarios'))

qual = json.load(open(os.path.join(P3, 'qualification.json')))
DF = obs.subband_offsets()
F = obs.hierarchical_codebook()                      # (64, 126)
CHUNK = 2048
rec = {"k_f": obs.K_F, "subband_offsets_Hz": DF.tolist(), "sites": []}

for site in qual['sites']:
    name, ti = site['scenario'], site['chosen_tx_index']
    t0 = time.time()
    macro = dm.load(name)
    d = macro[ti]
    pw, ph = np.asarray(d.power), np.asarray(d.phase)
    el, az = np.asarray(d.aod_el), np.asarray(d.aod_az)
    dly, inter = np.asarray(d.delay), np.asarray(d.inter)
    rx = np.asarray(d.rx_pos)

    # --- Phase 3 rule R2, verbatim ------------------------------------------
    masks = [np.isfinite(x) for x in (pw, ph, dly, el, az)]
    consistent = np.ones(pw.shape[0], bool)
    for m in masks[1:]:
        consistent &= (m == masks[0]).all(1)
    zero_path = masks[0].sum(1) == 0
    cand = np.flatnonzero(consistent & ~zero_path & np.isfinite(rx).all(1))
    h0 = g1.narrowband(pw[cand], ph[cand], el[cand], az[cand])
    elig = cand[np.isfinite(h0).all(1)]
    del h0
    digest = hashlib.sha256(elig.astype(np.int64).tobytes()).hexdigest()[:16]
    want = [t for t in site['transmitters'] if t['tx_index'] == ti][0]
    assert digest == want['eligible_sha256'], (name, digest, want['eligible_sha256'])
    n = elig.size

    pwe, phe = pw[elig], ph[elig]
    ele, aze, dle = el[elig], az[elig], dly[elig]
    pbar = np.zeros((n, obs.N_CODE)); se = np.zeros((n, obs.N_CODE))
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        acc_p = np.zeros((e - s, obs.N_CODE)); acc_s = np.zeros((e - s, obs.N_CODE))
        for df in DF:
            h = g1.narrowband_at(pwe[s:e], phe[s:e], ele[s:e], aze[s:e],
                                 dle[s:e], float(df), chunk=CHUNK)
            p = np.abs(h @ F) ** 2
            acc_p += p
            acc_s += np.log2(1.0 + obs.P_TX * p / obs.SIGMA_D2)
        pbar[s:e] = acc_p / obs.K_F
        se[s:e] = acc_s / obs.K_F
    assert np.isfinite(pbar).all() and np.isfinite(se).all()

    np.savez_compressed(
        os.path.join(OUT, f"{name}.npz"),
        pbar=pbar.astype(np.float32), se=se.astype(np.float32),
        rx_pos=rx[elig].astype(np.float32),
        los=(inter[elig, 0] == 0), elig=elig.astype(np.int64))
    nb = pbar[:, obs.LAYER6_OFFSET:]
    g = obs.snr_db(nb.max(1))
    r = {"scenario": name, "role": site.get('role'), "tx_set_id": site['chosen_tx_set_id'],
         "n": int(n), "eligible_sha256": digest, "n_los": int((inter[elig, 0] == 0).sum()),
         "oracle_snr_dB_pct": {q: round(float(np.percentile(g, q)), 2)
                               for q in (1, 5, 25, 50, 75, 95, 99)},
         "frac_below_0dB": round(float((g < 0).mean()), 4),
         "oracle_se_mean": round(float(se[:, obs.LAYER6_OFFSET:].max(1).mean()), 4),
         "seconds": round(time.time() - t0, 1)}
    rec['sites'].append(r)
    print(f"[{site['order']}] {name:24s} n={n:6d} digest ok  "
          f"oracle SNR p50={r['oracle_snr_dB_pct'][50]:+7.2f} dB  "
          f"SE={r['oracle_se_mean']:.3f}  {r['seconds']:.0f}s", flush=True)
    del macro, d, pbar, se

json.dump(rec, open(os.path.join(HERE, 'dataset.json'), 'w'), indent=1)
print("\ndataset built:", sum(s['n'] for s in rec['sites']), "receivers over",
      len(rec['sites']), "sites")
