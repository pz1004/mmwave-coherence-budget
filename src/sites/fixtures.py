"""Phase 3 channel-unit fixtures.

Every check is computed by hand from the documented physics and compared with
what the DeepMIMO loader/generator produces.  Nothing here selects a beam,
compares a policy, or touches a held-out site: this establishes units only.

Usage:  python fixtures.py <scenario> [tx_index]
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import deepmimo as dm

import g1

HERE = os.path.dirname(os.path.abspath(__file__))
dm.config.set('scenarios_folder', os.path.join(HERE, 'scenarios'))

SCEN = sys.argv[1] if len(sys.argv) > 1 else 'city_2_chicago_28'
TXI = int(sys.argv[2]) if len(sys.argv) > 2 else 0
RNG = np.random.default_rng(881)          # same input seed as Phase 1
N_SC = 512                                 # OFDM grid used only for fixture F2
R = {"scenario": SCEN, "tx_index": TXI, "checks": []}


def chk(name, passed, gate=True, **kw):
    """gate=True checks are prerequisites for site qualification; gate=False
    checks are measurements whose thresholds were declared in advance and whose
    outcome is reported either way."""
    R["checks"].append({"name": name, "pass": bool(passed), "gate": gate, **kw})
    print(f"{'PASS' if passed else 'FAIL'}  [{'gate' if gate else 'info'}]  {name}  " +
          "  ".join(f"{k}={v}" for k, v in kw.items()))


d = dm.load(SCEN)[TXI]
pw, ph = np.asarray(d.power), np.asarray(d.phase)
el, az = np.asarray(d.aod_el), np.asarray(d.aod_az)
dly, inter = np.asarray(d.delay), np.asarray(d.inter)
rx, tx = np.asarray(d.rx_pos), np.asarray(d.tx_pos)[0]
n_valid = np.isfinite(pw).sum(1)
act = np.flatnonzero(n_valid > 0)
lam = g1.C_LIGHT / float(d.rt_params['frequency'])

# -- F1  noise convention --------------------------------------------------
exact = g1.thermal_noise_w()
conv = 10 ** ((-174.0 + 10 * np.log10(g1.BANDWIDTH) + g1.NOISE_FIGURE_DB - 30) / 10)
chk("F1 sigma_d2 == -174 dBm/Hz + 10log10(B) + NF",
    abs(conv / g1.SIGMA_D2 - 1) < 1e-4,
    frozen_W=g1.SIGMA_D2, convention_W=float(f"{conv:.6e}"),
    exact_kT0BF_W=float(f"{exact:.6e}"),
    rounding_gap_dB=round(10 * np.log10(exact / g1.SIGMA_D2), 4))

# -- F2  hand-built narrowband channel == DeepMIMO generator ---------------
sub = RNG.choice(act, size=min(256, act.size), replace=False)
p = dm.ChannelParameters()
p.bs_antenna.shape = np.array(g1.ANT_SHAPE)
p.bs_antenna.spacing = g1.ANT_SPACING
p.ue_antenna.shape = np.array([1, 1])
p.freq_domain = 1
p.ofdm.subcarriers = N_SC
p.ofdm.selected_subcarriers = np.arange(1)
p.ofdm.bandwidth = g1.BANDWIDTH
p.num_paths = pw.shape[1]
H = d.compute_channels(p)                                     # (U,1,64,1)
h_dm = np.sqrt(N_SC) * H[sub, 0, :, 0].astype(np.complex128)
h_hand = g1.narrowband(pw[sub], ph[sub], el[sub], az[sub])
den = np.abs(h_dm).max()
err_raw = np.abs(h_dm - h_hand).max() / den
# DeepMIMO rotates the stored float32 angles through arccos(cos(theta)) before
# building the array response.  With zero rotation that is the identity in exact
# arithmetic but moves the angles by up to ~1e-5 deg in float32, which the 64
# element phases then accumulate.  Feeding the generator's own rotated angles
# into the same hand sum isolates the identity from that artifact.
elr = np.rad2deg(np.asarray(d['_aod_el_rot'])[sub])
azr = np.rad2deg(np.asarray(d['_aod_az_rot'])[sub])
h_rot = g1.narrowband(pw[sub], ph[sub], elr, azr)
err_rot = np.abs(h_dm - h_rot).max() / den
ang_dev = float(np.nanmax(np.abs(np.rad2deg(np.asarray(d['_aod_el_rot'])) - el)))
chk("F2 hand narrowband == sqrt(N)*DeepMIMO OFDM k=0", err_rot < 1e-6,
    n_users=int(sub.size), rel_err_matched_angles=float(f"{err_rot:.3e}"),
    rel_err_raw_angles=float(f"{err_raw:.3e}"),
    library_angle_roundtrip_max_dev_deg=float(f"{ang_dev:.2e}"),
    raw_angle_gap_dB=round(float(20 * np.log10(1 + err_raw)), 7))

# -- F2b  delay / subcarrier convention ------------------------------------
K2 = 256
p.ofdm.selected_subcarriers = np.array([0, K2])
H2 = d.compute_channels(p)
h_dm2 = np.sqrt(N_SC) * H2[sub, 0, :, 1].astype(np.complex128)
h_rot2 = g1.narrowband_at(pw[sub], ph[sub], elr, azr, dly[sub],
                          df_hz=K2 * g1.BANDWIDTH / N_SC)
e2 = np.abs(h_dm2 - h_rot2).max() / np.abs(h_dm2).max()
chk("F2b delay convention exp(-j2 pi tau df) at df = B/2", e2 < 1e-6,
    subcarrier=K2, df_MHz=K2 * g1.BANDWIDTH / N_SC / 1e6,
    rel_err=float(f"{e2:.3e}"))

# -- F3  codebook ----------------------------------------------------------
F = g1.codebook()
hh = h_hand[:8]
direct = np.abs(hh @ F) ** 2
chk("F3 codebook unit-norm, unitary, Parseval, FFT identity",
    abs(np.linalg.norm(F, axis=0) - 1).max() < 1e-12
    and np.abs(F.conj().T @ F - np.eye(64)).max() < 1e-12
    and np.abs(direct - g1.beam_power(hh)).max() / direct.max() < 1e-12
    and np.abs(direct.sum(1) / (np.abs(hh) ** 2).sum(1) - 1).max() < 1e-12,
    max_col_norm_err=float(f"{abs(np.linalg.norm(F,axis=0)-1).max():.1e}"),
    fft_identity_rel_err=float(f"{np.abs(direct-g1.beam_power(hh)).max()/direct.max():.1e}"))

# -- F4  free-space law fixes the power reference ---------------------------
los = act[(inter[act, 0] == 0) & (n_valid[act] >= 1)]
dist = np.linalg.norm(rx[los] - tx[None, :], axis=1)
fspl_db = 20 * np.log10(lam / (4 * np.pi * dist))
off = pw[los, 0] - fspl_db
chk("F4 LoS path power == Friis with 0 dBW isotropic transmit",
    abs(np.median(off)) < 0.05 and np.percentile(np.abs(off - np.median(off)), 99) < 0.5,
    n_los=int(los.size), median_offset_dB=round(float(np.median(off)), 4),
    p99_abs_dev_dB=round(float(np.percentile(np.abs(off - np.median(off)), 99)), 4),
    range_m=[round(float(dist.min()), 1), round(float(dist.max()), 1)])

# -- F5  single-path beam gain lies in the DFT scalloping band -------------
one = act[n_valid[act] == 1]
if one.size:
    h1 = g1.narrowband(pw[one], ph[one], el[one], az[one])
    best = g1.beam_power(h1).max(1)
    p_lin = 10 ** (pw[one, 0] / 10)
    ratio = best / (g1.N_ANT * p_lin)
    lo = (2 / np.pi) ** 2
    chk("F5 single-path array gain in [(2/pi)^2, 1] x N", 
        ratio.min() > lo - 1e-9 and ratio.max() < 1 + 1e-9,
        n_users=int(one.size), min_ratio=round(float(ratio.min()), 5),
        max_ratio=round(float(ratio.max()), 5), scallop_floor=round(lo, 5))

# -- F6  is the narrowband model adequate for BEAM SELECTION over 100 MHz? --
sub2 = RNG.choice(act, size=min(2000, act.size), replace=False)
ks = np.array([0, 64, 128, 192, 256, 320, 384, 448, 511])
p2 = dm.ChannelParameters()
p2.bs_antenna.shape = np.array(g1.ANT_SHAPE); p2.bs_antenna.spacing = g1.ANT_SPACING
p2.ue_antenna.shape = np.array([1, 1]); p2.freq_domain = 1
p2.ofdm.subcarriers = N_SC; p2.ofdm.selected_subcarriers = ks
p2.ofdm.bandwidth = g1.BANDWIDTH; p2.num_paths = pw.shape[1]
Hs = d.compute_channels(p2)
Hs = np.sqrt(N_SC) * Hs[sub2, 0, :, :].astype(np.complex128)         # (B,64,K)
bp = np.abs(np.fft.fft(Hs, axis=1)) ** 2 / g1.N_ANT                  # (B,M,K)
b0 = bp[:, :, 0].argmax(1)
bk = bp.argmax(1)
agree = (bk == b0[:, None]).mean()
tot = bp.sum(-1)
loss_db = 10 * np.log10(tot.max(1) / tot[np.arange(tot.shape[0]), b0])
# delay spread of the same users
w = np.where(np.isfinite(pw[sub2]), 10 ** (pw[sub2] / 10), 0.0)
t = np.where(np.isfinite(dly[sub2]), dly[sub2], 0.0)
mu = (w * t).sum(1) / w.sum(1)
rms = np.sqrt((w * (t - mu[:, None]) ** 2).sum(1) / w.sum(1))
chk("F6 narrowband beam choice holds across 100 MHz", agree > 0.9, gate=False,
    subcarrier_argmax_agreement=round(float(agree), 4),
    median_band_avg_loss_dB=round(float(np.median(loss_db)), 4),
    p95_band_avg_loss_dB=round(float(np.percentile(loss_db, 95)), 4),
    median_rms_delay_spread_ns=round(float(np.median(rms) * 1e9), 2),
    p95_rms_delay_spread_ns=round(float(np.percentile(rms, 95)), 2) if False
    else round(float(np.percentile(rms, 95) * 1e9), 2),
    coherence_bw_MHz_median=round(float(1 / (5 * np.median(rms)) / 1e6), 2))

# -- F7  resulting SNR scale ------------------------------------------------
h = g1.narrowband(pw[act], ph[act], el[act], az[act])
g = g1.snr(g1.beam_power(h).max(1))
gdb = 10 * np.log10(np.maximum(g, 1e-30))
chk("F7 best-beam SNR distribution is physically plausible",
    np.median(gdb) > 0 and np.percentile(gdb, 99.9) < 90,
    n_users=int(act.size),
    snr_dB_pct={q: round(float(np.percentile(gdb, q)), 2)
                for q in (1, 5, 25, 50, 75, 95, 99)},
    frac_below_0dB=round(float((gdb < 0).mean()), 4))

R["all_gate_pass"] = all(c["pass"] for c in R["checks"] if c["gate"])
R["all_pass"] = all(c["pass"] for c in R["checks"])
json.dump(R, open(os.path.join(HERE, f'fixtures_{SCEN}.json'), 'w'), indent=1)
print("\ngate checks: " + ("ALL PASS" if R["all_gate_pass"] else "FAILED")
      + " | informational: "
      + ("all pass" if R["all_pass"] else "at least one below its declared threshold"))
