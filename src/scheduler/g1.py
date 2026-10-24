"""G1 observation law and narrowband channel construction, frozen for Phase 3.

Physical constants are reused verbatim from the Phase-1 calibration (calibrate.py
lines 41-42) and from the pilot design: 30 dBm transmit power, 100 MHz noise
bandwidth, 7 dB noise figure, 10 us pilot + 10 us feedback per probe.

The array convention is read from the DeepMIMO source, not assumed:
  _ant_indices((64,1)) places elements at y = 0..63, x = z = 0, so the panel is a
  ULA along +y with spacing d = 0.5*lambda, and _array_response uses
      a_n = exp(j * 2*pi*d/lambda * n * sin(theta) * sin(phi))
  with theta the POLAR angle carried in aod_el (0 deg = +z) and phi = aod_az.
  With zero array rotation _rotate_angles_batch is the identity on (theta, phi).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- physics ---
F_C = 28e9                     # Hz, scenario carrier (checked against rt_params)
C_LIGHT = 299_792_458.0        # m/s
P_TX = 1.0                     # W, 30 dBm  (pilot-design.md, explicit assumption)
BANDWIDTH = 100e6              # Hz         (pilot-design.md)
NOISE_FIGURE_DB = 7.0          # dB         (pilot-design.md)
SIGMA_D2 = 1.9953e-12          # W, G1 data-noise power   (calibrate.py:41)

# --- pilot noise, per sub-band -------------------------------------------
# The 10 us pilot spans the whole 100 MHz, so it carries
#     L = B * t_pilot = 1e8 * 10e-6 = 1000
# complex dimensions IN TOTAL.  That is the entire budget; G1's rule is that
# the despreading gain "is included exactly once ... No extra cost-free
# repeat/averaging is allowed."
#
# G2 measures the band by averaging N_SUBBANDS = 16 sub-bands (obs.K_F), so each
# sub-band is 100 MHz / 16 = 6.25 MHz wide and receives only L / 16 = 62.5 of
# those symbols.  Its post-despreading noise is therefore
#     sigma_p^2 = sigma_d^2 / (L / N_SUBBANDS) = sigma_d^2 / 62.5,
# and averaging the 16 sub-bands recovers the full 1000-fold gain -- once.
#
# Charging sigma_d^2 / 1000 per sub-band AND averaging 16 of them spends every
# pilot symbol sixteen times: it implies 16,000 independent dimensions where
# 1000 exist, and makes every probe 12.04 dB cleaner than the pilot can be.
PILOT_SYMBOLS = 1000           # L = BANDWIDTH * 10 us, the whole pilot budget
N_SUBBANDS = 16                # must equal obs.K_F; asserted there
SIGMA_P2 = SIGMA_D2 / (PILOT_SYMBOLS / N_SUBBANDS)   # = SIGMA_D2 / 62.5
T_P_US = 20.0                  # us per probe: 10 pilot + 10 feedback
M_BEAMS = 64
N_ANT = 64

ANT_SHAPE = (N_ANT, 1)         # DeepMIMO bs_antenna shape -> ULA along +y
ANT_SPACING = 0.5              # wavelengths


def thermal_noise_w(bandwidth: float = BANDWIDTH,
                    nf_db: float = NOISE_FIGURE_DB) -> float:
    """kTB*F at 290 K, in watts.  Hand fixture for SIGMA_D2."""
    k_b, t_0 = 1.380649e-23, 290.0
    return k_b * t_0 * bandwidth * 10 ** (nf_db / 10)


def codebook(m: int = M_BEAMS, n: int = N_ANT) -> np.ndarray:
    """Normalized DFT codebook, columns f_m with ||f_m|| = 1.

    f_m[n] = exp(-j 2 pi n m / M) / sqrt(N), so h^T F = fft(h) / sqrt(N).
    """
    nn = np.arange(n)[:, None]
    mm = np.arange(m)[None, :]
    return np.exp(-2j * np.pi * nn * mm / m) / np.sqrt(n)


def path_gains(power_dbw: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """Complex path gain sqrt(P) * exp(j*phase), NaN-padded paths -> 0."""
    ok = np.isfinite(power_dbw) & np.isfinite(phase_deg)
    p_lin = np.where(ok, 10.0 ** (np.where(ok, power_dbw, 0.0) / 10.0), 0.0)
    ph = np.deg2rad(np.where(ok, phase_deg, 0.0))
    return np.sqrt(p_lin) * np.exp(1j * ph)


def spatial_freq(aod_el_deg: np.ndarray, aod_az_deg: np.ndarray) -> np.ndarray:
    """psi = 2*pi*spacing * sin(theta_polar) * sin(phi), radians per element."""
    th = np.deg2rad(aod_el_deg)
    ph = np.deg2rad(aod_az_deg)
    return 2 * np.pi * ANT_SPACING * np.sin(th) * np.sin(ph)


def narrowband(power_dbw, phase_deg, aod_el_deg, aod_az_deg,
               chunk: int = 4096) -> np.ndarray:
    """Flat-fading channel h[u, n] = sum_p sqrt(P_p) e^{j phi_p} e^{j n psi_p}.

    Equals sqrt(N_sc) * H[u, 0, :, k=0] from DeepMIMO's OFDM generator; fixture
    F2 checks that identity numerically.
    """
    n_u = power_dbw.shape[0]
    out = np.empty((n_u, N_ANT), dtype=np.complex128)
    ant = np.arange(N_ANT)[None, :, None]                      # (1, N, 1)
    for s in range(0, n_u, chunk):
        e = min(s + chunk, n_u)
        g = path_gains(power_dbw[s:e], phase_deg[s:e])[:, None, :]   # (B,1,P)
        psi = spatial_freq(aod_el_deg[s:e], aod_az_deg[s:e])
        psi = np.where(np.isfinite(psi), psi, 0.0)[:, None, :]       # (B,1,P)
        out[s:e] = (g * np.exp(1j * ant * psi)).sum(-1)
    return out


def beam_power(h: np.ndarray) -> np.ndarray:
    """|h^T f_m|^2 for the normalized DFT codebook, via the FFT identity."""
    return np.abs(np.fft.fft(h, axis=-1)) ** 2 / N_ANT


def snr(beam_pow: np.ndarray) -> np.ndarray:
    """G1 post-beamforming data SNR gamma = P_tx |h^T f|^2 / sigma_d^2."""
    return P_TX * beam_pow / SIGMA_D2


def narrowband_at(power_dbw, phase_deg, aod_el_deg, aod_az_deg, delay_s,
                  df_hz: float = 0.0, chunk: int = 4096) -> np.ndarray:
    """Channel at a frequency offset df from the ray-tracing reference.

    h(df)[u, n] = sum_p sqrt(P_p) e^{j phi_p} e^{-j 2 pi tau_p df} e^{j n psi_p}.
    df = 0 reproduces narrowband().  The delay convention matches DeepMIMO's
    OFDM factor exp(-j 2 pi tau_n k / N) with tau_n = tau * B and df = k * B / N.
    """
    n_u = power_dbw.shape[0]
    out = np.empty((n_u, N_ANT), dtype=np.complex128)
    ant = np.arange(N_ANT)[None, :, None]
    for s in range(0, n_u, chunk):
        e = min(s + chunk, n_u)
        g = path_gains(power_dbw[s:e], phase_deg[s:e])
        tau = np.where(np.isfinite(delay_s[s:e]), delay_s[s:e], 0.0)
        g = (g * np.exp(-2j * np.pi * tau * df_hz))[:, None, :]
        psi = spatial_freq(aod_el_deg[s:e], aod_az_deg[s:e])
        psi = np.where(np.isfinite(psi), psi, 0.0)[:, None, :]
        out[s:e] = (g * np.exp(1j * ant * psi)).sum(-1)
    return out
