"""G2: the wideband observation law for Phase 4.  Frozen before any policy.

G1 (Phase 1-3) is kept verbatim for physics, geometry and noise constants; this
module only extends it from one frequency to the 100 MHz band, because Phase 3's
fixture F6 FAILED -- measured coherence bandwidth is 6-23 MHz, so a beam chosen
on the band-centre channel is not the beam a 100 MHz pilot would choose, and the
error reaches 2.97 dB at the 95th percentile.  Nothing here is tuned; every
number is either inherited from G1 or fixed by the band.

  B         = 100 MHz               (G1)
  K_f       = 16 equal sub-bands, evaluated at their centres
  h_j       = channel at sub-band centre j            (g1.narrowband_at)
  p[u,c,j]  = |h_j(u)^T f_c|^2       for codeword c   (analog beam: flat in f)

  MEASUREMENT.  Probing codeword c costs t_p = 20 us (10 pilot + 10 feedback,
  G1) and returns the band-averaged received power under pilot noise
  sigma_p^2 = sigma_d^2 / 1000 (1000-symbol despreading, G1):

      y = (1/K_f) sum_j |sqrt(P_tx) h_j^T f_c + w_j|^2 ,  w_j ~ CN(0, sigma_p^2)

  which is exactly (sigma_p^2 / 2K_f) * chi'^2(df = 2 K_f, lambda = 2 K_f S /
  sigma_p^2) with S = P_tx * mean_j p[u,c,j].  The noncentral chi-square depends
  on the TOTAL noncentrality only, so the band-average measurement is a function
  of the band-average power alone -- the per-sub-band powers need not be stored.
  The receiver knows its own noise floor and reports the debiased estimate
  max(y - sigma_p^2, 1e-3 sigma_p^2).

  PAYOFF.  An analog beam is frequency-flat, so committing to codeword c at
  elapsed time d earns the frequency-selective average

      R = max(0, 1 - d/T) * (1/K_f) sum_j log2(1 + P_tx p[u,c,j] / sigma_d^2)

  which reduces to G1's max(0,1-d/T) log2(1+gamma) on a flat channel.

  CODEBOOK.  A 6-layer binary sub-array hierarchy.  Layer l has 2^l codewords
  formed on the first 2^l elements, so its beamwidth is 2*pi/2^l in psi and its
  array gain is 2^l -- the "ideal sector gain" Zhang et al. (2020) assume, here
  realised by an actual sub-array rather than assumed.  Layer 6 is exactly the
  64-beam normalised DFT codebook of G1, so every Phase 1-3 number still refers
  to the same beams.  Children of (l,b) are (l+1,2b) and (l+1,2b+1).
  Flat index: idx(l,b) = 2^l - 2 + b, total 126 codewords.
"""
from __future__ import annotations

import numpy as np

import g1

K_F = 16                              # sub-bands across the 100 MHz band
N_LAYER = 6                           # hierarchy depth; layer 6 == G1 codebook
N_CODE = 2 ** (N_LAYER + 1) - 2       # 126
M_BEAMS = g1.M_BEAMS                  # 64 narrow beams
T_P_US = g1.T_P_US                    # 20 us per probe, any layer
C_SEL_US = 1.88                       # selection/emit charge (Phase 2, measured)
SIGMA_D2, SIGMA_P2 = g1.SIGMA_D2, g1.SIGMA_P2
P_TX = g1.P_TX

# SIGMA_P2 is a PER-SUB-BAND quantity and is derived in g1 from the pilot's
# total symbol budget divided by the number of sub-bands.  If K_F here and
# N_SUBBANDS there ever disagree, the measurement below silently claims more
# processing gain than the pilot has -- which is exactly the 16x error this
# assertion exists to prevent.
assert K_F == g1.N_SUBBANDS, (K_F, g1.N_SUBBANDS)
assert abs(K_F * SIGMA_D2 / SIGMA_P2 - g1.PILOT_SYMBOLS) < 1e-6, (
    "band-averaged processing gain must equal the pilot's symbol budget")


def subband_offsets(k_f: int = K_F, bandwidth: float = g1.BANDWIDTH):
    """Centres of k_f equal sub-bands spanning [-B/2, +B/2]."""
    edges = np.linspace(-bandwidth / 2, bandwidth / 2, k_f + 1)
    return 0.5 * (edges[1:] + edges[:-1])


def layer_of(idx):
    """Layer index 1..6 of a flat codeword index."""
    idx = np.asarray(idx)
    return (np.log2(idx + 2)).astype(int)


def flat_index(layer: int, beam) -> np.ndarray:
    return 2 ** layer - 2 + np.asarray(beam)


LAYER6_OFFSET = 2 ** N_LAYER - 2       # 62; layer-6 beam b is flat index 62 + b


def hierarchical_codebook(n_ant: int = g1.N_ANT) -> np.ndarray:
    """(N_ANT, 126) complex codewords, unit norm, columns in flat-index order."""
    F = np.zeros((n_ant, N_CODE), dtype=np.complex128)
    for l in range(1, N_LAYER + 1):
        n_act = 2 ** l
        n = np.arange(n_act)[:, None]
        b = np.arange(n_act)[None, :]
        F[:n_act, flat_index(l, np.arange(n_act))] = \
            np.exp(-2j * np.pi * n * b / n_act) / np.sqrt(n_act)
    return F


def spread_order(m: int = M_BEAMS) -> np.ndarray:
    """Frozen probe order for the narrow codebook: bit-reversal of 0..M-1.

    Maximally spread, deterministic, declared before any result.  Used by every
    method that probes without a predictor's opinion, and for the first
    K_0 probes of every method that has one.
    """
    bits = int(np.log2(m))
    idx = np.arange(m)
    rev = np.zeros(m, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def measure(pbar_true, rng, k_f: int = K_F):
    """Noisy band-averaged power measurement, watts. Shape-preserving.

    pbar_true: band-averaged |h^T f|^2 (linear, W per unit transmit power).
    """
    s = P_TX * np.asarray(pbar_true, dtype=np.float64)
    lam = 2.0 * k_f * s / SIGMA_P2
    z = rng.standard_normal(s.shape) + np.sqrt(lam)
    x = z * z + rng.chisquare(2 * k_f - 1, size=s.shape)
    y = SIGMA_P2 / (2.0 * k_f) * x
    return np.maximum(y - SIGMA_P2, 1e-3 * SIGMA_P2)


def snr_db(power_w):
    return 10.0 * np.log10(np.maximum(P_TX * np.asarray(power_w), 1e-30) / SIGMA_D2)


def net_rate(se_bps_hz, elapsed_us, t_coh_us):
    """R = max(0, 1 - d/T) * SE, the G1 payoff with the G2 wideband SE."""
    return np.maximum(0.0, 1.0 - np.asarray(elapsed_us) / t_coh_us) * se_bps_hz
