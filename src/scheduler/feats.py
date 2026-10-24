"""The 194-dimensional G3 feature vector (protocol amendment A1), in torch.

One definition, used by training, by the scheduler and by the evaluation, so a
mismatch between them is impossible.  Layout:
    [  0: 64)  masked log-SNR / 20, clipped to [-40, +60] dB; unprobed
               beams carry the -40 dB sentinel, never 0 dB (protocol A3)
    [ 64:128)  probed mask
    [128:192)  one-hot of the best measured beam so far
    [192]      probes taken / 64
    [193]      best measured SNR (dB) / 20
"""
import torch

import obs

CLIP_LO, CLIP_HI = -40.0, 60.0
UNPROBED = CLIP_LO          # sentinel (A3): must sit OUTSIDE the informative range
_LOG10 = 2.302585092994046
NEG = -1e30


def snr_db_t(y_w):
    return 10.0 / _LOG10 * torch.log(torch.clamp(obs.P_TX * y_w, min=1e-30)
                                     / obs.SIGMA_D2)


def build(y_w, mask):
    """y_w (B,64) measured power in W; mask (B,64) bool.  -> (B,194) float32."""
    s = torch.clamp(snr_db_t(y_w), CLIP_LO, CLIP_HI)
    m = mask.to(y_w.dtype)
    masked = torch.where(mask, s, torch.full_like(s, UNPROBED))
    best = torch.argmax(torch.where(mask, s, torch.full_like(s, NEG)), dim=1)
    hot = torch.zeros_like(m)
    hot.scatter_(1, best[:, None], 1.0)
    k = m.sum(1, keepdim=True)
    bs = torch.gather(s, 1, best[:, None])
    bs = torch.where(k > 0, bs, torch.full_like(bs, CLIP_LO))
    return torch.cat([masked / 20.0, m, hot, k / 64.0, bs / 20.0], dim=1).float(), best
