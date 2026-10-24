"""Beam-prediction inference components for the Phase 1 latency atlas.

Every component is a *per-decision* operation: the work a receiver or base
station must complete once, inside one coherence interval, to emit a beam
decision.  Training cost is out of scope by construction.

Provenance tags
---------------
exact        layer sizes and activations are stated in the cited text
reconstruct  the class and properties are stated; exact widths are read from
             a figure we could not parse, so the widths here are a documented
             reconstruction and are labelled as such in every output
spec         defined by this project's own frozen candidate specification
"""

import math
import numpy as np
import torch
import torch.nn as nn

N_BEAMS = 64          # 64-beam DFT codebook / 64-element ULA
N_ANT = 64


# ---------------------------------------------------------------- init ----

def glorot_(module: nn.Module, seed: int) -> nn.Module:
    """Glorot-uniform weights, zero biases, under an explicit seed.

    G3 calibrates with untrained weights: latency must not depend on what a
    network learned, only on its shape.
    """
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight, generator=g)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_uniform_(p, generator=g)
                else:
                    nn.init.zeros_(p)
    return module


# ------------------------------------------------- Graff et al. networks ----
# arXiv:2201.04657 sec. IV-A.  RSU array N = 64 elements.

class GraffEig(nn.Module):
    """Eigenvector prediction network, Fig. 6.  [exact]

    "The input to the network is a 64 element vector containing the complex
    values of the input eigenvector ... converted from complex values to
    magnitude and phase components ... stacked together in a single vector of
    128 elements ... 5 fully-connected layers containing 128, 256, 512, 256,
    and 128 activation units ... LeakyReLU ... alpha = 0.1 ... dropout layer
    after the activations of the 3rd fully-connected layer ... rate of 50% ...
    output of the last layer is then scaled to unit norm."
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.ModuleList([
            nn.Linear(128, 128), nn.Linear(128, 256), nn.Linear(256, 512),
            nn.Linear(512, 256), nn.Linear(256, 128),
        ])
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (1, 64, 2) real/imag -> magnitude and phase, stacked to 128.
        # This front-end is real per-decision work and is charged.
        mag = torch.sqrt(z[..., 0] ** 2 + z[..., 1] ** 2)
        pha = torch.atan2(z[..., 1], z[..., 0])
        x = torch.cat([mag, pha], dim=-1)
        for i, layer in enumerate(self.fc):
            x = self.act(layer(x))
            if i == 2:
                x = self.drop(x)
        return x / x.norm(dim=-1, keepdim=True)


class GraffCol(nn.Module):
    """Covariance-vector prediction network.  [exact]

    "the network architecture used for eigenvector prediction can be reused
    with slight modifications. The middle layer containing 512 neurons is
    neglected, and dropout is not used ... we use Tanh."
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.ModuleList([
            nn.Linear(128, 128), nn.Linear(128, 256),
            nn.Linear(256, 256), nn.Linear(256, 128),
        ])
        self.act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z[..., 0], z[..., 1]], dim=-1)   # two-channel input
        for layer in self.fc:
            x = self.act(layer(x))
        return x


class GraffAPS(nn.Module):
    """APS prediction network, Fig. 5.  [reconstruct]

    The text states only the class: "1D convolutional layers ... are adopted
    in the proposed NN. LeakyReLU is used as the activation function."  The
    widths live in a figure image that did not survive text extraction, so
    the three-layer 32/64/1 stack below is our reconstruction and is reported
    as such.  It is deliberately at the small end of plausible: a wider
    reconstruction would only raise the measured cost.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, 5, padding=2), nn.LeakyReLU(0.1),
            nn.Conv1d(64, 1, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------- survey-class beam predictors ----

class BeamMLP(nn.Module):
    """Fully-connected beam classifier.  [reconstruct]

    The canonical DL beam predictor: received power over a few probed beams
    in, a distribution over the 64-beam codebook out.  arXiv:2111.11177 sec.
    III describes this class ("a fully-connected (FC) DNN for the optimal
    beam-pair prediction"); widths are the common 256/256 configuration.
    """

    def __init__(self, n_in: int = 8, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_BEAMS),
        )

    def forward(self, x):
        return self.net(x)


class BeamCNN(nn.Module):
    """CNN beam predictor over a low-frequency CSI / wide-beam profile.
    [reconstruct]

    arXiv:2111.11177 Fig. 4(b),(c): "firstly convolutional layers are utilized
    to extract some [features] ... introduced after the convolutional layers,
    which downsamples".  Two conv stages plus an FC head.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 16, 256), nn.ReLU(), nn.Linear(256, N_BEAMS),
        )

    def forward(self, x):
        h = self.conv(x)
        return self.head(h.flatten(1))


class BeamLSTM(nn.Module):
    """Cascaded-LSTM beam tracker.  [reconstruct]

    arXiv:2111.11177 Fig. 4(c),(d): LSTM beam tracking from a history of
    previous CSI estimates.  Two layers, hidden 128, history length 8.
    Exposes both a full-sequence pass and a single cached-state step, because
    only the second is the honest per-decision charge for a tracker already
    running.
    """

    def __init__(self, n_in: int = 64, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, N_BEAMS)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return self.head(h[:, -1]), state


class SmallTransformer(nn.Module):
    """Small transformer encoder over a 64-token beam/antenna sequence.
    [reconstruct]

    Stands for the transformer-based beam predictors that motivate the
    multimodal-latency figure, at a size a base station might plausibly
    accept: 4 layers, d_model 128, 8 heads.
    """

    def __init__(self, d_model: int = 128, layers: int = 4, heads: int = 8):
        super().__init__()
        self.embed = nn.Linear(2, d_model)
        self.pos = nn.Parameter(torch.zeros(1, N_ANT, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, N_BEAMS)

    def forward(self, x):
        h = self.enc(self.embed(x) + self.pos)
        return self.head(h.mean(dim=1))


# ------------------------------------------ this project's own components ----

class StageValueNet(nn.Module):
    """One scalar stage value network.  [spec]

    G2: "two scalar-output networks S_hat_k and C_hat_k, each with two
    128-unit ReLU hidden layers and a linear output", over G1's 194 inputs.
    """

    def __init__(self, n_in: int = 194, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class RankerNet(nn.Module):
    """The fixed ranker.  [spec]

    G1/G2: eight real inputs (real and imaginary parts of z_j/sigma_d for the
    four initial beams), 64 log-power targets, two 128-unit hidden layers.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, N_BEAMS),
        )

    def forward(self, x):
        return self.net(x)


class EarlyExitMLP(nn.Module):
    """Anytime beam predictor with three exits.  [spec / proposed]

    The mechanism the compute-aware controller schedules: each block may be
    followed by a decision, so the per-decision cost is cumulative in the
    number of blocks executed.
    """

    def __init__(self, n_in: int = 194, hidden: int = 256, n_exits: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()
        d = n_in
        for _ in range(n_exits):
            self.blocks.append(nn.Sequential(nn.Linear(d, hidden), nn.ReLU()))
            self.exits.append(nn.Linear(hidden, N_BEAMS))
            d = hidden

    def forward(self, x, n_blocks: int):
        for i in range(n_blocks):
            x = self.blocks[i](x)
        return self.exits[n_blocks - 1](x)
