"""Shared simulation machinery: one definition of the state, the measurement,
the probe rule and the elapsed-time accounting, used by the scheduler
estimation and by the evaluation so the two cannot drift apart."""
from __future__ import annotations
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'latency'))
import obs, feats
import models as M

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_MAX = 64                 # the scheduler may probe the whole codebook
N_EXIT = 3
NB = 40                    # bucket 0 = nothing measured; 1..39 = 3 dB bins from -48
B_LO, B_W = -48.0, 3.0
NC = 10                    # level class 0 = nothing measured; 1..9 = 10 dB bins from -40
C_LO, C_W = -40.0, 10.0
COMMIT_M, COMMIT_P, PROBE, COMPUTE = 0, 1, 2, 3
ACTION_NAMES = ['commit_measured', 'commit_predicted', 'probe', 'compute']


def bucket(best_snr_db, any_probed):
    b = 1 + torch.clamp(((best_snr_db - B_LO) / B_W).floor().long(), 0, NB - 2)
    return torch.where(any_probed, b, torch.zeros_like(b))


def level_class(mean_snr_db, any_probed):
    """A persistent receiver-level covariate: the mean measured SNR so far.

    Without it the controller's state is (exit, probes, best-so-far), and that
    pools receivers whose beam profiles differ systematically -- a strong
    receiver holding a mediocre beam and a weak receiver holding its best beam
    occupy the same cell and have completely different futures.  Iterating the
    pooled chain then over-predicts the value of probing at moderate counts and
    under-predicts it at large ones.  The running mean separates them, it is
    maintained by the per-probe bookkeeping every method already pays, and it
    leaves the consult a single table index.
    """
    c = 1 + torch.clamp(((mean_snr_db - C_LO) / C_W).floor().long(), 0, NC - 2)
    return torch.where(any_probed, c, torch.zeros_like(c))


def bucket_centre_snr():
    c = np.full(NB, B_LO - B_W / 2)
    c[1:] = B_LO + B_W * (np.arange(NB - 1) + 0.5)
    return c


class Site:
    def __init__(self, name):
        z = np.load(os.path.join(HERE, 'data', f'{name}.npz'))
        self.name = name
        self.pbar = torch.tensor(z['pbar'].astype(np.float64), device=DEV)
        self.se = torch.tensor(z['se'].astype(np.float32), device=DEV)
        self.pn = self.pbar[:, obs.LAYER6_OFFSET:]
        self.sn = self.se[:, obs.LAYER6_OFFSET:]
        self.rx = z['rx_pos']
        self.los = torch.tensor(z['los'], device=DEV)
        self.n = self.pn.shape[0]
        self.oracle_beam = self.sn.argmax(1)
        self.oracle_se = self.sn.gather(1, self.oracle_beam[:, None])[:, 0]
        self.strat_snr = (10 * torch.log10(obs.P_TX * self.pn.max(1).values
                                           / obs.SIGMA_D2)).cpu().numpy()


def measure(pbar, gen):
    """G2 band-averaged power measurement (watts), torch."""
    lam = 2.0 * obs.K_F * obs.P_TX * pbar / obs.SIGMA_P2
    z = torch.randn(pbar.shape, generator=gen, device=DEV, dtype=torch.float64)
    x = (z + torch.sqrt(lam)) ** 2
    r = torch.randn((*pbar.shape, 2 * obs.K_F - 1), generator=gen, device=DEV,
                    dtype=torch.float64)
    return torch.clamp(obs.SIGMA_P2 / (2 * obs.K_F) * (x + (r * r).sum(-1))
                       - obs.SIGMA_P2, min=1e-3 * obs.SIGMA_P2)


def load_predictor(seed):
    m = M.EarlyExitMLP().to(DEV)
    m.load_state_dict(torch.load(os.path.join(HERE, 'models', f'ee_{seed}.pt')))
    return m.eval()


def exit_logits(model, f, e):
    with torch.no_grad():
        x = f
        for i in range(e):
            x = model.blocks[i](x)
        return model.exits[e - 1](x)


def spread_score():
    """Priority of the frozen bit-reversal order as a score (higher = sooner)."""
    s = np.zeros(64, np.float32)
    s[obs.spread_order()] = np.arange(64, 0, -1)
    return torch.tensor(s, device=DEV)[None, :]


def elapsed_us(k, i, tau, t_probe, c_lookup=0.0, c_rank=0.0):
    """d(i,k) = k*(t_p + per-probe bookkeeping) + tau_i + (k+i+1) lookups.

    tau is the RESTART charge (latency.py: tau_restart_us), the sum of the fresh
    invocations 1..i, because the executor recomputes from block 0 on every
    compute action -- an intervening probe changes the network's input, so there
    is no activation to resume from.  Since COMPUTE advances i by exactly +1 and
    i never decreases, each exit is reached at most once on any trajectory, so
    the sum is a function of i alone and this stays deterministic in (i, k).

    Uniform accounting, charged identically to every method: a probe costs the
    20 us pilot+feedback slot plus the receiver-side work of recording it; the
    final selection among measured beams is then free, because the running best
    is already maintained (this replaces Phase 2's separate c_sel, which charged
    the same selection work once at the end instead of incrementally).
    Deterministic in (i, k), which is what makes exact backward induction
    legitimate under the G3 convention.

    After an inference call the controller picks each next codeword by a
    masked argmax over the exit's 64 scores, work that no other charge
    contains.  c_rank charges it on every probe of a decision sequence that
    has invoked the predictor (i >= 1), the probes before the first call
    included; that overcharges, but keeps d a function of (i, k).  c_rank = 0
    adds no arithmetic."""
    d = k * t_probe + tau[i] + (k + i + 1) * c_lookup
    if c_rank:
        d = d + c_rank * k * (i >= 1)
    return d


def elapsed_exact_us(k, i, tau, t_probe, c_lookup, c_rank, kappa):
    """The ranking time counted exactly.

    d(i, k, kappa) = k t_p + tau_i + (k + i + 1) c_lookup + 1[i >= 1] (k - kappa) c_rank,
    with kappa the probe count at the first inference call.  Before that call the controller
    probes in the fixed order, so it ranks nothing; after it, each probe's codeword is a masked
    argmax over the exit's scores, so the k - kappa probes taken since the call are the ranking
    operations it performs.  elapsed_us charges all k of them.  kappa may be a tensor, one entry
    per first-call probe count; at kappa = 0 the result equals elapsed_us with the same c_rank,
    operation for operation, and at i = 0 kappa does not enter."""
    d = k * t_probe + tau[i] + (k + i + 1) * c_lookup
    return d + c_rank * (k - kappa) * (i >= 1)


def lookup_charge_aug(lat):
    """The consult of the kappa-augmented policy table (65 blocks of the original shape, one
    per first-call probe count), timed on that table with retime.py's replay method
    (exact.py --time-lookup, lat_lookup_aug.json).  It is charged to the controller
    variants that use the augmented table; the others keep lookup_charge(lat)."""
    import json as _json
    assert lat['env']['profile'] == 'cpu-1t-pcore', lat['env']['profile']
    rec = _json.load(open(os.path.join(HERE, 'lat_lookup_aug.json')))
    assert rec['table_shape'] == [K_MAX + 1, N_EXIT + 1, K_MAX + 1, NB, NC, 2], rec['table_shape']
    return rec['charge_us']


def measure_chi2(pbar, gen):
    """The raw statistic behind measure(), before scaling, the
    noise-floor subtraction and the clip -- the same draws in the same order, so
    a generator seeded as evaluate.py seeds `gen` reproduces y exactly through
    report_of().  Zhang et al. detect at the receiver, which sees this energy."""
    lam = 2.0 * obs.K_F * obs.P_TX * pbar / obs.SIGMA_P2
    z = torch.randn(pbar.shape, generator=gen, device=DEV, dtype=torch.float64)
    x = (z + torch.sqrt(lam)) ** 2
    r = torch.randn((*pbar.shape, 2 * obs.K_F - 1), generator=gen, device=DEV,
                    dtype=torch.float64)
    return x + (r * r).sum(-1)


def report_of(X):
    """measure()'s reported value from the raw statistic: the same arithmetic."""
    return torch.clamp(obs.SIGMA_P2 / (2 * obs.K_F) * X - obs.SIGMA_P2,
                       min=1e-3 * obs.SIGMA_P2)


# ------------------------------------------------ compute-action variants ----
# The controller's compute action means "run the anytime predictor one exit
# deeper".  These specs let the same recursion schedule other predictors and
# other exit rules, so two questions -- is depth or invocation count doing the
# work, and does a stronger predictor change the conclusions -- can be
# answered by the controller itself.  One object is read by the planner's
# rollouts (sched.explore) and by the executor (methods.scheduled), so the two
# cannot disagree about what a compute runs.
KSET = (4, 8, 16, 32, 64)       # probe counts the per-K FC/CNN networks exist for


class Spec:
    """What a compute action runs, where it is legal, and what it costs.

    kind    'ee' the anytime predictor, or 'fc' / 'cnn' the per-K networks
    heads   exit head run at level e (ee only): heads[e]
    ks      probe counts at which compute is legal; None = any
    mix     (hit, oracle) from oracle_mix(), or None: the anytime predictor of
            a predictor-quality condition; ignored for 'fc' / 'cnn'
    """
    def __init__(self, kind='ee', heads=(1, 2, 3), ks=None, mix=None):
        self.kind, self.heads, self.ks = kind, tuple(heads), (None if ks is None else tuple(ks))
        self.mix = mix


# ------------------------------------------------------ predictor quality -----
# A predictor of controllable quality, to ask whether the controller's
# results are limited by its predictor: on a fixed subset of
# receivers -- each in it with probability q, drawn once per (site, seed) from
# a generator of its own -- the anytime predictor's logits are overwritten so
# that the rate-optimal beam ranks first, at every exit and every probe count.
# Elsewhere the trained predictor is untouched, and so is every charge, so q
# moves quality and nothing else.  The subsets are nested in q, and no other
# random stream moves, so q = 0 reproduces every number bit for bit.
def mix_key(*parts):
    """evaluate.stable_seed's recipe, without importing evaluate."""
    import zlib
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def oracle_mix(site, name, seed, q):
    """(hit, oracle) for the oracle-mixed predictor of quality q, or None."""
    if q is None:
        return None
    u = np.random.default_rng(mix_key('oq', name, seed)).random(site.n)
    return torch.tensor(u < q, device=DEV), site.oracle_beam


def apply_mix(lg, mix, sel=None):
    """lg, with the oracle beam put first for the hit receivers among `sel`.
    lg holds one row per receiver in `sel` (a boolean mask), or per receiver."""
    if mix is None:
        return lg
    hit, oracle = mix
    if sel is not None:
        hit, oracle = hit[sel], oracle[sel]
    if not bool(hit.any()):
        return lg
    lg = lg.clone()
    h = torch.nonzero(hit)[:, 0]
    lg[h, oracle[h]] = lg[h].max(1).values + 1
    return lg


def with_mix(spec, mix):
    """The compute spec of a mixed run: the anytime predictor carries the mix,
    the per-K FC and CNN networks do not."""
    if mix is None:
        return spec
    if spec is None:
        return Spec(mix=mix)
    if spec.kind == 'ee':
        return Spec(spec.kind, spec.heads, spec.ks, mix=mix)
    return spec


def load_nets(seed):
    """The per-K FC and CNN networks of one seed (models/{fc,cnn}_k{K}_{seed}.pt)."""
    nets = {}
    for k in KSET:
        fc = M.BeamMLP(n_in=k).to(DEV)
        fc.load_state_dict(torch.load(os.path.join(HERE, 'models', f'fc_k{k}_{seed}.pt')))
        cnn = M.BeamCNN().to(DEV)
        cnn.load_state_dict(torch.load(os.path.join(HERE, 'models', f'cnn_k{k}_{seed}.pt')))
        nets[f'fc{k}'], nets[f'cnn{k}'] = fc.eval(), cnn.eval()
    return nets


def compute_logits(spec, model, nets, yn, probed, f, sel, e, k):
    """Logits over the 64 narrow beams for the receivers `sel` at level e.

    yn is the 64-beam measurement, probed the (n, 64) mask, f the anytime
    predictor's feature matrix, k the probe counts.  FC and CNN take exactly
    the inputs methods.fixed_net gives them, which is valid because a variant
    using them never reorders: its probed set IS the frozen-order prefix of
    length k, the distribution those networks were trained on."""
    if spec.kind == 'ee':
        return apply_mix(exit_logits(model, f[sel], spec.heads[e]), spec.mix, sel)
    rows = torch.nonzero(sel)[:, 0]
    out = torch.zeros((rows.numel(), 64), device=DEV)
    s = torch.clamp(feats.snr_db_t(yn[rows]), feats.CLIP_LO, feats.CLIP_HI)
    kr = k[rows]
    for kk in spec.ks:
        m = kr == kk
        if not m.any():
            continue
        with torch.no_grad():
            if spec.kind == 'fc':
                idx = torch.tensor(obs.spread_order()[:kk].copy(), device=DEV)
                out[m] = nets[f'fc{kk}'](s[m][:, idx].float() / 20.0).float()
            else:
                msk = probed[rows][m]
                x = torch.stack([torch.where(msk, s[m], torch.full_like(s[m], feats.UNPROBED))
                                 / 20.0, msk.to(s.dtype)], 1).float()
                out[m] = nets[f'cnn{kk}'](x).float()
    return out


def variant_specs(lat, scale=1.0):
    """key -> (Spec, restart-style tau vector, reorder, max_exit) for the
    compute-action variants.  tau[i] is what reaching
    level i costs, charged exactly as sim.elapsed_us charges the headline
    controller, so d(i, k) stays deterministic in (i, k)."""
    f = [lat['tau_fresh_us'][str(j)] for j in (1, 2, 3)]
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    fcmax = max(ch[f'fc_k{k}'] for k in KSET)          # conservative: the dearest K
    v = lambda *t: np.array([0.0, *t]) * scale
    return {
        # repeated fresh invocations of exit 1: separates invocation count from depth
        'rep1': (Spec('ee', heads=(1, 1, 1)), v(f[0], 2 * f[0], 3 * f[0]), True, N_EXIT),
        # one fresh pass straight to exit 3, at most once
        'deep1': (Spec('ee', heads=(3,)), v(f[2], f[2], f[2]), True, 1),
        # the anytime predictor under the per-K networks' restrictions (for R3)
        'anyr': (Spec('ee', heads=(3,), ks=KSET), v(f[2], f[2], f[2]), False, 1),
        'fcc': (Spec('fc', ks=KSET), v(fcmax, fcmax, fcmax), False, 1),
        'cnnc': (Spec('cnn', ks=KSET), v(ch['cnn'], ch['cnn'], ch['cnn']), False, 1),
    }


def lookup_charge(lat):
    """The per-decision table-consult charge the planner and the executor use.

    On the primary profile the charge is the consult timed on the real table (retime.py,
    lat_lookup_real.json).  latency.py's own 'lookup' row timed it on a stand-in table;
    LOOKUP_CHARGE=standin reproduces that charge, for the isolated diff that measures the
    difference.  Both sched.py and evaluate.py call this, so the two cannot disagree."""
    import json as _json
    ch = {r['component']: r['charge_us'] for r in lat['rows']}
    p = os.path.join(HERE, 'lat_lookup_real.json')
    on = os.environ.get('LOOKUP_CHARGE', 'real') != 'standin'      # isolated diffs only
    if on and lat['env']['profile'] == 'cpu-1t-pcore':
        return _json.load(open(p))['charge_us']
    return ch['lookup']
