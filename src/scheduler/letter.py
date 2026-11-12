#!/usr/bin/env python3
"""The numbers of the letter "Inference-Latency Budgets for Learned Beam Prediction Under 5G NR
Timing" (protocol.json, dated change of 2026-11-10: no rule, stored output or verdict changes).
Nothing is simulated or timed: it reads stored outputs and the NR timing modules and recomputes,
for a 4-probe predictor against each reference, the inference budget tau*, the slack, the sweep's
efficiencies and the efficiency each implementation's exclusion needs, as the long version's
generator computes them, so that every number the letter prints has released code.

References (keys): csirs12, csirs30, csirs8 (paired 120 kHz CSI-RS sweep, 12 or 8 resources per
slot, 120 or 30 kHz control carrier; nrtime.py), tdd_DDDSU, tdd_7DS8U (nrtdd.py, rule R30), csirs60,
mu60_DDDSU, mu60_DDDSUUUU (60 kHz, nrmu.py and nrtddmu.py, rules R35-R36), gd3, gd4 (the SS-guided
refinement with 8 or 16 SS blocks, ssbref.py, rule R31), probe, rx4, ssb, slot, hier6 (the per-probe
model at t_p, with four receive beams, at the SS burst's and a slot's spacing, and Hier-6).
Implementations: ResNet-18 on eight P-core threads and ViT-B/16 and ResNet-18 on the GPU (FP32; each
latency's lower value over the runtimes measured, reconcile.json, rule R21), ViT-B/16's FP16 graph
(matched.json, rule R22) and rule R38's frozen builds (deploy.json).

Readings, per reference: tau* over every declarable timing and trigger slot (range and the timing of
each end), the slack (1 - eta)(T_max - d) at the measured efficiency, density 3's bound and the
smallest bound (largest over timings), and the efficiencies; per implementation and reference: the
lowest efficiency at which (4) excludes it at adverse inputs, the largest over timings and trigger
slots (and the timing that sets it), and its status at the smallest bound: excluded, unresolved
(only slower runtimes excluded, rule R21) or not excluded; per FP32 exclusion: the horizon
T_max + margin / (1 - eta) at the smallest and the measured bounds, the least over the NR references
that exclude it (rule R47); the trained networks' latency range; the reduced-precision speed-ups.

Gates (none recomputes a value with the formula that produced it):
  G1  the timing modules are the files rules R30, R35 and R36 declared (their sha256 in protocol.json)
      and their own gates hold (nrtdd gate F, nrmu gate M, nrtddmu gate P)
  G2  every timing row's budget is s_S - s_L + W; W is two and nine slots under DDDSU and 7DS8U; the
      paired range's top is the declared timing's budget
  G3  the CSI-RS efficiencies and smallest bounds equal margins_post.json's (rules R16, R19); the
      per-probe margins at the bound equal reconcile.json's margin_us
  G4  every reference's bounds satisfy smallest <= density 3 <= measured < 1
  G5  the unresolved entries are rule R41's: ResNet-18 on eight threads against 7DS8U, four receive
      beams and the SS burst, ViT-B/16 against 60 kHz DDDSUUUU and a slot per probe; and ResNet-18
      on the GPU per probe, as the long version states it
  G6  the trained networks' largest latency is below every reference's smallest budget
  G7  under either time-division pattern the refinement reports with the candidate's resources
  G8  at the fastest stored call every FP32 exclusion at the smallest bounds holds but the two the
      long version names (ResNet-18 against 60 kHz DDDSU, ViT-B/16 against four receive beams)

usage:  python3 letter.py [--gate | --dry]      -> letter.json (refuses to overwrite it)
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'letter.json')

import nrtime as NRT                      # noqa: E402
import nrtdd as TDD                       # noqa: E402
import nrmu as MU                         # noqa: E402
import nrtddmu as MUT                     # noqa: E402


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def J(name):
    return json.load(open(os.path.join(HERE, name)))


PROT = J('protocol.json')
ANA = {a['key']: a for a in PROT['analyses']}
RULES = J('rules_rx.json')
K = RULES['R5']['K']
T = float(max(J('report_rx.json')['grid_us']))           # T_max
TP = J('rho.json')['t_p_us']                             # per-probe time, one P-core
SLOT, SS_US = 125.0, 5000.0
SS_BEAM = SS_US / 64
MG = J('margins.json')                                   # (4)'s adverse per-probe inputs
T_AIR, DS_HI = MG['t_air_us'], MG['dS_hi_us']
RC = J('reconcile.json')
R5ROW = {(r['component'], r['profile']): r for r in RC['r5_rows']}
MGR = {(r['component'], r['profile']): r for r in RC['margin_rows']}
MP = J('margins_post.json')

# ---- the sweep's efficiencies (the long version's _cs_meas, CS_LB_MIN, CS_ETA8, CS_LB8_MIN, GD,
# ETA_SWEEP_MIN, ETA_LB_PROBE, RX_E, RS_ETA): lowest over scenarios, laws and variants ----
_SITES = [s for f in ('results_rx.json', 'validation_rx.json', 'heldout_rx.json')
          for s in J(f)['sites']]
ETA_SWEEP = min(s['methods']['sweep64']['eta'] for s in _SITES)
CS, CG, LV = J('csirs.json'), J('csirs_cfg.json'), J('levels.json')
CS_N = CS['density_samples']
_LAWS = [f"{e}|{n}" for e in ('coherent', 'per_re') for n in CS_N.values()]
_SRC = {'exact': CS['configs'], **{v: CG['variants'][v]['configs']
                                   for v in ('spread|worst', 'contiguous|worst')}}
_r16 = lambda key, f: min(src[key][f] for src in _SRC.values())
CS_MEAS = min(_r16(k, 'eta_min') for k in _LAWS)
CS_LB3 = _r16(f"coherent|{CS_N['3']}", 'eta_lb_min')
CS_SMALL = min(_r16(k, 'eta_lb_min') for k in _LAWS)
_LV8 = ('exact', 'spread8|worst', 'contiguous8|worst')
_lb8 = lambda key: min(LV['summary'][s][key]['eta_lb_min'] for s in _LV8[1:])
CS8_MEAS = min(LV['summary'][s][k]['eta_min'] for s in _LV8 for k in _LAWS)
CS8_LB3 = min(_lb8(f"coherent|{CS_N['3']}"), CS_LB3)
CS8_SMALL = min(min(_lb8(k) for k in _LAWS), CS_SMALL)
SB = J('ssbref.json')
GD = {}
for _l in SB['layers']:
    _R = SB['references'][f'guided{_l}']
    GD[_l] = dict(meas=min(_R[k]['eta_min'] for k in _LAWS),
                  bound=_R[f"coherent|{CS_N['3']}"]['eta_lb_min'],
                  small=min(_R[k]['eta_lb_min'] for k in _LAWS))
ETA_LB_PROBE = CS['configs']['coherent|1000']['eta_lb_min']
UEP = J('uepairs.json')
RX4 = dict(meas=UEP['summary']['4|all']['eta_min'], bound=UEP['summary']['4|all']['eta_lb_min'])
RX4['small'] = RX4['bound']
HIER = dict(meas=min(s['methods']['hier6']['eta'] for s in _SITES),
            bound=J('refbounds.json')['references']['hier6']['eta_lb_min'])
HIER['small'] = HIER['bound']

_cs = dict(meas=CS_MEAS, bound=CS_LB3, small=CS_SMALL)
_pp = dict(meas=ETA_SWEEP, bound=ETA_LB_PROBE, small=ETA_LB_PROBE)
BOUNDS = {'csirs12': _cs, 'csirs30': _cs, 'csirs8': dict(meas=CS8_MEAS, bound=CS8_LB3, small=CS8_SMALL),
          'csirs60': _cs, 'tdd_DDDSU': _cs, 'tdd_7DS8U': _cs, 'mu60_DDDSU': _cs, 'mu60_DDDSUUUU': _cs,
          'gd3': GD[3], 'gd4': GD[4], 'probe': _pp, 'rx4': RX4, 'ssb': _pp, 'slot': _pp,
          'hier6': HIER}
REFS = list(BOUNDS)
LONG_REFS = list(REFS)            # the long version's fifteen (its Table III)
NR_REFS = [k for k in REFS if k.startswith(('csirs', 'tdd_', 'mu60_', 'gd'))]

# ---- timings: one row per declarable timing (and trigger slot), with the reference's completion
# d (s_S; the refinement reports with the candidate, so its d is s_L), s_L, W and tau* ----
def _row(d, s_L, W, tag):
    return dict(d=d, s_L=s_L, W=W, budget=d - s_L + W, tag=tag)


_gt = {c: TDD.times('paired', 0, 12, 120, *c, K) for c in NRT.capabilities(120)}
_nrt = lambda n, f: [(c, NRT.times(n, f, *c, K)) for c in NRT.capabilities(f)]
TIMINGS = {
    'csirs12': [_row(r['s_S'], r['s_L'], r['W'], dict(x=c[0], kb=c[1])) for c, r in _gt.items()],
    'csirs30': [_row(t['d_S'], t['reduced'], SLOT, dict(x=c[0], kb=c[1])) for c, t in _nrt(12, 30)],
    'csirs8': [_row(t['d_S'], t['reduced'], SLOT, dict(x=c[0], kb=c[1])) for c, t in _nrt(8, 120)],
    'csirs60': [_row(r['s_S'], r['s_L'], r['h'], dict(x=r.get('x'), kb=r.get('kb')))
                for r in MU.budgets(60, 12, K)],
    'gd3': [_row(r['s_L'], r['s_L'], r['W'], dict(x=c[0], kb=c[1])) for c, r in _gt.items()],
}
TIMINGS['gd4'] = TIMINGS['gd3']
_TDD_ROWS = [r for r in TDD.table(K) if r['ctrl'] == 120 and r['n'] == 12]
_MUT_ROWS = [r for r in MUT.table(60, K) if r['n'] == 12]
for _p in TDD.PATTERNS:
    TIMINGS[f'tdd_{_p}'] = [_row(r['s_S'], r['s_L'], r['W'], dict(x=r['x'], kb=r['kb'], phase=r['phase']))
                            for r in _TDD_ROWS if r['pattern'] == _p]
for _p in MUT.PATTERNS[60]:
    TIMINGS[f'mu60_{_p}'] = [_row(r['s_S'], r['s_L'], r['W'], dict(x=r['x'], kb=r['kb'], phase=r['phase']))
                             for r in _MUT_ROWS if r['pattern'] == _p]
_RXP = MU.rx_probe(4, TP, k=K, t_max=T)
TIMINGS.update({
    'probe': [_row(64 * TP, K * TP, 0.0, {})], 'rx4': [_row(_RXP['d_S'], _RXP['d_S'] - _RXP['budget'], 0.0, {})],
    'ssb': [_row(SS_US, K * SS_BEAM, 0.0, {})], 'slot': [_row(64 * SLOT, K * SLOT, 0.0, {})],
    'hier6': [_row(12 * TP, K * TP, 0.0, {})]})
# the SS-guided refinement under the two time-division patterns (the long version states tau* = W
# there without tabulating it): one set of m = 2^(6-l) resources, reported with the candidate's
for _l in GD:
    _m = 2 ** (6 - _l)
    for _p in TDD.PATTERNS:
        TIMINGS[f'gd{_l}_{_p}'] = [_row(r[f'guided{_m}'], r['s_L'], r['W'],
                                        dict(x=r['x'], kb=r['kb'], phase=r['phase']))
                                   for r in _TDD_ROWS if r['pattern'] == _p]
        BOUNDS[f'gd{_l}_{_p}'] = GD[_l]
REFS = list(BOUNDS)
NR_REFS = [k for k in REFS if k.startswith(('csirs', 'tdd_', 'mu60_', 'gd'))]
assert set(TIMINGS) == set(BOUNDS)

# ---- what explains the bound: the smallest t bound, the law and scenario of the smallest
# (bootstrap) bound, and the users short of CQI 1 (rule R37's covered population) ----
_recs = lambda key: [src[key]['per_scenario'][n] for src in _SRC.values() for n in CS['scenarios']]
CS_T_MIN = min(r['eta_lb_t'] for k in _LAWS for r in _recs(k))
_kmin = min(_LAWS, key=lambda k: _r16(k, 'eta_lb_min'))
_nmin, _rmin = min(((n, src[_kmin]['per_scenario'][n]) for src in _SRC.values() for n in CS['scenarios']),
                   key=lambda x: x[1]['eta_lb'])
CP = J('covpop.json')
_cps = CP['bounds']['cqi1']['csirs_cfg']
BOUND_WHY = dict(
    law=_kmin, scenario=_nmin, eta_lb=_rmin['eta_lb'], eta_lb_boot=_rmin['eta_lb_boot'],
    eta_lb_t=_rmin['eta_lb_t'], t_min=CS_T_MIN,
    covered_small=min(_cps[v][k]['eta_lb_min'] for v in _cps for k in _LAWS),
    short_of_cqi1=[1 - max(s['cqi1']['users_share'] for s in CP['shares'].values()),
                   1 - min(s['cqi1']['users_share'] for s in CP['shares'].values())],
    short_genie_max=1 - min(s['cqi1']['genie_share'] for s in CP['shares'].values()))
_DS = J('manifest.json')['dataset']['sites']
DATASET = dict(scenarios=len(_DS), users=[min(s['n'] for s in _DS), max(s['n'] for s in _DS)],
               below_0dB=[min(s['frac_below_0dB'] for s in _DS), max(s['frac_below_0dB'] for s in _DS)])
# NR's periodicities (TS 38.331 V19.4.0, 6.3.2): a periodic CSI report every 4-320 slots and SS
# bursts every 5-160 ms, as rule R47 reads them
REPORT_SLOTS, SSB_MS = (4, 5, 8, 10, 16, 20, 40, 80, 160, 320), (5, 10, 20, 40, 80, 160)
# Fig. 1's timing: DDDSU at X3 = KB2 = 14 symbols, the trigger slot giving that timing's largest
# budget, with the schedule nrtdd.py computes and the pattern's symbol kinds
_FIG = max((r for r in _TDD_ROWS if r['pattern'] == 'DDDSU' and (r['x'], r['kb']) == (14, 14)),
           key=lambda r: r['budget'])
_car = TDD.Carrier('DDDSU', _FIG['phase'])
FIGURE = dict(ref='tdd_DDDSU', timing=dict(x=14, kb=14, phase=_FIG['phase']), s_S_us=_FIG['s_S'],
              s_L_us=_FIG['s_L'], W_us=_FIG['W'], budget_us=_FIG['budget'], sweep=_FIG['sweep'],
              reduced_slot=_FIG['reduced_slot'],
              kinds=[''.join(_car.kind(s, y) for y in range(14)) for s in range(16)])


def need(tau, k):
    """The lowest efficiency at which (4) excludes a 4-probe implementation of latency tau against
    reference k, at adverse inputs: the largest over the reference's timings and trigger slots."""
    if k == 'probe':
        return (T - (K * T_AIR + tau)) / (T - DS_HI)
    if k == 'rx4':
        return (T - (K * T_AIR + tau)) / (T - 4 * DS_HI)
    if k == 'hier6':
        return (T - (K * T_AIR + tau)) / (T - 12 * DS_HI / 64)
    if k == 'slot':
        return (T - (K * SLOT + tau)) / (T - 64 * SLOT)
    if k == 'ssb':
        return (T - (K * SS_BEAM + tau)) / (T - SS_US)
    if k in ('csirs30', 'csirs8'):
        n, f = (12, 30) if k == 'csirs30' else (8, 120)
        return max(NRT.threshold(tau, T, n, f, *c, K) for c in NRT.capabilities(f))
    return max((T - (r['s_L'] - r['W'] + tau)) / (T - r['d']) for r in TIMINGS[k])


def need_at(tau, k):
    """The timing (and trigger slot) that sets need(tau, k), where the reference has several."""
    if k in ('probe', 'rx4', 'hier6', 'slot', 'ssb', 'csirs30', 'csirs8'):
        return None
    return max(TIMINGS[k], key=lambda r: (T - (r['s_L'] - r['W'] + tau)) / (T - r['d']))['tag']


def tau_at(k, eta):
    """The latency at which need(., k) equals eta (need decreases in tau)."""
    a, z = 0.0, 4 * T
    for _ in range(200):
        m = (a + z) / 2
        a, z = (a, m) if need(m, k) <= eta else (m, z)
    return z


# ---- implementations: each latency's lower value (the needs) and the 95th percentile shown ----
SC1, SC2, RNK = ('resnet18_vision', 'cpu-8t-pcore'), ('vit_b16_vision', 'gpu-resident'), \
    ('resnet18_vision', 'gpu-resident')
MT = J('matched.json')
_mopt = MT['optimized_build']
DPL = J('deploy.json')
_dep = {g: DPL['groups'][g] for g in ('rn_gpu', 'vit_gpu', 'rn_cpu')}
_dopt = {g: v['optimized_build'] for g, v in _dep.items()}
IMPL = {
    'rn8t_fp32': dict(label='ResNet-18, eight P-core threads, FP32', tau_lo=MGR[SC1]['tau_lo_us'],
                      p95=min(R5ROW[SC1]['charges_us'].values()),
                      runtimes_lo=[v['lower_us'] for v in MGR[SC1]['runtimes'].values()], rules=['R5', 'R21']),
    'vit_gpu_fp32': dict(label='ViT-B/16, GPU, FP32', tau_lo=MGR[SC2]['tau_lo_us'],
                         p95=min(R5ROW[SC2]['charges_us'].values()),
                         runtimes_lo=[v['lower_us'] for v in MGR[SC2]['runtimes'].values()], rules=['R5', 'R21']),
    'rn_gpu_fp32': dict(label='ResNet-18, GPU, FP32', tau_lo=MGR[RNK]['tau_lo_us'],
                        p95=min(R5ROW[RNK]['charges_us'].values()),
                        runtimes_lo=[v['lower_us'] for v in MGR[RNK]['runtimes'].values()], rules=['R5', 'R21']),
    'vit_gpu_fp16': dict(label='ViT-B/16, GPU, FP16 graph (matched study)',
                         tau_lo=MT['secondary'][_mopt]['tau_lo_us'],
                         p95=min(MT['sessions'][s]['builds'][_mopt]['p95_us'] for s in MT['sessions_read']),
                         runtimes_lo=None, rules=['R22']),
}
for _g, _lab in (('rn_gpu', 'ResNet-18, GPU, compiled FP16'), ('vit_gpu', 'ViT-B/16, GPU, FP16 graph'),
                 ('rn_cpu', 'ResNet-18, eight P-core threads, INT8')):
    _s = _dep[_g]['summary']
    IMPL[f'dep_{_g}'] = dict(label=_lab + ' (deployment study)', tau_lo=_s[_dopt[_g]]['tau_lo_us'],
                             p95=_s[_dopt[_g]]['p95_min_us'], runtimes_lo=None, rules=['R38'],
                             speedup=_s['fp32']['p95_min_us'] / _s[_dopt[_g]]['p95_min_us'])
# the fastest call any stored session recorded (timespread.json, rule R50): an exclusion that holds
# there holds for every recorded call, so for their mean too (Proposition 1(d) of the long version)
_TS = J('timespread.json')['implementations']
_FAST = {'rn8t_fp32': 'rn_cpu8t_fp32', 'vit_gpu_fp32': 'vit_gpu_fp32', 'rn_gpu_fp32': 'rn_gpu_fp32',
         'vit_gpu_fp16': 'vit_gpu_fp16', 'dep_rn_cpu': 'rn_cpu8t_int8', 'dep_rn_gpu': 'rn_gpu_fp16c',
         'dep_vit_gpu': 'vit_gpu_fp16'}
for _i, _k in _FAST.items():
    IMPL[_i]['fastest_call'] = _TS[_k]['fastest_call_us']

# the trained measurement-only networks (one P-core): the four fully connected networks, the CNN and
# the early-exit network at its first exit
_CH = {r['component']: r['charge_us'] for r in J('lat_cpu1t_pcore.json')['rows']}
TRAINED = {c: _CH[c] for c in ('fc_k4', 'fc_k8', 'fc_k16', 'fc_k32', 'cnn', 'ee_fresh1')}

# the rules behind each part of the letter, with protocol.json's post hoc flag
PARTS = {'csirs12': ['R16', 'R19', 'R29'], 'csirs30': ['R16', 'R19', 'R29'],
         'csirs8': ['R16', 'R19', 'R29'], 'csirs60': ['R16', 'R35'], 'tdd_DDDSU': ['R16', 'R30'],
         'tdd_7DS8U': ['R16', 'R30'], 'mu60_DDDSU': ['R16', 'R36'], 'mu60_DDDSUUUU': ['R16', 'R36'],
         'gd3': ['R31'], 'gd4': ['R31'], 'probe': ['R5'], 'rx4': ['R44'], 'ssb': ['R29'],
         'slot': ['R29'], 'hier6': ['R17'], 'unresolved': ['R21', 'R41'], 'horizons': ['R41', 'R47'],
         'gd3_DDDSU': ['R30', 'R31'], 'gd3_7DS8U': ['R30', 'R31'], 'gd4_DDDSU': ['R30', 'R31'],
         'gd4_7DS8U': ['R30', 'R31'], 'fastest_call': ['R50']}


def readings():
    out = {'references': {}, 'implementations': {}, 'needs': {}, 'horizons': {}}
    for k in REFS:
        rows, b = TIMINGS[k], BOUNDS[k]
        lo_r, hi_r = min(rows, key=lambda r: r['budget']), max(rows, key=lambda r: r['budget'])
        d_min = min(r['d'] for r in rows)
        out['references'][k] = dict(
            bounds=b, n_timings=len(rows), tau_star_us=[lo_r['budget'], hi_r['budget']],
            tau_star_at=[lo_r['tag'], hi_r['tag']], d_us=[d_min, max(r['d'] for r in rows)],
            W_us=sorted({r['W'] for r in rows}),
            slack_us={e: (1 - b[e]) * (T - d_min) for e in ('meas', 'bound', 'small')},
            post_hoc={r: PROT['rules'][r]['post_hoc'] for r in PARTS[k]})
    for i, v in IMPL.items():
        out['implementations'][i] = {f: v[f] for f in v}
        for k in REFS:
            n = need(v['tau_lo'], k)
            ex = {e: n <= BOUNDS[k][e] for e in ('meas', 'bound', 'small')}
            unres = (not ex['small'] and v['runtimes_lo'] is not None
                     and any(need(t, k) <= BOUNDS[k]['small'] for t in v['runtimes_lo']))
            nf = need(v['fastest_call'], k)
            out['needs'][f'{i}|{k}'] = dict(
                need=n, at=need_at(v['tau_lo'], k), excluded=ex,
                status='excluded' if ex['small'] else 'unresolved' if unres else 'not excluded',
                need_fastest_call=nf, excluded_at_fastest_call=nf <= BOUNDS[k]['small'])
    out['horizons_long'] = {}
    for i in ('rn8t_fp32', 'vit_gpu_fp32'):
        for tag, refs in (('horizons', NR_REFS), ('horizons_long', [k for k in NR_REFS if k in LONG_REFS])):
            bold = [k for k in refs if out['needs'][f'{i}|{k}']['excluded']['small']]
            for e in ('small', 'meas'):
                hz = {k: T + (IMPL[i]['tau_lo'] - tau_at(k, BOUNDS[k][e])) / (1 - BOUNDS[k][e]) for k in bold}
                kmin = min(hz, key=hz.get)
                out[tag][f'{i}|{e}'] = dict(us=hz[kmin], against=kmin, over=bold)
                if tag == 'horizons':        # each exclusion's own horizon, for any subset of references
                    out.setdefault('horizon_by_ref', {}).update({f'{i}|{k}|{e}': v for k, v in hz.items()})
    out['trained_us'] = dict(min=min(TRAINED.values()), max=max(TRAINED.values()), components=TRAINED)
    out['figure'] = FIGURE
    out['bound_why'] = BOUND_WHY
    out['dataset'] = DATASET
    out['nr_periods'] = dict(report_slots=REPORT_SLOTS, ssb_ms=SSB_MS, T_max_slots=T / SLOT,
                             ssb_within_horizon={h: [p for p in SSB_MS if p * 1000 <= out['horizons'][h]['us']]
                                                 for h in out['horizons']})
    out['post_hoc'] = {p: {r: PROT['rules'][r]['post_hoc'] for r in rs} for p, rs in PARTS.items()}
    return out


def gate():
    g, bad = {}, []
    # G1
    pins = {'nrtdd.py': ('tdd_ssb', 'nrtdd.py'), 'nrtime.py': ('tdd_ssb', 'nrtime.py'),
            'nrmu.py': ('mu_rx', 'nrmu.py'), 'nrtddmu.py': ('tdd_mu', 'nrtddmu.py')}
    g['G1_modules_as_declared'] = all(sha(os.path.join(HERE, f)) == ANA[a]['scripts_sha256'][s]
                                      for f, (a, s) in pins.items())
    g['G1_module_gates'] = bool(TDD.gate_paired() and MU.gate_m() and MUT.gate_p())
    # G2
    g['G2_budget_identity'] = all(abs(r['budget'] - (r['s_S'] - r['s_L'] + r['W'])) < 1e-9
                                  for r in _TDD_ROWS + _MUT_ROWS) and \
        all(abs(r['budget'] - (r['s_S'] - r['s_L'] + r['h'])) < 1e-9 for r in MU.budgets(60, 12, K))
    g['G2_waits'] = {p: sorted({r['W'] for r in TIMINGS[f'tdd_{p}']}) for p in TDD.PATTERNS} == \
        {'DDDSU': [2 * SLOT], '7DS8U': [9 * SLOT]}
    _t = NRT.times(12, 120, 14, 14, K)
    g['G2_paired_top_declared'] = abs(max(r['budget'] for r in TIMINGS['csirs12'])
                                      - (_t['d_S'] - _t['reduced'] + SLOT)) < 1e-9
    # G3
    e12, e8 = MP['efficiency']['12'], MP['efficiency']['8']
    g['G3_csirs_efficiencies'] = (abs(CS_MEAS - min(e12['measured'].values())) < 1e-12
                                  and abs(CS_SMALL - min(e12['bound'].values())) < 1e-12
                                  and abs(CS8_SMALL - min(min(e8['bound'].values()), CS_SMALL)) < 1e-12)
    g['G3_per_probe_margins'] = all(
        abs(MGR[k]['tau_lo_us'] - tau_at('probe', ETA_LB_PROBE) - MGR[k]['margin_us']) < 1e-3
        for k in (SC1, SC2))
    # G4
    g['G4_bounds_ordered'] = all(b['small'] <= b['bound'] <= b['meas'] < 1 for b in BOUNDS.values())
    # G5
    R = readings()
    unres = sorted(key for key, v in R['needs'].items() if v['status'] == 'unresolved'
                   and key.split('|')[1] in LONG_REFS)
    g['G5_unresolved'] = unres == sorted(['rn8t_fp32|tdd_7DS8U', 'rn8t_fp32|rx4', 'rn8t_fp32|ssb',
                                          'vit_gpu_fp32|mu60_DDDSUUUU', 'vit_gpu_fp32|slot',
                                          'rn_gpu_fp32|probe'])
    # G7: under either pattern the refinement reports with the candidate (as the long version asserts)
    g['G7_refinement_reports_with_candidate'] = all(abs(r[f'guided{m}'] - r['s_L']) < 1e-9
                                                    for r in _TDD_ROWS for m in (4, 8))
    # G8: at the fastest stored call the FP32 exclusions at the smallest bounds hold but two, as the
    # long version states (rule R50): ResNet-18's against 60 kHz DDDSU, ViT-B/16's against 4 rx beams
    lost = sorted(key for key, v in R['needs'].items() if key.split('|')[0] in ('rn8t_fp32', 'vit_gpu_fp32')
                  and key.split('|')[1] in LONG_REFS and v['status'] == 'excluded'
                  and not v['excluded_at_fastest_call'])
    g['G8_fastest_call_losses'] = lost == ['rn8t_fp32|mu60_DDDSU', 'vit_gpu_fp32|rx4']
    # G6
    g['G6_trained_within_every_budget'] = max(TRAINED.values()) < min(
        min(r['budget'] for r in TIMINGS[k]) for k in REFS)
    for k, v in g.items():
        if not v:
            bad.append(k)
    g['holds'] = not bad
    return g, bad, R


def main():
    g, bad, R = gate()
    print(f"gate: {g}")
    if bad:
        sys.exit(f"letter: gate failed: {bad}")
    if '--gate' in sys.argv:
        return
    if '--dry' in sys.argv:
        for k in REFS:
            r = R['references'][k]
            print(f"{k:14s} tau* {r['tau_star_us'][0]:8.1f}-{r['tau_star_us'][1]:8.1f}  slack "
                  f"{r['slack_us']['meas']:7.1f}/{r['slack_us']['small']:7.1f}  eta "
                  f"{r['bounds']['meas']:.5f}/{r['bounds']['small']:.5f}  needs " + " ".join(
                      f"{R['needs'][f'{i}|{k}']['need']:.4f}{'*' if R['needs'][f'{i}|{k}']['status'] == 'excluded' else '+' if R['needs'][f'{i}|{k}']['status'] == 'unresolved' else ' '}"
                      for i in ('rn8t_fp32', 'vit_gpu_fp32', 'dep_rn_cpu', 'vit_gpu_fp16')))
        print({k: round(v['us'], 1) for k, v in R['horizons'].items()}, R['trained_us']['min'],
              R['trained_us']['max'])
        return
    assert not os.path.exists(OUT), f"{OUT} exists: written once"
    files = ('protocol.json', 'rules_rx.json', 'report_rx.json', 'rho.json', 'margins.json',
             'reconcile.json', 'margins_post.json', 'results_rx.json', 'validation_rx.json',
             'heldout_rx.json', 'csirs.json', 'csirs_cfg.json', 'levels.json', 'ssbref.json',
             'uepairs.json', 'refbounds.json', 'matched.json', 'deploy.json', 'lat_cpu1t_pcore.json',
             'timespread.json', 'covpop.json', 'manifest.json')
    out = dict(script_sha256=sha(os.path.abspath(__file__)), gate=g, K=K, T_max_us=T, t_p_us=TP,
               modules_sha256={f: sha(os.path.join(HERE, f)) for f in
                               ('nrtime.py', 'nrtdd.py', 'nrmu.py', 'nrtddmu.py')},
               inputs_sha256={f: sha(os.path.join(HERE, f)) for f in files}, **R)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
