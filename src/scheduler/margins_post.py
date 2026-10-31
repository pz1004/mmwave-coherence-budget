#!/usr/bin/env python3
"""The exclusion margins read as rules R16, R19 and R21 prescribe (protocol.json, the change of
2026-10-31): the CSI-RS thresholds, margins and delay limits the paper states, and each
exclusion's excess under channel aging, at the runtimes rule R21 reads.  Nothing is timed,
simulated or re-solved, and no stored file changes.

margins.py and reconcile.py store beside each margin one under the CSI-RS procedure (nr_dS_us,
nr_dL_lo_us, nr_margin_us) from the resources' ends alone: d_L the end of the reduced set's fourth
resource less one slot plus the latency's lower value, d_S the end of the sweep's 64th, 12
one-symbol resources per 125 us slot, the short last set at the start of its slot's last 12
symbols, every report time and common delay zero, at csirs.json's bounds.  Rule R19 made every
CSI-RS threshold and margin the paper states nrtime.py's, report times included, and rule R16
reads the lowest efficiency and bound over the exact comparison and the configured variants.
aging_bound.py stores each exclusion's excess over tau* at the stored timings; rule R21 reads
every number computed from the margins at every runtime measured.  No stored file held what these
rules prescribe.  The superseded fields stay as written, because other stored files record those
files' hashes.

K, T_max and the implementations are margins.json's and reconcile.json's; times are in us.

  timings     nrtime.times per control carrier (120, 30 kHz), resources per slot (12, 8) and
              declarable timing (beamReportTiming x, beamSwitchTiming kb): the sweep's last
              report s_S and the reduced set's s_L, from the first triggering slot's start
  efficiency  per resources per slot and estimator|samples (coherent or per_re; 198, 66 and 33
              samples per resource, densities 3, 1 and 0.5), the lowest over its comparisons of
              'bound', the simultaneous one-sided bound at 0.05/7, 'measured', the simulated
              efficiency, and 'bound_needed', rule R18's bound at 0.05 per scenario.  12 per
              slot: csirs.json's exact comparison and csirs_cfg.json's 'spread|worst' and
              'contiguous|worst' (R16), 'bound_needed' from levels.json over the same three; 8 per
              slot: the exact comparison and levels.json's 'spread8|worst' and 'contiguous8|worst'
              (R19)
  rows        per implementation, reconcile.json's margin row (R21): the latency of record tau,
              the fastest runtime's 95th percentile, and the latency's lower value tau_lo.  Per
              variant carrier|resources per slot, the threshold, nrtime.worst at tau, the highest
              over the declarable timings, with the timing that sets it; the margin of (4),
              d_L - (1 - eta) T_max - eta d_S with d_L = s_L - slot + latency and d_S = s_S, the
              smallest over the declarable timings, at tau_lo ('adverse') and at tau ('record'),
              at each of the variant's efficiencies.  At 120 kHz with 12 per slot and tau_lo, at
              the smallest 'bound' over both estimators and the three densities: the timing that
              makes the margin smallest, its d_L and d_S, the margin, and the horizon
              (d_L - eta d_S) / (1 - eta), the longest period up to which the exclusion holds
  limits      ResNet-18 on the GPU, which the procedure excludes and the per-probe model does
              not, at tau with 12 per slot: per carrier and timing, the longest delivery time of
              the sweep's last report at which a camera-only candidate (K = 0) is still excluded
              (nrtime.k0_room), at density 3's coherent 'measured' and 'bound'; per carrier and
              estimator|samples, the longest delay of the sweep's report that the reduced set's
              does not share at which the exclusion holds (nrtime.unshared_room), the smallest
              over the timings, at 'measured'
  aging       each exclusion of rule R5 at R21's runtimes: tau and its runtime, tau*, the excess
              tau - tau*, and per speed whether it exceeds aging_bound.json's largest slack,
              beside aging_bound.json's tau and excess at the stored timings

Gates:
  M1_superseded_reproduced  the resources' ends (nrtime.res_end in csirs.py's placement),
                            margins.json's and reconcile.json's rows and csirs.json's bounds
                            reproduce every nr_* field of both files to 1e-6
  M2_thresholds_levels      nrtime.worst at levels.json's tau, ResNet-18 on the GPU's latency of
                            record, reproduces its four thresholds to 1e-12
  M3_nrtime_pin             nrtime.py is the file levels.json records
  M4_aging_reproduced       at rules.json's charges, tau and the excess reproduce
                            aging_bound.json's exactly, and R21's runtimes move only
                            reconcile.json's changed rows
  M5_rows                   reconcile.json's gate holds; its rows name rules.json's
                            implementations, with the same tau* and exclusions; each margin row's
                            tau is the smallest of its R21 charges

usage:  python3 margins_post.py [--check-only]      -> margins_post.json (written once)
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nrtime as NRT

OUT = os.path.join(HERE, 'margins_post.json')
INPUTS = ('reconcile.json', 'margins.json', 'csirs.json', 'csirs_cfg.json', 'levels.json',
          'aging_bound.json', 'rules.json')
VARIANTS = ((120, 12), (30, 12), (120, 8), (30, 8))
LEVELS_KEY = {(120, 12): '120k12', (30, 12): '30k12', (120, 8): '120k8', (30, 8): '30k8'}
SOURCES = {12: ('exact', 'spread|worst', 'contiguous|worst'),
           8: ('exact', 'spread8|worst', 'contiguous8|worst')}
QUANTITIES = ('bound', 'measured', 'bound_needed')
LIMITS_ROW = ('resnet18_vision', 'gpu-resident')
J = lambda f: json.load(open(os.path.join(HERE, f)))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
vk = lambda f, n: f"{f}|{n}"


def efficiency(CS, CG, LV, keys):
    """Per resources per slot and estimator|samples, the lowest over the comparisons."""
    cfg = {'exact': CS['configs'], **{v: CG['variants'][v]['configs'] for v in SOURCES[12][1:]}}
    sm = LV['summary']
    return {
        '12': {'bound': {k: min(c[k]['eta_lb_min'] for c in cfg.values()) for k in keys},
               'measured': {k: min(c[k]['eta_min'] for c in cfg.values()) for k in keys},
               'bound_needed': {k: min(sm[s][k]['eta_lb_need_min'] for s in SOURCES[12])
                                for k in keys}},
        # levels.json's summary holds the exact comparison's efficiency, not its bound, which is
        # csirs.json's
        '8': {'bound': {k: min([CS['configs'][k]['eta_lb_min']]
                               + [sm[s][k]['eta_lb_min'] for s in SOURCES[8][1:]]) for k in keys},
              'measured': {k: min(sm[s][k]['eta_min'] for s in SOURCES[8]) for k in keys},
              'bound_needed': {k: min(sm[s][k]['eta_lb_need_min'] for s in SOURCES[8])
                               for k in keys}}}


def timings(k):
    """nrtime's report starts per variant and declarable timing."""
    out = {}
    for f, n in VARIANTS:
        out[vk(f, n)] = []
        for x, kb in NRT.capabilities(f):
            t = NRT.times(n, f, x, kb, k)
            out[vk(f, n)].append({'x': x, 'kb': kb, 's_S_us': t['d_S'], 's_L_us': t['reduced']})
    return out


def margin(tm, tau, e, t_max):
    """(4)'s margin at efficiency e, the smallest over the declarable timings."""
    return min(c['s_L_us'] - NRT.SLOT + tau - (1 - e) * t_max - e * c['s_S_us'] for c in tm)


def compute():
    CS, CG, LV = J('csirs.json'), J('csirs_cfg.json'), J('levels.json')
    MG, RC, AB, RU = J('margins.json'), J('reconcile.json'), J('aging_bound.json'), J('rules.json')
    k, t_max = MG['K'], MG['t_max_us']
    dens = CS['density_samples']
    keys = [f"{e}|{n}" for e in ('coherent', 'per_re') for n in dens.values()]
    R5S = {(r['component'], r['profile']): r for r in RU['R5']['rows']}
    R5R = {(r['component'], r['profile']): r for r in RC['r5_rows']}
    gates = {}

    # M1: the superseded fields, from the resources' ends alone at csirs.json's bounds
    d_old = NRT.res_end(63, 12, None)
    eta_old = {d: CS['configs'][f'coherent|{n}']['eta_lb_min'] for d, n in dens.items()}
    ok = abs(MG['nr_dS_us'] - d_old) < 1e-6 and MG['eta_lo']['csirs'] == eta_old
    for r in MG['rows'] + RC['margin_rows']:
        ok &= abs(r['nr_dL_lo_us'] - (NRT.res_end(k - 1, 12, None) - NRT.SLOT
                                      + r['tau_lo_us'])) < 1e-6
        ok &= all(abs(r['nr_margin_us'][d] - (r['nr_dL_lo_us'] - (1 - e) * t_max - e * d_old))
                  < 1e-6 for d, e in eta_old.items())
    gates['M1_superseded_reproduced'] = bool(ok)

    # M2, M3: nrtime.py is the corrected model levels.json's thresholds come from
    rn = next(m for m in RC['margin_rows'] if (m['component'], m['profile']) == LIMITS_ROW)
    gates['M2_thresholds_levels'] = bool(
        LV['tau_us'] == rn['tau_us'] and LV['t_max_us'] == t_max
        and all(abs(NRT.worst(LV['tau_us'], t_max, n, f, k)[0] - LV['thresholds'][LEVELS_KEY[(f, n)]])
                < 1e-12 for f, n in VARIANTS))
    gates['M3_nrtime_pin'] = sha(os.path.join(HERE, 'nrtime.py')) == LV['nrtime_sha256']

    # M4: aging_bound.json's reading at the stored timings; R21 moves only its changed rows
    ok = True
    for e in AB['exclusions']:
        tau_st = min(R5S[(e['component'], e['profile'])]['charges_us'].values())
        ok &= tau_st == e['tau_us'] and tau_st - e['tau_star_us'] == e['excess_us']
    moved = sorted([list(kk) for kk in R5S if min(R5S[kk]['charges_us'].values())
                    != min(R5R[kk]['charges_us'].values())])
    gates['M4_aging_reproduced'] = bool(ok and moved == sorted(RC['changed']))

    # M5: the rows R21 reads are rules.json's implementations
    gates['M5_rows'] = bool(
        RC['gate']['holds']
        and {(m['component'], m['profile']) for m in RC['margin_rows']} == set(R5S) == set(R5R)
        and all(m['tau_us'] == min(R5R[(m['component'], m['profile'])]['charges_us'].values())
                for m in RC['margin_rows'])
        and all(R5S[kk]['tau_star_us'] == R5R[kk]['tau_star_us']
                and R5S[kk]['excluded'] == R5R[kk]['excluded'] for kk in R5S)
        and {(e['component'], e['profile']) for e in AB['exclusions']}
        == {kk for kk in R5S if R5S[kk]['excluded']})
    gate = dict(gates, holds=all(gates.values()))

    eff = efficiency(CS, CG, LV, keys)
    tms = timings(k)
    rows = []
    for m in RC['margin_rows']:
        tau, tlo = m['tau_us'], m['tau_lo_us']
        row = {'component': m['component'], 'profile': m['profile'],
               'runtime': m['fastest_runtime'], 'tau_us': tau,
               'tau_lo_runtime': m['tau_lo_runtime'], 'tau_lo_us': tlo,
               'threshold': {}, 'margin_us': {'adverse': {}, 'record': {}}}
        for f, n in VARIANTS:
            e_, x, kb = NRT.worst(tau, t_max, n, f, k)
            row['threshold'][vk(f, n)] = {'eta': e_, 'x': x, 'kb': kb}
            for name, lat in (('adverse', tlo), ('record', tau)):
                row['margin_us'][name][vk(f, n)] = {
                    q: {kk: margin(tms[vk(f, n)], lat, e, t_max) for kk, e in eff[str(n)][q].items()}
                    for q in QUANTITIES}
        # the paper's procedure at adverse inputs and the smallest simultaneous bound
        b = eff['12']['bound']
        kmin = min(b, key=b.get)
        e = b[kmin]
        c = min(tms['120|12'], key=lambda c: (c['s_L_us'] - NRT.SLOT + tlo - (1 - e) * t_max
                                              - e * c['s_S_us']))
        dl = c['s_L_us'] - NRT.SLOT + tlo
        row['least_bound'] = {'key': kmin, 'eta': e, 'x': c['x'], 'kb': c['kb'],
                              's_S_us': c['s_S_us'], 's_L_us': c['s_L_us'], 'd_L_us': dl,
                              'margin_us': row['margin_us']['adverse']['120|12']['bound'][kmin],
                              'horizon_us': (dl - e * c['s_S_us']) / (1 - e)}
        rows.append(row)

    lr = next(r for r in rows if (r['component'], r['profile']) == LIMITS_ROW)
    c3 = f"coherent|{dens['3']}"
    limits = {'component': LIMITS_ROW[0], 'profile': LIMITS_ROW[1], 'tau_us': lr['tau_us'],
              'k0_room_us': {str(f): [{'x': x, 'kb': kb,
                                       'measured': NRT.k0_room(eff['12']['measured'][c3],
                                                               lr['tau_us'], t_max, 12, f, x, kb),
                                       'bound': NRT.k0_room(eff['12']['bound'][c3],
                                                            lr['tau_us'], t_max, 12, f, x, kb)}
                                      for x, kb in NRT.capabilities(f)] for f in (120, 30)},
              'unshared_room_us': {str(f): {kk: min(NRT.unshared_room(e, lr['tau_us'], t_max, 12,
                                                                      f, x, kb, k)
                                                    for x, kb in NRT.capabilities(f))
                                            for kk, e in eff['12']['measured'].items()}
                                   for f in (120, 30)}}

    aging = []
    for e in AB['exclusions']:
        ch = R5R[(e['component'], e['profile'])]['charges_us']
        rt = min(ch, key=ch.get)
        ex = ch[rt] - e['tau_star_us']
        aging.append({'component': e['component'], 'profile': e['profile'], 'runtime': rt,
                      'tau_us': ch[rt], 'tau_star_us': e['tau_star_us'], 'excess_us': ex,
                      'holds_at': {v: ex > s for v, s in AB['max_slack_us'].items()},
                      'stored_tau_us': e['tau_us'], 'stored_excess_us': e['excess_us']})

    return {'rules': ['R16', 'R19', 'R21'], 'script_sha256': sha(os.path.abspath(__file__)),
            'nrtime_sha256': sha(os.path.join(HERE, 'nrtime.py')),
            'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in INPUTS},
            'K': k, 'T_max_us': t_max, 'slot_us': NRT.SLOT, 'densities': dens, 'gate': gate,
            'timings': tms, 'efficiency': eff, 'rows': rows, 'limits': limits,
            'aging': {'max_slack_us': AB['max_slack_us'], 'exclusions': aging}}


def main():
    out = compute()
    print(json.dumps(out['gate']))
    if not out['gate']['holds']:
        raise SystemExit("gate failed: nothing written")
    for r in out['rows']:
        lb = r['least_bound']
        print(f"{r['component']:18s} {r['profile']:13s} tau {r['tau_us']:9.2f} "
              f"eta* {r['threshold']['120|12']['eta']:8.4f}  at {lb['key']} {lb['eta']:.4f}: "
              f"margin {lb['margin_us']:9.1f}  horizon {lb['horizon_us'] / 1e3:8.1f} ms")
    for e in out['aging']['exclusions']:
        print(f"aging {e['component']:18s} {e['profile']:13s} tau {e['tau_us']:9.2f} "
              f"({e['runtime']}) excess {e['excess_us']:9.2f}, stored {e['stored_excess_us']:9.2f}")
    if '--check-only' in sys.argv:
        same = json.loads(json.dumps(out)) == json.load(open(OUT))
        print(f"margins_post.json {'reproduces' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    assert not os.path.exists(OUT), f"{OUT} exists: written once (use --check-only)"
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
