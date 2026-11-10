#!/usr/bin/env python3
"""Rule R50: how much an implementation's 95th-percentile latency moves between timing sessions,
and the lowest one any session recorded (protocol.json analysis 'timespread', declared before it
ran; R50 is post hoc).  Nothing is timed; it reads stored timing records only.

Records, each a session (or a run of repeated runs) of one harness:
  atlas    reconcile.json r5_rows: the stored timings' charge of every runtime (Phase-1 and full
           call counts, the whole-decision TorchScript build), one session each
  trace    reconcile.json 'repeated' (rule R20): the three runs of each decisive case, each run's
           charge (the largest of its seeds' 95th percentiles), per runtime (eager, TorchScript)
  matched  matched.json (rule R22): sessions 1-3, ViT-B/16 on the GPU, FP32 and FP16 (CUDA graph)
  cadence  cadence.json (rule R24): sessions 1-18, the same builds at spacings 0, 10 and 50 ms
  audit    the decision audit's sessions (rules R25-R26, latency/profile_audit): every slot of the
           frozen order, its record that did not fail; the window's calls; nearest-rank 95th
           percentile computed here as matched.py computes it
  deploy   deploy.json (rule R38): sessions 1-3 of each group, FP32 and the frozen build
Implementations: ResNet-18 on eight P-core threads (FP32; INT8), ResNet-18 on the GPU (FP32;
compiled FP16), ViT-B/16 on the GPU (FP32; FP16).

Readings, per implementation:
  (a) per harness, runtime and condition (spacing), the sessions' 95th percentiles and the ratio of
      the largest to the smallest;
  (b) over every session of every harness and runtime: the lowest 95th percentile, its record,
      and the ratio of the largest to the smallest;
  (c) the fastest call any record stores (its minimum).
The paper's verdicts take each FP32 implementation's lowest charge over the atlas and trace runtimes
(Table IV); reading (b) adds the later studies' FP32 sessions, which time an image-only workload.

Gate S1: every charge the paper's Table IV uses is among the sessions read (the three FP32
implementations' r5_rows charges; the FP16, INT8 and compiled builds' deploy.json and matched.json
session 95th percentiles), so reading (b) is at most each.
Gate S2: each audit session recomputed here has the call count and minimum audit_direct.json
stores for it.

usage:  python3 timespread.py [--gate]      -> timespread.json (refuses to overwrite it)
"""
import hashlib, json, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAT = os.path.join(HERE, '..', 'latency')
OUT = os.path.join(HERE, 'timespread.json')
IMPL = {'rn_cpu8t_fp32': ('resnet18_vision', 'cpu-8t-pcore'),
        'rn_gpu_fp32': ('resnet18_vision', 'gpu-resident'),
        'vit_gpu_fp32': ('vit_b16_vision', 'gpu-resident')}
TRACE = {'A': 'rn_cpu8t_fp32', 'C': 'rn_gpu_fp32', 'B': 'vit_gpu_fp32'}
MATCHED = {'fp32': 'vit_gpu_fp32', 'fp16_graph': 'vit_gpu_fp16'}
AUDIT = {'resnet18_ts': ('rn_gpu_fp32', 'TorchScript'), 'vit_fp32': ('vit_gpu_fp32', 'eager'),
         'vit_fp16_graph': ('vit_gpu_fp16', 'CUDA graph')}
DEPLOY = {('rn_cpu', 'fp32'): ('rn_cpu8t_fp32', 'eager'), ('rn_cpu', 'int8'): ('rn_cpu8t_int8', 'int8'),
          ('rn_gpu', 'fp32'): ('rn_gpu_fp32', 'eager'),
          ('rn_gpu', 'fp16_cmax'): ('rn_gpu_fp16c', 'torch.compile'),
          ('vit_gpu', 'fp32'): ('vit_gpu_fp32', 'eager'),
          ('vit_gpu', 'fp16_graph'): ('vit_gpu_fp16', 'CUDA graph')}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def J(name):
    return json.load(open(os.path.join(HERE, name)))


def p95(D):
    return float(np.sort(D)[math.ceil(0.95 * D.size) - 1])


def sessions():
    """Every session record: dicts with impl, harness, runtime, condition, session, p95_us,
    min_us (None where the record keeps no minimum)."""
    out = []
    RC = J('reconcile.json')
    for r in RC['r5_rows']:
        key = next((k for k, v in IMPL.items() if v == (r['component'], r['profile'])), None)
        if key is None:
            continue
        for rt, c in r['charges_us'].items():
            if 'repeated runs' in rt:
                continue                       # the trace's runs, read below one by one
            out.append(dict(impl=key, harness='atlas', runtime=rt, condition='', session='1',
                            p95_us=c, min_us=None))
    for case, impl in TRACE.items():
        rep = RC['repeated'][case]
        for rt, v in rep['runtimes'].items():
            for i, c in enumerate(v['run_charges_us']):
                out.append(dict(impl=impl, harness='trace', runtime=rt.split(',')[0],
                                condition='', session=str(i + 1), p95_us=c, min_us=None))
        out[-1]['fastest_call_us'] = RC['fastest_call_repeated_us'][case]
    M = J('matched.json')
    for s, v in M['sessions'].items():
        for b, x in v['builds'].items():
            out.append(dict(impl=MATCHED[b], harness='matched', runtime=b, condition='',
                            session=s, p95_us=x['p95_us'], min_us=x['min_us']))
    C = J('cadence.json')
    for s, v in C['sessions'].items():
        out.append(dict(impl=MATCHED[v['build']], harness='cadence', runtime=v['build'],
                        condition=f"{v['spacing_ms']} ms", session=s, p95_us=v['p95_us'],
                        min_us=v['min_us']))
    import audit_screen as AS
    fz = AS.frozen()[2]['frozen']
    for o in fz['order']:
        stem = None
        for st in (f"a{o['slot']:02d}_a2", f"a{o['slot']:02d}"):
            p = os.path.join(AS.OUTD, st + '.json')
            if os.path.exists(p) and not json.load(open(p))['failed']:
                stem = st
                break
        assert stem, f"slot {o['slot']} has no record that did not fail"
        z = np.load(os.path.join(AS.OUTD, stem + '.npz'))
        s, e = z['start_ns'], z['end_ns']
        D = (e[s >= int(z['window_from_ns'])] - s[s >= int(z['window_from_ns'])]) / 1e3
        impl, rt = AUDIT[o['build']]
        out.append(dict(impl=impl, harness='audit', runtime=rt, condition='10 ms',
                        session=str(o['round']), p95_us=p95(D), min_us=float(D.min()),
                        n=int(D.size), stem=stem))
    DP = J('deploy.json')
    for g, G in DP['groups'].items():
        for s, v in G['sessions'].items():
            for b, x in v['builds'].items():
                impl, rt = DEPLOY[(g, b)]
                out.append(dict(impl=impl, harness='deploy', runtime=rt, condition='', session=s,
                                p95_us=x['p95_us'], min_us=x['min_us']))
    return out


def readings(S):
    R = {}
    for impl in sorted({x['impl'] for x in S}):
        rows = [x for x in S if x['impl'] == impl]
        groups = {}
        for x in rows:
            groups.setdefault(f"{x['harness']}|{x['runtime']}|{x['condition']}", []).append(x)
        per = {k: {'sessions': len(v), 'p95_us': [x['p95_us'] for x in v],
                   'ratio': max(x['p95_us'] for x in v) / min(x['p95_us'] for x in v)}
               for k, v in groups.items()}
        low = min(rows, key=lambda x: x['p95_us'])
        mins = [x['min_us'] for x in rows if x.get('min_us') is not None] + \
            [x['fastest_call_us'] for x in rows if 'fastest_call_us' in x]
        R[impl] = {'groups': per,
                   'within_ratio_max': max((v['ratio'] for v in per.values() if v['sessions'] > 1),
                                           default=None),
                   'lowest_p95_us': low['p95_us'],
                   'lowest_at': {k: low[k] for k in ('harness', 'runtime', 'condition', 'session')},
                   'highest_p95_us': max(x['p95_us'] for x in rows),
                   'overall_ratio': max(x['p95_us'] for x in rows) / low['p95_us'],
                   'fastest_call_us': min(mins) if mins else None,
                   'n_sessions': len(rows)}
    return R


def gate(S):
    g, bad = {}, []
    RC = J('reconcile.json')
    have = {(x['impl'], round(x['p95_us'], 6)) for x in S}
    for r in RC['r5_rows']:
        key = next((k for k, v in IMPL.items() if v == (r['component'], r['profile'])), None)
        if key is not None:
            for rt, c in r['charges_us'].items():
                if (key, round(c, 6)) not in have:
                    bad.append(('S1', key, rt, c))
    DP = J('deploy.json')
    for g_, G in DP['groups'].items():
        for s, v in G['sessions'].items():
            for b, x in v['builds'].items():
                if (DEPLOY[(g_, b)][0], round(x['p95_us'], 6)) not in have:
                    bad.append(('S1', g_, b, s))
    for s, v in J('matched.json')['sessions'].items():
        for b, x in v['builds'].items():
            if (MATCHED[b], round(x['p95_us'], 6)) not in have:
                bad.append(('S1', 'matched', b, s))
    g['S1_paper_charges_read'] = not [b for b in bad if b[0] == 'S1']
    AD = J('audit_direct.json')['timing']
    import audit_screen as AS
    fz = AS.frozen()[2]['frozen']
    aud = [x for x in S if x['harness'] == 'audit']
    for b in AS.BUILDS:
        impl = AUDIT[b][0]
        mine = {int(x['session']): x for x in aud if x['impl'] == impl}
        cal = AD[b]['calibration']
        for i, rnd in enumerate(range(1, fz['calibration_rounds'] + 1)):
            if mine[rnd]['n'] != cal['n_calls'][i]:
                bad.append(('S2', b, rnd, 'n'))
        for lt in AD[b]['later']:
            x = mine[lt['round']]
            if x['n'] != lt['n_calls'] or abs(x['min_us'] - lt['min_us']) > 1e-9:
                bad.append(('S2', b, lt['round']))
        if abs(min(mine[r]['min_us'] for r in range(1, fz['calibration_rounds'] + 1))
               - cal['min_us']) > 1e-9:
            bad.append(('S2', b, 'calibration min'))
    g['S2_audit_sessions_match'] = not [b for b in bad if b[0] == 'S2']
    g['holds'] = not bad
    return g, bad


def main():
    S = sessions()
    g, bad = gate(S)
    print(f"gate: {g}")
    if bad:
        sys.exit(f"timespread: gate failed: {bad[:10]}")
    if '--gate' in sys.argv:
        return
    assert not os.path.exists(OUT), f"{OUT} exists: written once"
    R = readings(S)
    out = {'rule': 'R50', 'script_sha256': sha(os.path.abspath(__file__)), 'gate': g,
           'sessions': S, 'implementations': R,
           'inputs_sha256': {f: sha(os.path.join(HERE, f)) for f in (
               'reconcile.json', 'matched.json', 'cadence.json', 'deploy.json',
               'audit_direct.json')}}
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, v in R.items():
        print(f"{k}: {v['n_sessions']} sessions; within-harness ratio up to "
              f"{v['within_ratio_max']}; overall {v['overall_ratio']:.3f}; lowest "
              f"{v['lowest_p95_us']:.1f} us at {v['lowest_at']}; fastest call {v['fastest_call_us']}")
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
