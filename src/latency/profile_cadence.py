#!/usr/bin/env python3
"""The cadence study: the matched study's image-only ViT-B/16 on the GPU, eager FP32 and the FP16
build in one CUDA graph, each invoked at three spacings.

Declared in protocol.json (analysis 'cadence', rule R24) before any of its sessions ran.
src/scheduler/cadence.py writes the session order into the analysis's frozen record (--freeze)
and reads the sessions (the default).

WHY.  The matched study (rule R22) timed both builds under sustained serialized service, back to
back.  This addendum asks whether a build's screening verdict changes when the same desktop
invokes inference at another spacing.  The spacing is the time between the starts of successive
calls.  It is a profiling condition set separately from the acquisition period T, not derived from
it, and it does not model traffic.

WHAT.  profile_matched.py's timed call, network, weights, frames, threads and pinning.  One
process is one session: one build at one spacing Delta (0, back to back; 10 ms; 50 ms).  After
the quiet check, the process loads the network and builds it (graph capture included, untimed),
then makes one call.  It then idles for 60 s, warms up for 60 s under its spacing and records
every call of a 120 s window.  Last come the execution checks on 256 further frames: logits of
shape (1, 64), all finite, the index the argmax of the same build's logits.  Pacing: call i + 1
starts no earlier than max(s_i + Delta, e_i), s_i and e_i the start and end of call i.  The
process sleeps until then, so there is no queue and the wait is not part of a call's completion
time.  An overrun (e_i > s_i + Delta) starts the next call at once.  The warm-up and the window
are one schedule, so the pacing carries across their boundary; a call belongs to the phase in
which it starts, and the schedule ends with the last call due to start within the window.  Each
call's actual start and end are kept, warm-up included.  The GPU's state is logged every second
by nvidia-smi, as in profile_matched.py, and the machine's state is read before the warm-up and
after the window.

  --slot k     session k (1-18) of the frozen order: its build and spacing are read from
               protocol.json, never given on the command line
  --attempt 2  the one replacement R24 (iv) allows, after an objective execution failure
  --smoke DIR  both builds at every spacing for a few seconds each, into DIR outside this
               directory, without the frozen record or a quiet machine; its output is discarded

Exit status: 0; 3 if the machine never became quiet (nothing timed); 4 on an objective
execution failure, with its record written.  An objective failure is a crash or CUDA error,
failed execution checks or a missing timing record.  Nothing else repeats a session:
temperature, clock changes, drift, encoder activity and new GPU processes are recorded as
results.  Outputs, never overwritten: profile_cadence/c<kk>[_a2].{json,npz}, each with its
nvidia-smi log c<kk>[_a2]_gpu.csv.

usage:  python3 profile_cadence.py --slot 1..18 [--attempt 2] | --smoke DIR
"""
import argparse, copy, hashlib, json, os, sys, time, traceback, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'scheduler'))
import profile_matched as PM

OUT = os.path.join(HERE, 'profile_cadence')
PROTOCOL = PM.PROTOCOL
KEY = 'cadence'
SELF = '../latency/profile_cadence.py'           # this script, as protocol.json names it
BUILDS = ('fp32', 'fp16_graph')
SPACINGS_MS = (0, 10, 50)
N_SLOTS = 18
IDLE_S, WARMUP_S, WINDOW_S = 60, 60, 120
SMOKE_S = 1.0
CAP = 400_000                                    # calls per phase, far above any phase's count


def paced(fn, seconds, delta_ns):
    """Calls of fn scheduled to start within `seconds`, call i + 1 starting no earlier than
    max(s_i + delta, e_i); the process sleeps until then.  Returns each call's start and end, in
    ns from the loop's start, and the loop's start as an epoch time."""
    import numpy as np
    pc = time.perf_counter_ns
    s = np.empty(CAP, dtype=np.int64)
    e = np.empty(CAP, dtype=np.int64)
    t_begin, epoch = pc(), time.time()
    stop = t_begin + int(seconds * 1e9)
    nxt, i = t_begin, 0
    while i < CAP:
        wait = nxt - pc()
        if wait > 0:
            time.sleep(wait / 1e9)
        t0 = pc()
        fn()
        t1 = pc()
        s[i], e[i] = t0 - t_begin, t1 - t_begin
        i += 1
        nxt = max(t0 + delta_ns, t1)
        if nxt >= stop:
            break
    assert i < CAP, "a phase reached the call cap"
    return s[:i].copy(), e[:i].copy(), epoch


def frozen_record():
    """The analysis entry's frozen record and the sha256 of that record."""
    P = json.load(open(PROTOCOL))
    A = [a for a in P['analyses'] if a['key'] == KEY]
    assert len(A) == 1, f"protocol.json declares analysis '{KEY}' exactly once"
    fz = A[0].get('frozen')
    assert fz, "the session order is not frozen in protocol.json: run cadence.py --freeze first"
    return fz, hashlib.sha256(json.dumps(fz, sort_keys=True).encode()).hexdigest()


def summary(s, e, delta_ns):
    import numpy as np
    d = e - s
    gap = np.diff(s)
    return {'n': int(d.size), 'mean_us': float(d.mean()) / 1e3,
            'p95_us': float(np.sort(d)[int(np.ceil(0.95 * d.size)) - 1]) / 1e3,
            'min_us': float(d.min()) / 1e3, 'max_us': float(d.max()) / 1e3,
            'overruns': int((d > delta_ns).sum()) if delta_ns > 0 else None,
            'start_to_start_median_us': float(np.median(gap)) / 1e3 if gap.size else None}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--slot', type=int, choices=range(1, N_SLOTS + 1))
    g.add_argument('--smoke', default='')
    ap.add_argument('--attempt', type=int, default=1, choices=(1, 2))
    a = ap.parse_args()
    smoke = bool(a.smoke)
    out_dir = os.path.abspath(a.smoke) if smoke else OUT
    if smoke:
        assert not out_dir.startswith(os.path.abspath(OUT)), "a smoke test never writes the outputs"
        assert a.attempt == 1, "a smoke test has one attempt"
    stem = 'smoke' if smoke else f"c{a.slot:02d}" + ('_a2' if a.attempt == 2 else '')
    out = os.path.join(out_dir, stem + '.json')
    for p in (out, out[:-5] + '.npz', out[:-5] + '_gpu.csv'):
        assert not os.path.exists(p), f"{p} exists: an output is written once"

    # checks that need no torch
    import trace_decisive as TD
    assert PM.sha_file(os.path.join(HERE, 'models.py')) == TD.SRC_SHA256['models.py'], \
        "models.py changed"
    assert sorted(os.sched_getaffinity(0)) == PM.AFFINITY, "run under taskset -c 0-15"
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(v) == '1', f"{v} must be 1"
    if smoke:
        fz, fz_sha = None, None
        plan = [(b, ms) for b in BUILDS for ms in SPACINGS_MS]
        phase_s = (SMOKE_S, SMOKE_S, 2 * SMOKE_S)
    else:
        fz, fz_sha = frozen_record()
        assert PM.sha_file(os.path.abspath(__file__)) == fz['scripts_sha256'][SELF], \
            "profile_cadence.py differs from the frozen script"
        slot = [o for o in fz['order'] if o['slot'] == a.slot]
        assert len(slot) == 1, a.slot
        plan = [(slot[0]['build'], slot[0]['spacing_ms'])]
        phase_s = (fz['phases_s']['idle'], fz['phases_s']['warmup'], fz['phases_s']['window'])
        assert phase_s == (IDLE_S, WARMUP_S, WINDOW_S)

    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    import numpy as np
    import calibrate                               # silences warnings, as run_all.sh did
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Quadro RTX 8000'
    dev = torch.device('cuda')
    env = {**calibrate.env_record('gpu-resident', 1),
           'torchvision': __import__('torchvision').__version__,
           'triton': __import__('triton').__version__,
           'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
           'driver': PM._run(['nvidia-smi', '--query-gpu=driver_version',
                              '--format=csv,noheader']),
           'script_sha256': PM.sha_file(os.path.abspath(__file__)),
           'models_sha256': PM.sha_file(os.path.join(HERE, 'models.py')),
           'env_vars': {k: os.environ.get(k) for k in (
               'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'CUDA_VISIBLE_DEVICES', 'PYTHONWARNINGS')}}
    if not smoke:
        bad = {k: (env[k], v) for k, v in fz['software'].items() if env[k] != v}
        assert not bad, f"the software differs from the frozen record: {bad}"

    ok, qlog = TD.quiet(enforce=not smoke)
    os.makedirs(out_dir, exist_ok=True)
    if not ok:
        json.dump({'stem': stem, 'started': False, 'reason': 'the machine was never quiet',
                   'quiet_log': qlog, 'state': PM.state(TD)},
                  open(os.path.join(out_dir, stem + '_notstarted.json'), 'w'), indent=1)
        print(f"{stem}: not started, machine never quiet", flush=True)
        sys.exit(3)

    apps0 = PM.compute_apps()
    log = PM.Logger(out[:-5] + '_gpu.csv')
    rec = {'stem': stem, 'mode': 'smoke' if smoke else 'session', 'slot': a.slot,
           'attempt': a.attempt, 'smoke': smoke, 'started': True, 'started_utc': PM.utc(),
           'frozen_sha256': fz_sha, 'frozen': fz, 'compute_apps_at_start': apps0,
           'protocol': {'weight_seed': PM.WEIGHT_SEED, 'frame_seed': PM.FRAME_SEED,
                        'ring': PM.RING, 'check_frames': PM.N_CHECK,
                        'image': [PM.IMG, PM.IMG, 3], 'mean': PM.MEAN, 'std': PM.STD,
                        'idle_s': phase_s[0], 'warmup_s': phase_s[1], 'window_s': phase_s[2],
                        'percentile': 'nearest-rank p95'},
           'runs': [], 'quiet_log': qlog, 'env': env}
    raw, failed = {}, None
    try:
        ring_np, check_np = PM.frames()
        ring = [torch.from_numpy(f).pin_memory() for f in ring_np]
        check = [torch.from_numpy(f).pin_memory() for f in check_np]
        vit32 = PM.network(torch)
        rec['weights_fp32_sha256'] = PM.state_sha(vit32)
        built = {}
        for name, ms in plan:
            w = {'build': name, 'spacing_ms': ms}
            rec['runs'].append(w)                  # kept, as far as it got, if a call fails
            if name not in built:
                for old_b in built.values():
                    old_b.free()
                torch.cuda.empty_cache()
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter('always')
                    t_init = time.time()
                    b = PM.Build(name, copy.deepcopy(vit32) if name == 'fp32' else vit32, torch,
                                 dev, ring)
                    w['init_s'] = time.time() - t_init
                w['build_warnings'] = sorted({f"{x.category.__name__}: {x.message}"[:500]
                                              for x in wl})
                built = {name: b}
            b = built[name]
            w['weights_sha256'] = b.weights_sha256
            if not smoke:
                assert b.weights_sha256 == fz['weights_sha256'][name], \
                    f"{name}'s weights differ from the frozen record"
            r = b.call()
            assert type(r) is int and 0 <= r < 64, (name, r)
            delta = int(ms * 1e6)
            w['idle_start_time'] = time.time()
            time.sleep(phase_s[0])
            w['idle_end_time'] = time.time()
            w['state_before'] = PM.state(TD)
            # one schedule for the warm-up and the window, so the pacing runs on across the
            # boundary; a call belongs to the phase in which it starts
            s, e, epoch = paced(b.call, phase_s[1] + phase_s[2], delta)
            w0 = int(phase_s[1] * 1e9)
            w['warmup_start_time'] = epoch
            w['window_start_time'] = epoch + phase_s[1]
            w['window_end_time'] = epoch + e[-1] / 1e9
            w['state_after'] = PM.state(TD)
            pre = f"{name}.{ms}." if smoke else ''
            raw[pre + 'start_ns'], raw[pre + 'end_ns'] = s, e
            raw[pre + 'window_from_ns'] = np.array(w0, dtype=np.int64)
            win = s >= w0
            w['warmup'] = summary(s[~win], e[~win], delta)
            w['window'] = summary(s[win], e[win], delta)
            if not win.any() or not (~win).any() or not np.all(e > s):
                raise RuntimeError('missing or invalid timing records')
            chk, _ = PM.execution_checks(b, check, None)
            w['execution_checks'] = chk
            print(f"  {stem} {name} at {ms} ms: {int(win.sum())} calls, mean "
                  f"{w['window']['mean_us']:.1f} us", flush=True)
            if not chk['passes']:
                raise RuntimeError('execution checks failed')
    except Exception as x:                         # an objective failure keeps its record
        failed = f"{type(x).__name__}: {x}"[:2000]
        rec['traceback'] = traceback.format_exc()[-4000:]
    finally:
        log.close()
    rec['ended_utc'] = PM.utc()
    rec['compute_apps_at_end'] = PM.compute_apps()
    seen = [x for w in rec['runs'] for k in ('state_before', 'state_after')
            for x in w.get(k, {}).get('compute_apps', [])] + rec['compute_apps_at_end']
    mine = str(os.getpid())
    rec['new_compute_apps'] = sorted({x for x in seen if x.split(',')[0].strip() != mine}
                                     - set(apps0))
    rec['failed'] = failed
    assert not os.path.exists(out), f"{out} appeared while timing"
    if raw:
        np.savez_compressed(out[:-5] + '.npz', **raw)
    json.dump(rec, open(out, 'w'), indent=1)
    print(f"wrote {out}{' (failed: ' + failed[:80] + ')' if failed else ''}", flush=True)
    sys.exit(4 if failed else 0)


if __name__ == '__main__':
    main()
