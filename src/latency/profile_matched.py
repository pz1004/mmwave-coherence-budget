#!/usr/bin/env python3
"""The matched implementation-profiling study: an image-only ViT-B/16 on the GPU, FP32 against one
optimized FP16 build.

Declared in protocol.json (analysis 'matched', rule R22) before it ran.  src/scheduler/matched.py
fixes the pilot's choices by R22's rules (--freeze) and reads the main sessions (the default).

WHY.  The paper's screen excludes ViT-B/16 on this GPU from FP32, batch-one timings of the
network and its argmax on an input already in device memory.  This study asks whether an
optimized half-precision build of the same graph removes that exclusion, keeps it or leaves it
unresolved, in expected effective rate (Proposition 1(d)), for a workload timed from the image to
the beam index on the host.  It is an implementation-sensitivity illustration, not a deployed
pipeline: the graph consumes one camera image and no beam report (K = 0), and the frame is
assumed to be in pinned host memory at the acquisition trigger, already 224 x 224 x 3 uint8
(capture, readout, decoding and resizing are left out, which favors the candidate).

WHAT.  One timed call: the frame's host-to-device copy into a preallocated device buffer;
conversion to the build's precision, division by 255, ImageNet mean/std normalization and
HWC -> NCHW; the forward pass; the argmax; the index copied to the host as a Python int, which
synchronizes.  time.perf_counter_ns around the call.  A ring of 64 distinct preallocated pinned
frames (numpy seed 881) is cycled, one per call.  One-time loading, compilation and graph capture
are excluded (a warm resident service).  The network is torchvision's vit_b_16(weights=None,
num_classes=64) with models.glorot_ under seed 17, built after torch.manual_seed(17) so that the
tensors glorot_ leaves (the position embedding, the attention input projections) are seeded too;
the FP16 builds convert these weights.  Builds:
  fp32          eager FP32, the historical runtime at this boundary (the comparator)
  fp16_eager    eager FP16                                                   (candidate O1)
  fp16_graph    FP16, the whole timed path captured in one CUDA graph        (candidate O2)
  fp16_compile  FP16, torch.compile of the path to the logits (Inductor, default mode,
                fullgraph, static shapes), then the argmax and the copy      (candidate O3)
Execution checks on 256 further frames (same generator): logits of shape (1, 64), all finite,
and the build's index equal to the argmax of its own logits; the largest FP16 - FP32 logit
difference and the argmax agreement with FP32 are recorded (execution checks, not accuracy).

Regime: sustained serialized service, back-to-back calls, batch one, one process with one CPU
thread (and one inter-op thread) pinned to CPUs 0-15 (run_profile_matched.sh), cudnn.benchmark on,
default clocks.  The GPU's state is logged every second by nvidia-smi in a separate process pinned
to CPUs 16-31 (utilization, clocks, temperature, power, performance state, clock-event reasons,
the encoder's utilization, sessions and frame rate); the machine's state (trace_decisive.state(),
plus the GPU's compute processes) is read before and after every build.

  --pilot      FP32 from idle, then fp16_eager, fp16_graph, fp16_compile and FP32 again, each for
               300 s back-to-back with every call kept; then the execution checks.
  --session k  the main session k (1-3): the build order of R22 (FP32 first in sessions 1 and 3);
               per build, the time-based warm-up W of the frozen record, then 10,000 timed calls
               and the recheck (250 and 2,500); the per-probe processing of the sweep
               (latency.py's probe_update, reproduced) timed first, 1,000 and 10,000 calls.
               Refuses to start unless protocol.json carries the analysis's frozen record.
  --smoke DIR  every build of the chosen mode for a few calls, into DIR outside this directory,
               timing nothing and without waiting for a quiet machine; its output is discarded.

A process waits for a quiet machine as trace_decisive does (exit 3 if it never comes).  A build
whose recheck drifts by more than 10%, or a compute process on the GPU that was not there when the
process started, makes the process contended (exit 2); a CUDA error writes a failure record
(exit 4).  run_profile_matched.sh repeats a session once, as attempt 2, on exit 2 or 4.  Outputs,
never overwritten: profile_matched/pilot.{json,npz} and s<k>[_a2].{json,npz}, each with its
nvidia-smi log <stem>_gpu.csv.

usage:  python3 profile_matched.py --pilot | --session 1|2|3 [--attempt 2] [--smoke DIR]
"""
import argparse, copy, hashlib, json, math, os, subprocess, sys, time, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'scheduler'))
OUT = os.path.join(HERE, 'profile_matched')
PROTOCOL = os.path.join(HERE, '..', 'scheduler', 'protocol.json')
KEY = 'matched'

AFFINITY = list(range(16))            # taskset -c 0-15: the eight P-cores, with SMT
LOGGER_CPUS = '16-31'                 # the E-cores, for the nvidia-smi logger
WEIGHT_SEED, FRAME_SEED = 17, 881
RING, N_CHECK, IMG = 64, 256, 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TIMED, RECHECK = 10_000, (250, 2_500)
PROBE_WARMUP, PROBE_TIMED = 1_000, 10_000
PILOT_S = 300
PILOT_ORDER = ('fp32', 'fp16_eager', 'fp16_graph', 'fp16_compile', 'fp32')
CANDIDATES = ('fp16_eager', 'fp16_graph', 'fp16_compile')
SESSION_ORDER = {1: ('fp32', 'opt'), 2: ('opt', 'fp32'), 3: ('fp32', 'opt')}
DRIFT_MAX = 0.10
SMOKE_CALLS = 5
LOG_FIELDS = ('timestamp,utilization.gpu,utilization.memory,utilization.encoder,clocks.sm,'
              'clocks.mem,temperature.gpu,power.draw,pstate,clocks_event_reasons.active,'
              'encoder.stats.sessionCount,encoder.stats.averageFps')


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:             # recorded, never fatal
        return f"unavailable: {e!r}"


def compute_apps():
    out = _run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'])
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def state(TD):
    s = TD.state()
    s['gpu_encoder'] = _run(['nvidia-smi', '--query-gpu=utilization.encoder,'
                             'encoder.stats.sessionCount,encoder.stats.averageFps',
                             '--format=csv,noheader,nounits'])
    s['compute_apps'] = compute_apps()
    s['time'] = time.time()
    return s


class Logger:
    """nvidia-smi sampled every second into a CSV, in a process pinned to the E-cores."""

    def __init__(self, path):
        self.f = open(path, 'w')
        self.p = subprocess.Popen(['taskset', '-c', LOGGER_CPUS, 'nvidia-smi',
                                   f'--query-gpu={LOG_FIELDS}', '--format=csv,noheader,nounits',
                                   '-lms', '1000'], stdout=self.f, stderr=subprocess.STDOUT)

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.f.close()


def frames():
    import numpy as np
    rng = np.random.default_rng(FRAME_SEED)
    ring = rng.integers(0, 256, size=(RING, IMG, IMG, 3), dtype=np.uint8)
    check = rng.integers(0, 256, size=(N_CHECK, IMG, IMG, 3), dtype=np.uint8)
    return ring, check


def network(torch):
    """ViT-B/16 as calibrate.build makes it, with the global RNG seeded as well."""
    import torchvision
    import models as M
    torch.manual_seed(WEIGHT_SEED)
    vit = torchvision.models.vit_b_16(weights=None, num_classes=64)
    M.glorot_(vit, WEIGHT_SEED)
    return vit.eval()


def state_sha(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Build:
    """One build: call() is the timed path; logits(frame) runs the same path to the logits."""

    def __init__(self, name, vit32, torch, dev, ring_pinned):
        self.name, self.torch, self.dev = name, torch, dev
        dt = torch.float32 if name == 'fp32' else torch.float16
        self.model = vit32 if name == 'fp32' else copy.deepcopy(vit32).half()
        self.model = self.model.to(dev).eval()
        self.weights_sha256 = state_sha(self.model)
        self.mean = torch.tensor(MEAN, device=dev, dtype=dt).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=dev, dtype=dt).view(1, 3, 1, 1)
        self.dt, self.ring, self.i = dt, ring_pinned, 0
        self.dbuf = torch.empty((IMG, IMG, 3), dtype=torch.uint8, device=dev)
        self.notes = {}
        model, mean, std = self.model, self.mean, self.std

        def to_logits(dbuf):
            x = dbuf.permute(2, 0, 1).unsqueeze(0).to(dt).contiguous()
            x = (x / 255.0 - mean) / std
            return model(x)

        if name in ('fp32', 'fp16_eager'):
            self._logits_dev = to_logits
            self.call = self._eager_call
        elif name == 'fp16_compile':
            import torch._dynamo
            self._logits_dev = torch.compile(to_logits, fullgraph=True, dynamic=False)
            self._logits_dev(self.dbuf)                 # compile once (excluded)
            self._logits_dev(self.dbuf)
            torch._dynamo.config.error_on_recompile = True
            self.call = self._eager_call
        elif name == 'fp16_graph':
            self.stage = torch.empty((IMG, IMG, 3), dtype=torch.uint8).pin_memory()
            self.out_host = torch.empty((), dtype=torch.int64).pin_memory()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):                  # warm-up before capture, as required
                for _ in range(3):
                    self.dbuf.copy_(self.stage, non_blocking=True)
                    torch.argmax(to_logits(self.dbuf)[0])
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.dbuf.copy_(self.stage, non_blocking=True)
                self.static_logits = to_logits(self.dbuf)
                idx = torch.argmax(self.static_logits[0])
                self.out_host.copy_(idx, non_blocking=True)
            torch.cuda.synchronize()
            self.call = self._graph_call
        else:
            raise ValueError(name)

    def _next(self):
        f = self.ring[self.i]
        self.i = (self.i + 1) % RING
        return f

    def _eager_call(self):
        self.dbuf.copy_(self._next(), non_blocking=True)
        return int(self.torch.argmax(self._logits_dev(self.dbuf)[0]))

    def _graph_call(self):
        self.stage.copy_(self._next())                 # host copy into the captured source
        self.graph.replay()
        self.torch.cuda.current_stream().synchronize()
        return int(self.out_host)

    def check(self, frame_pinned):
        """(index from the timed path, logits on the CPU) for one check frame."""
        torch = self.torch
        if self.name == 'fp16_graph':
            self.stage.copy_(frame_pinned)
            self.graph.replay()
            torch.cuda.current_stream().synchronize()
            return int(self.out_host), self.static_logits.float().cpu().clone()
        self.dbuf.copy_(frame_pinned, non_blocking=True)
        y = self._logits_dev(self.dbuf)
        return int(torch.argmax(y[0])), y.float().cpu().clone()

    def free(self):
        for a in ('model', 'graph', 'static_logits', '_logits_dev', 'dbuf', 'stage', 'out_host'):
            if hasattr(self, a):
                delattr(self, a)


def execution_checks(b, check_pinned, ref_logits):
    """Shape, finiteness and self-consistency; FP32 agreement and logit difference recorded."""
    import numpy as np
    ok_shape, ok_finite, ok_self, agree, maxdiff, logits = True, True, True, 0, 0.0, []
    for k, fr in enumerate(check_pinned):
        idx, y = b.check(fr)
        ok_shape &= tuple(y.shape) == (1, 64)
        ok_finite &= bool(np.isfinite(y.numpy()).all())
        ok_self &= idx == int(y[0].argmax())
        logits.append(y[0].numpy())
        if ref_logits is not None:
            agree += int(idx == int(np.argmax(ref_logits[k])))
            maxdiff = max(maxdiff, float(np.abs(y[0].numpy() - ref_logits[k]).max()))
    out = {'n_frames': len(check_pinned), 'shape_ok': bool(ok_shape),
           'finite_ok': bool(ok_finite), 'self_consistent': bool(ok_self),
           'passes': bool(ok_shape and ok_finite and ok_self)}
    if ref_logits is not None:
        out.update({'argmax_agreement_with_fp32': agree / len(check_pinned),
                    'max_abs_logit_difference_vs_fp32': maxdiff})
    return out, np.stack(logits)


def time_window(fn, seconds=None, n=None):
    """Back-to-back calls, every one timed; by duration or by count.  Returns the call times and
    each call's end, both in ns, the end relative to the window's start."""
    import numpy as np
    pc = time.perf_counter_ns
    cap = n if n is not None else 2_000_000
    d = np.empty(cap, dtype=np.int64)
    e = np.empty(cap, dtype=np.int64)
    t_begin = pc()
    stop = t_begin + int(seconds * 1e9) if seconds is not None else None
    i = 0
    while i < cap:
        t0 = pc()
        fn()
        t1 = pc()
        d[i], e[i] = t1 - t0, t1 - t_begin
        i += 1
        if stop is not None and t1 >= stop:
            break
    return d[:i].copy(), e[:i].copy()


def probe_processing(np):
    """latency.py's probe_update (lines 109-115), reproduced with the same state (seed 881)."""
    import obs
    from calibrate import nearest_rank_p95, timed
    rng = np.random.default_rng(881)
    y = (10 ** (rng.uniform(-4, 5, 64) / 10)) * obs.SIGMA_D2
    meas = np.zeros(64, bool)
    meas[obs.spread_order()[:16]] = True
    sdb = np.clip(10 * np.log10(np.maximum(obs.P_TX * y, 1e-30) / obs.SIGMA_D2), -40.0, 60.0)
    fbuf = np.zeros(194, np.float32)
    fbuf[:64] = np.where(meas, sdb, 0.0) / 20.0
    fbuf[64:128] = meas
    st = [int(np.argmax(np.where(meas, sdb, -np.inf))), 16, 0]
    spread = obs.spread_order()
    acc = [0.0]

    def probe_update():
        c = spread[st[1] % 64]
        v = 10.0 * math.log10(max(obs.P_TX * float(y[c]), 1e-30) / obs.SIGMA_D2)
        fbuf[c] = v * 0.05
        fbuf[64 + c] = 1.0
        acc[0] += v
        return v > sdb[st[0]]

    x = timed(probe_update, PROBE_WARMUP, PROBE_TIMED)
    return {'n_warmup': PROBE_WARMUP, 'n_timed': PROBE_TIMED,
            'p95_us': nearest_rank_p95(x) / 1e3, 'median_us': float(np.median(x)) / 1e3}


def frozen_record():
    P = json.load(open(PROTOCOL))
    A = [a for a in P['analyses'] if a['key'] == KEY]
    assert len(A) == 1, f"protocol.json declares analysis '{KEY}' exactly once"
    return A[0], hashlib.sha256(json.dumps(A[0], sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--pilot', action='store_true')
    g.add_argument('--session', type=int, choices=(1, 2, 3))
    ap.add_argument('--attempt', type=int, default=1, choices=(1, 2))
    ap.add_argument('--smoke', default='')
    a = ap.parse_args()
    smoke = bool(a.smoke)
    out_dir = os.path.abspath(a.smoke) if smoke else OUT
    if smoke:
        assert not out_dir.startswith(os.path.abspath(OUT)), "a smoke test never writes the outputs"
    stem = 'pilot' if a.pilot else f"s{a.session}" + ('_a2' if a.attempt == 2 else '')
    if a.pilot:
        assert a.attempt == 1, "the pilot has one attempt"
    out = os.path.join(out_dir, stem + '.json')
    for p in (out, out[:-5] + '.npz', out[:-5] + '_gpu.csv'):
        assert not os.path.exists(p), f"{p} exists: an output is written once"

    # gates that need no torch
    import trace_decisive as TD
    assert sha_file(os.path.join(HERE, 'models.py')) == TD.SRC_SHA256['models.py'], \
        "models.py changed"
    assert sorted(os.sched_getaffinity(0)) == AFFINITY, "run under taskset -c 0-15"
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(v) == '1', f"{v} must be 1"
    entry, entry_sha = frozen_record() if not smoke else ({}, None)
    if a.session and not smoke:
        fz = entry.get('frozen')
        assert fz and fz.get('build') in CANDIDATES and fz.get('warmup_s'), \
            "the pilot's choices are not frozen in protocol.json: run matched.py --freeze first"
        opt, warmup_s = fz['build'], fz['warmup_s']
        builds = tuple(opt if s == 'opt' else s for s in SESSION_ORDER[a.session])
    elif a.session:                                 # a smoke test of a session: every build
        builds, warmup_s = ('fp32',) + CANDIDATES, 0
    else:
        builds = PILOT_ORDER

    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    import numpy as np
    import calibrate                               # silences warnings, as run_all.sh did
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Quadro RTX 8000'
    dev = torch.device('cuda')

    ok, qlog = TD.quiet(enforce=not smoke)
    os.makedirs(out_dir, exist_ok=True)
    if not ok:
        json.dump({'stem': stem, 'started': False, 'reason': 'the machine was never quiet',
                   'quiet_log': qlog, 'state': state(TD)},
                  open(os.path.join(out_dir, stem + '_notstarted.json'), 'w'), indent=1)
        print(f"{stem}: not started, machine never quiet", flush=True)
        sys.exit(3)

    apps0 = compute_apps()
    log = Logger(out[:-5] + '_gpu.csv')
    rec = {'stem': stem, 'mode': 'pilot' if a.pilot else 'session', 'session': a.session,
           'attempt': a.attempt, 'smoke': smoke, 'started': True, 'started_utc': utc(),
           'declaration_sha256': entry_sha, 'frozen': entry.get('frozen'),
           'builds_in_order': list(builds), 'compute_apps_at_start': apps0,
           'protocol': {'weight_seed': WEIGHT_SEED, 'frame_seed': FRAME_SEED, 'ring': RING,
                        'check_frames': N_CHECK, 'image': [IMG, IMG, 3], 'mean': MEAN,
                        'std': STD, 'timed': TIMED, 'recheck': list(RECHECK),
                        'pilot_s': PILOT_S, 'drift_max': DRIFT_MAX,
                        'percentile': 'nearest-rank p95'},
           'windows': [], 'quiet_log': qlog}
    raw, failed = {}, None
    try:
        ring_np, check_np = frames()
        ring = [torch.from_numpy(f).pin_memory() for f in ring_np]
        check = [torch.from_numpy(f).pin_memory() for f in check_np]
        if a.session and not smoke:
            rec['probe_processing'] = probe_processing(np)
        vit32 = network(torch)
        rec['weights_fp32_sha256'] = state_sha(vit32)
        ref_logits = None
        for j, name in enumerate(builds):
            w = {'build': name, 'position': j}
            with warnings.catch_warnings(record=True) as wl:      # those raised while building
                warnings.simplefilter('always')
                t_init = time.time()
                try:
                    b = Build(name, vit32 if name != 'fp32' else copy.deepcopy(vit32), torch,
                              dev, ring)
                    w.update({'built': True, 'init_s': time.time() - t_init,
                              'weights_sha256': b.weights_sha256})
                except Exception as e:            # a candidate that does not build
                    if not (a.pilot and name in CANDIDATES):
                        raise
                    w.update({'built': False, 'error': f"{type(e).__name__}: {e}"[:2000]})
                    b = None
            w['build_warnings'] = sorted({f"{x.category.__name__}: {x.message}"[:500] for x in wl})
            if b is not None:
                r = b.call()
                assert type(r) is int and 0 <= r < 64, (name, r)
                w['state_before'] = state(TD)
                if smoke:
                    for _ in range(SMOKE_CALLS):
                        b.call()
                elif a.pilot:
                    w['window_start_time'] = time.time()
                    try:
                        d, e = time_window(b.call, seconds=PILOT_S)
                        w['runs'] = True
                    except Exception as x:        # a recompilation or a failed replay
                        if name not in CANDIDATES:
                            raise
                        w.update({'runs': False, 'error': f"{type(x).__name__}: {x}"[:2000]})
                        d = e = np.empty(0, dtype=np.int64)
                    w['window_end_time'] = time.time()
                    raw[f"{j}.{name}.ns"], raw[f"{j}.{name}.end_ns"] = d, e
                    w['n_calls'] = int(d.size)
                else:
                    w['warmup_start_time'] = time.time()
                    dw, _ = time_window(b.call, seconds=warmup_s)
                    w['warmup_s'], w['warmup_calls'] = warmup_s, int(dw.size)
                    w['window_start_time'] = time.time()
                    d, e = time_window(b.call, n=TIMED)
                    time_window(b.call, n=RECHECK[0])
                    y, _ = time_window(b.call, n=RECHECK[1])
                    w['window_end_time'] = time.time()
                    raw[f"{name}.ns"], raw[f"{name}.end_ns"] = d, e
                    raw[f"{name}.recheck.ns"] = y
                    p95, rp95 = calibrate.nearest_rank_p95(d), calibrate.nearest_rank_p95(y)
                    w.update({'n_timed': int(d.size), 'recheck_n_warmup': RECHECK[0],
                              'recheck_n_timed': int(y.size), 'p95_ns': p95,
                              'recheck_p95_ns': rp95, 'drift': abs(rp95 - p95) / p95})
                w['state_after'] = state(TD)
                chk, lg = execution_checks(b, check, ref_logits)
                w['execution_checks'] = chk
                if name == 'fp32' and ref_logits is None:
                    ref_logits = lg
                if name == 'fp16_compile':
                    import torch._dynamo
                    w['dynamo_counters'] = {k: dict(v) for k, v in
                                            torch._dynamo.utils.counters.items()}
                    torch._dynamo.config.error_on_recompile = False
                    torch._dynamo.reset()
                b.free()
                del b
            torch.cuda.empty_cache()
            rec['windows'].append(w)
            print(f"  {stem} {name}: {'done' if w['built'] else 'does not build'}", flush=True)
    except RuntimeError as e:                      # torch's CUDA errors are RuntimeErrors
        if not any(t in str(e) for t in ('CUDA', 'cuda', 'cuDNN', 'CUBLAS', 'cublas')):
            raise
        failed = f"{type(e).__name__}: {e}"[:2000]
    finally:
        log.close()
    rec['ended_utc'] = utc()
    rec['compute_apps_at_end'] = compute_apps()
    new_apps = sorted({x for w in rec['windows'] for k in ('state_before', 'state_after')
                       for x in w.get(k, {}).get('compute_apps', [])
                       if x.split(',')[0].strip() != str(os.getpid())} - set(apps0))
    rec['new_compute_apps'] = new_apps
    rec['contended'] = (not a.pilot and not smoke and failed is None and
                        (any(w.get('drift', 0) > DRIFT_MAX for w in rec['windows'])
                         or bool(new_apps)))
    rec['failed'] = failed
    rec['env'] = {**calibrate.env_record('gpu-resident', 1),
                  'torchvision': __import__('torchvision').__version__,
                  'triton': __import__('triton').__version__,
                  'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                  'driver': _run(['nvidia-smi', '--query-gpu=driver_version',
                                  '--format=csv,noheader']),
                  'script_sha256': sha_file(os.path.abspath(__file__)),
                  'models_sha256': sha_file(os.path.join(HERE, 'models.py')),
                  'env_vars': {k: os.environ.get(k) for k in (
                      'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                      'CUDA_VISIBLE_DEVICES', 'PYTHONWARNINGS')}}
    assert not os.path.exists(out), f"{out} appeared while timing"
    if raw:
        np.savez_compressed(out[:-5] + '.npz', **raw)
    json.dump(rec, open(out, 'w'), indent=1)
    print(f"wrote {out}{' (contended)' if rec['contended'] else ''}"
          f"{' (failed: ' + failed[:80] + ')' if failed else ''}", flush=True)
    sys.exit(4 if failed else 2 if rec['contended'] else 0)


if __name__ == '__main__':
    main()
