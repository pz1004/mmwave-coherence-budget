#!/usr/bin/env python3
"""The deployment-runtime study: compiled reduced-precision builds of the two vision models the
screen excludes, each against its FP32 build, timed as the matched study (profile_matched.py) times
ViT-B/16.

Declared in protocol.json (analysis 'deploy', rule R38) before it ran.  src/scheduler/deploy.py
fixes the pilot's choices by R38's rules (--freeze) and reads the main sessions (the default).
profile_matched.py is imported unchanged for its helpers (machine state, logger, frames, timing
windows, execution checks, the sweep's per-probe processing).

WHY.  The screen excludes ResNet-18 on eight P-core threads and ViT-B/16 on the GPU from FP32,
batch-one timings; one FP16 build of ViT-B/16 (rule R22) was 3.2 times faster.  This study asks
whether builds a deployment would use, with the runtime installed here (PyTorch 2.12; TensorRT,
ONNX Runtime and OpenVINO are not installed and are not added), keep or remove the exclusions.

WHAT.  The image-only workload of the matched study, timed from the frame to the beam index on the
host: on the GPU, the frame's copy from pinned host memory, conversion, normalization, the forward
pass, the argmax and the index's copy to the host; on the CPU, the frame from host memory,
conversion, normalization, the forward pass and the argmax.  A ring of 64 frames (numpy seed 881),
one per call.  One-time loading, compilation, quantization and graph capture are excluded (a warm
resident service).  Networks: torchvision's resnet18 and vit_b_16 (weights=None, num_classes=64)
with models.glorot_ under seed 17, built after torch.manual_seed(17).  Groups and builds:
  rn_gpu   ResNet-18, GPU, one CPU thread:  fp32 (comparator); fp16_eager; fp16_graph (the whole
           timed path in one CUDA graph); fp16_cmax (torch.compile of the path to the logits,
           Inductor mode 'max-autotune', fullgraph, static shapes)
  vit_gpu  ViT-B/16, GPU, one CPU thread:   fp32 (comparator); fp16_graph (rule R22's frozen
           build); fp16_cmax
  rn_cpu   ResNet-18, CPU, eight threads:   fp32 (comparator); cpu_compile (torch.compile,
           Inductor's default mode, fullgraph, static shapes); int8 (eager-mode post-training
           static quantization, torchvision's quantizable ResNet-18 with the same weights, fused,
           the x86 engine's default qconfig, calibrated on the 64 ring frames)
Execution checks on 256 further frames: logits of shape (1, 64), finite, and the build's index
the argmax of its own logits; agreement with FP32's argmax and the largest logit difference are
recorded (execution checks, not accuracy).

Regime: sustained serialized service, back-to-back calls, batch one, one process per group
(run_profile_deploy.sh: taskset -c 0-15; the GPU groups with one thread, OMP_NUM_THREADS = 1; the
CPU group with eight intra-op threads, OMP_NUM_THREADS = 8, one inter-op thread), cudnn.benchmark
on, default clocks, the governor as found.  The GPU's state is logged every second (nvidia-smi, on
the E-cores); the machine's state is read before and after every build.

  --pilot G      FP32 from idle, the candidates, FP32 again, each for 300 s back to back, every
                 call kept; then the execution checks
  --session k G  the main session k (1-3) of group G: FP32 first in sessions 1 and 3; per build the
                 frozen warm-up, then 10,000 timed calls and the recheck (250 and 2,500); in group
                 rn_gpu the sweep's per-probe processing first (1,000 and 10,000 calls).  Refuses
                 to start unless protocol.json carries the analysis's frozen record
  --smoke DIR    every build for a few calls into DIR outside the outputs, timing nothing

A process waits for a quiet machine (trace_decisive.quiet; exit 3 if it never comes).  A build whose
recheck drifts by more than 10%, or a compute process on the GPU that was not there at the start,
makes a session contended (exit 2); a CUDA error writes a failure record (exit 4).  Outputs, never
overwritten: profile_deploy/pilot_<G>.{json,npz} and s<k>_<G>[_a2].{json,npz}, each with
<stem>_gpu.csv.

usage:  python3 profile_deploy.py --pilot G | --session k G [--attempt 2] [--smoke DIR]
"""
import argparse, copy, hashlib, json, os, sys, time, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'scheduler'))
import profile_matched as PM

OUT = os.path.join(HERE, 'profile_deploy')
PROTOCOL = os.path.join(HERE, '..', 'scheduler', 'protocol.json')
KEY = 'deploy'
AFFINITY = list(range(16))
GROUPS = {'rn_gpu': ('resnet18', 'cuda', 1, ('fp16_eager', 'fp16_graph', 'fp16_cmax')),
          'vit_gpu': ('vit_b_16', 'cuda', 1, ('fp16_graph', 'fp16_cmax')),
          'rn_cpu': ('resnet18', 'cpu', 8, ('cpu_compile', 'int8'))}
SESSION_ORDER = {1: ('fp32', 'opt'), 2: ('opt', 'fp32'), 3: ('fp32', 'opt')}
TIMED, RECHECK, PILOT_S, DRIFT_MAX, SMOKE_CALLS = PM.TIMED, PM.RECHECK, PM.PILOT_S, PM.DRIFT_MAX, 5
CALIB_FRAMES = 64


def network(torch, arch):
    import torchvision
    import models as M
    torch.manual_seed(PM.WEIGHT_SEED)
    net = getattr(torchvision.models, arch)(weights=None, num_classes=64)
    M.glorot_(net, PM.WEIGHT_SEED)
    return net.eval()


def int8_resnet(torch, net32, ring):
    """torchvision's quantizable ResNet-18 with net32's weights, fused, calibrated, converted."""
    import torchvision
    from torch.ao import quantization as tq
    torch.backends.quantized.engine = 'x86'
    q = torchvision.models.quantization.resnet18(weights=None, quantize=False, num_classes=64)
    q.load_state_dict(net32.state_dict(), strict=True)
    q.eval()
    q.fuse_model(is_qat=False)
    q.qconfig = tq.get_default_qconfig('x86')
    tq.prepare(q, inplace=True)
    mean = torch.tensor(PM.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(PM.STD).view(1, 3, 1, 1)
    with torch.no_grad():
        for f in ring[:CALIB_FRAMES]:
            x = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
            q((x / 255.0 - mean) / std)
    tq.convert(q, inplace=True)
    return q


class Build:
    """One build: call() is the timed path; check(frame) returns (index, logits on the CPU)."""

    def __init__(self, name, arch, device, net32, torch, ring_np, ring_pinned):
        self.name, self.torch, self.i = name, torch, 0
        self.notes = {}
        if device == 'cuda':
            self._gpu(name, net32, torch, ring_pinned)
        else:
            self._cpu(name, net32, torch, ring_np)

    # ---- GPU builds: profile_matched.Build's paths, for either network ----
    def _gpu(self, name, net32, torch, ring_pinned):
        dev = torch.device('cuda')
        dt = torch.float32 if name == 'fp32' else torch.float16
        self.model = (net32 if name == 'fp32' else copy.deepcopy(net32).half()).to(dev).eval()
        self.weights_sha256 = PM.state_sha(self.model)
        # attributes, not locals: a captured graph reads them by address for the build's lifetime
        self.mean = torch.tensor(PM.MEAN, device=dev, dtype=dt).view(1, 3, 1, 1)
        self.std = torch.tensor(PM.STD, device=dev, dtype=dt).view(1, 3, 1, 1)
        self.ring = ring_pinned
        self.dbuf = torch.empty((PM.IMG, PM.IMG, 3), dtype=torch.uint8, device=dev)
        model, mean, std = self.model, self.mean, self.std

        def to_logits(dbuf):
            x = dbuf.permute(2, 0, 1).unsqueeze(0).to(dt).contiguous()
            return model((x / 255.0 - mean) / std)

        self.kind = 'gpu_eager'
        if name in ('fp32', 'fp16_eager'):
            self._logits = to_logits
        elif name == 'fp16_cmax':
            import torch._dynamo
            self._logits = torch.compile(to_logits, mode='max-autotune', fullgraph=True,
                                         dynamic=False)
            for _ in range(3):                           # compile and record (excluded)
                self._logits(self.dbuf)
            torch.cuda.synchronize()
            torch._dynamo.config.error_on_recompile = True
        elif name == 'fp16_graph':
            self.kind = 'gpu_graph'
            self.stage = torch.empty((PM.IMG, PM.IMG, 3), dtype=torch.uint8).pin_memory()
            self.out_host = torch.empty((), dtype=torch.int64).pin_memory()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
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
        else:
            raise ValueError(name)

    # ---- CPU builds ----
    def _cpu(self, name, net32, torch, ring_np):
        mean = torch.tensor(PM.MEAN).view(1, 3, 1, 1)
        std = torch.tensor(PM.STD).view(1, 3, 1, 1)
        self.ring = [torch.from_numpy(f) for f in ring_np]
        if name == 'int8':
            self.model = int8_resnet(torch, net32, ring_np)
        else:
            self.model = net32
        self.weights_sha256 = PM.state_sha(net32)
        model = self.model

        def to_logits(frame):
            x = frame.permute(2, 0, 1).unsqueeze(0).float().contiguous()
            return model((x / 255.0 - mean) / std)

        self.kind = 'cpu'
        if name in ('fp32', 'int8'):
            self._logits = to_logits
        elif name == 'cpu_compile':
            import torch._dynamo
            self._logits = torch.compile(to_logits, fullgraph=True, dynamic=False)
            for _ in range(3):
                self._logits(self.ring[0])
            torch._dynamo.config.error_on_recompile = True
        else:
            raise ValueError(name)

    def _next(self):
        f = self.ring[self.i]
        self.i = (self.i + 1) % PM.RING
        return f

    def call(self):
        torch = self.torch
        if self.kind == 'gpu_graph':
            self.stage.copy_(self._next())
            self.graph.replay()
            torch.cuda.current_stream().synchronize()
            return int(self.out_host)
        if self.kind == 'gpu_eager':
            self.dbuf.copy_(self._next(), non_blocking=True)
            return int(torch.argmax(self._logits(self.dbuf)[0]))
        return int(torch.argmax(self._logits(self._next())[0]))

    def check(self, frame):
        torch = self.torch
        if self.kind == 'gpu_graph':
            self.stage.copy_(frame)
            self.graph.replay()
            torch.cuda.current_stream().synchronize()
            return int(self.out_host), self.static_logits.float().cpu().clone()
        if self.kind == 'gpu_eager':
            self.dbuf.copy_(frame, non_blocking=True)
            y = self._logits(self.dbuf)
        else:
            y = self._logits(frame)
        return int(torch.argmax(y[0])), y.float().cpu().clone()

    def free(self):
        for a in ('model', 'graph', 'static_logits', '_logits', 'dbuf', 'stage', 'out_host',
                  'mean', 'std'):
            if hasattr(self, a):
                delattr(self, a)


def frozen_record():
    P = json.load(open(PROTOCOL))
    A = [a for a in P['analyses'] if a['key'] == KEY]
    assert len(A) == 1, f"protocol.json declares analysis '{KEY}' exactly once"
    return A[0], hashlib.sha256(json.dumps(A[0], sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--pilot', choices=sorted(GROUPS))
    g.add_argument('--session', nargs=2, metavar=('K', 'GROUP'))
    ap.add_argument('--attempt', type=int, default=1, choices=(1, 2))
    ap.add_argument('--smoke', default='')
    a = ap.parse_args()
    smoke = bool(a.smoke)
    if a.session:
        k, group = int(a.session[0]), a.session[1]
        assert k in (1, 2, 3) and group in GROUPS
    else:
        k, group = None, a.pilot
    arch, device, threads, cands = GROUPS[group]
    out_dir = os.path.abspath(a.smoke) if smoke else OUT
    if smoke:
        assert not out_dir.startswith(os.path.abspath(OUT)), "a smoke test never writes the outputs"
    stem = (f"pilot_{group}" if k is None else f"s{k}_{group}" + ('_a2' if a.attempt == 2 else ''))
    if k is None:
        assert a.attempt == 1, "the pilot has one attempt"
    out = os.path.join(out_dir, stem + '.json')
    for p in (out, out[:-5] + '.npz', out[:-5] + '_gpu.csv'):
        assert not os.path.exists(p), f"{p} exists: an output is written once"

    import trace_decisive as TD
    assert PM.sha_file(os.path.join(HERE, 'models.py')) == TD.SRC_SHA256['models.py'], \
        "models.py changed"
    assert sorted(os.sched_getaffinity(0)) == AFFINITY, "run under taskset -c 0-15"
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(v) == str(threads), f"{v} must be {threads}"
    entry, entry_sha = frozen_record() if not smoke else ({}, None)
    if k is not None and not smoke:
        fz = (entry.get('frozen') or {}).get(group)
        assert fz and fz.get('build') in cands and fz.get('warmup_s') is not None, \
            "the pilot's choices are not frozen in protocol.json: run deploy.py --freeze first"
        opt, warmup_s = fz['build'], fz['warmup_s']
        builds = tuple(opt if s == 'opt' else s for s in SESSION_ORDER[k])
    elif k is not None:
        builds, warmup_s = ('fp32',) + cands, 0
    else:
        builds = ('fp32',) + cands + ('fp32',)

    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = True
    import numpy as np
    import calibrate
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'Quadro RTX 8000'

    ok, qlog = TD.quiet(enforce=not smoke)
    os.makedirs(out_dir, exist_ok=True)
    if not ok and not smoke:
        json.dump({'stem': stem, 'started': False, 'reason': 'the machine was never quiet',
                   'quiet_log': qlog, 'state': PM.state(TD)},
                  open(os.path.join(out_dir, stem + '_notstarted.json'), 'w'), indent=1)
        print(f"{stem}: not started, machine never quiet", flush=True)
        sys.exit(3)

    apps0 = PM.compute_apps()
    log = PM.Logger(out[:-5] + '_gpu.csv')
    rec = {'stem': stem, 'group': group, 'arch': arch, 'device': device, 'threads': threads,
           'mode': 'pilot' if k is None else 'session', 'session': k, 'attempt': a.attempt,
           'smoke': smoke, 'started': True, 'started_utc': PM.utc(),
           'declaration_sha256': entry_sha, 'frozen': (entry.get('frozen') or {}).get(group),
           'builds_in_order': list(builds), 'compute_apps_at_start': apps0,
           'protocol': {'weight_seed': PM.WEIGHT_SEED, 'frame_seed': PM.FRAME_SEED,
                        'ring': PM.RING, 'check_frames': PM.N_CHECK,
                        'image': [PM.IMG, PM.IMG, 3], 'mean': PM.MEAN, 'std': PM.STD,
                        'timed': TIMED, 'recheck': list(RECHECK), 'pilot_s': PILOT_S,
                        'drift_max': DRIFT_MAX, 'calibration_frames': CALIB_FRAMES,
                        'percentile': 'nearest-rank p95'},
           'windows': [], 'quiet_log': qlog}
    raw, failed = {}, None
    try:
        ring_np, check_np = PM.frames()
        ring = [torch.from_numpy(f).pin_memory() for f in ring_np] if device == 'cuda' else None
        check = ([torch.from_numpy(f).pin_memory() for f in check_np] if device == 'cuda'
                 else [torch.from_numpy(f) for f in check_np])
        if k is not None and not smoke and group == 'rn_gpu':
            rec['probe_processing'] = PM.probe_processing(np)
        net32 = network(torch, arch)
        rec['weights_fp32_sha256'] = PM.state_sha(net32)
        ref_logits = None
        for j, name in enumerate(builds):
            w = {'build': name, 'position': j}
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                t_init = time.time()
                try:
                    b = Build(name, arch, device, copy.deepcopy(net32), torch, ring_np, ring)
                    w.update({'built': True, 'init_s': time.time() - t_init,
                              'weights_sha256': b.weights_sha256})
                except Exception as e:
                    if not (k is None and name in cands):
                        raise
                    w.update({'built': False, 'error': f"{type(e).__name__}: {e}"[:2000]})
                    b = None
            w['build_warnings'] = sorted({f"{x.category.__name__}: {x.message}"[:500] for x in wl})
            if b is not None:
                r = b.call()
                assert type(r) is int and 0 <= r < 64, (name, r)
                w['state_before'] = PM.state(TD)
                if smoke:
                    for _ in range(SMOKE_CALLS):
                        b.call()
                elif k is None:
                    w['window_start_time'] = time.time()
                    try:
                        d, e = PM.time_window(b.call, seconds=PILOT_S)
                        w['runs'] = True
                    except Exception as x:
                        if name not in cands:
                            raise
                        w.update({'runs': False, 'error': f"{type(x).__name__}: {x}"[:2000]})
                        d = e = np.empty(0, dtype=np.int64)
                    w['window_end_time'] = time.time()
                    raw[f"{j}.{name}.ns"], raw[f"{j}.{name}.end_ns"] = d, e
                    w['n_calls'] = int(d.size)
                else:
                    w['warmup_start_time'] = time.time()
                    dw, _ = PM.time_window(b.call, seconds=warmup_s)
                    w['warmup_s'], w['warmup_calls'] = warmup_s, int(dw.size)
                    w['window_start_time'] = time.time()
                    d, e = PM.time_window(b.call, n=TIMED)
                    PM.time_window(b.call, n=RECHECK[0])
                    y, _ = PM.time_window(b.call, n=RECHECK[1])
                    w['window_end_time'] = time.time()
                    raw[f"{name}.ns"], raw[f"{name}.end_ns"] = d, e
                    raw[f"{name}.recheck.ns"] = y
                    p95, rp95 = calibrate.nearest_rank_p95(d), calibrate.nearest_rank_p95(y)
                    w.update({'n_timed': int(d.size), 'recheck_n_warmup': RECHECK[0],
                              'recheck_n_timed': int(y.size), 'p95_ns': p95,
                              'recheck_p95_ns': rp95, 'drift': abs(rp95 - p95) / p95})
                w['state_after'] = PM.state(TD)
                chk, lg = PM.execution_checks(b, check, ref_logits)
                w['execution_checks'] = chk
                if name == 'fp32' and ref_logits is None:
                    ref_logits = lg
                if name in ('fp16_cmax', 'cpu_compile'):
                    import torch._dynamo
                    w['dynamo_counters'] = {kk: dict(v) for kk, v in
                                            torch._dynamo.utils.counters.items()}
                    torch._dynamo.config.error_on_recompile = False
                    torch._dynamo.reset()
                b.free()
                del b
            if device == 'cuda':
                torch.cuda.empty_cache()
            rec['windows'].append(w)
            print(f"  {stem} {name}: {'done' if w['built'] else 'does not build'}", flush=True)
    except RuntimeError as e:
        if not any(t in str(e) for t in ('CUDA', 'cuda', 'cuDNN', 'CUBLAS', 'cublas')):
            raise
        failed = f"{type(e).__name__}: {e}"[:2000]
    finally:
        log.close()
    rec['ended_utc'] = PM.utc()
    rec['compute_apps_at_end'] = PM.compute_apps()
    new_apps = sorted({x for w in rec['windows'] for kk in ('state_before', 'state_after')
                       for x in w.get(kk, {}).get('compute_apps', [])
                       if x.split(',')[0].strip() != str(os.getpid())} - set(apps0))
    rec['new_compute_apps'] = new_apps
    rec['contended'] = (k is not None and not smoke and failed is None and
                        (any(w.get('drift', 0) > DRIFT_MAX for w in rec['windows'])
                         or bool(new_apps)))
    rec['failed'] = failed
    rec['env'] = {**calibrate.env_record(f'deploy-{group}', threads),
                  'torchvision': __import__('torchvision').__version__,
                  'triton': __import__('triton').__version__,
                  'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                  'quantized_engine': torch.backends.quantized.engine,
                  'driver': PM._run(['nvidia-smi', '--query-gpu=driver_version',
                                     '--format=csv,noheader']),
                  'script_sha256': PM.sha_file(os.path.abspath(__file__)),
                  'profile_matched_sha256': PM.sha_file(os.path.join(HERE, 'profile_matched.py')),
                  'models_sha256': PM.sha_file(os.path.join(HERE, 'models.py')),
                  'env_vars': {v: os.environ.get(v) for v in (
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
