#!/usr/bin/env python3
"""The decision audit's timing campaign: three image-only GPU builds, each invoked every 10 ms, in
fifteen randomized rounds.

Declared in protocol.json (analysis 'decision_audit', rule R26) before any of its sessions ran.
src/scheduler/audit_screen.py writes the session order into the analysis's frozen record
(--freeze), makes the screening decisions from rounds 1-5 (--decide) and records their hash
before round 6 may start; src/scheduler/audit_direct.py reads every session.

WHY.  The audit asks how often the screen, calibrated on part of a city, excludes implementations
that a 10 ms feasibility check keeps, and whether the exclusions hold against a direct calculation
of a perfect predictor's rate in the same finite population and in later timing sessions.  Rounds
1-5 are the calibration timing record from which the screen decides; rounds 6-15 are the later
sessions on which the frozen decisions are evaluated.

WHAT.  profile_matched.py's timed call, frames, threads and pinning: the frame (224 x 224 x 3
uint8) waits in pinned host memory at the trigger; the call copies it to the device, converts and
normalizes it, runs the network and the argmax, and returns the beam index to the host as a Python
int, which synchronizes.  Builds, image-only workloads with no beam report (K = 0):
  resnet18_ts     ResNet-18, FP32, the path from the device frame to the index traced and frozen
                  by TorchScript (calibrate_opt2.frozen: trace, freeze, optimize_for_inference)
  vit_fp16_graph  ViT-B/16, profile_matched's fp16_graph: the whole call in one CUDA graph
  vit_fp32        ViT-B/16, profile_matched's fp32: eager FP32
Weights: torchvision's networks with 64 outputs, models.glorot_ under seed 17 after
torch.manual_seed(17), as profile_matched.network.  One process is one session: after the quiet
check, the process loads and builds the network (tracing and graph capture included, untimed) and
makes one call; then it idles 60 s, warms up for 60 s and records every call of a 120 s window in
one paced schedule, call i + 1 starting no earlier than max(s_i + 10 ms, e_i), the process sleeping
until then; last come the execution checks on 256 further frames: logits of shape (1, 64), all
finite, the timed path's index the argmax of the build's own logits (for resnet18_ts, a second
traced module to the logits), with the eager FP32 network's argmax agreement and largest logit
difference recorded.  Telemetry as profile_cadence.py: nvidia-smi every second, the machine's state
and the GPU's compute processes before the warm-up and after the window.

  --slot k     session k (1-45) of the frozen order; slots 16-45 (rounds 6-15) refuse to start
               unless protocol.json records the hash of the screening decisions and the decisions
               file matches it
  --attempt 2  the one replacement R26 allows, after an objective execution failure
  --smoke DIR  every build for a few seconds, into DIR outside this directory, without the frozen
               record or a quiet machine; its output is discarded

Exit status: 0; 3 if the machine never became quiet (nothing timed); 4 on an objective execution
failure (a crash or CUDA error, failed execution checks or a missing timing record), with its
record written.  Nothing else repeats a session: temperature, clocks, drift, encoder activity and
new GPU processes are results.  Outputs, never overwritten: profile_audit/a<kk>[_a2].{json,npz},
each with its nvidia-smi log a<kk>[_a2]_gpu.csv.

usage:  python3 profile_audit.py --slot 1..45 [--attempt 2] | --smoke DIR
"""
import argparse, copy, hashlib, json, os, sys, time, traceback, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'scheduler'))
import profile_matched as PM
import profile_cadence as PC

OUT = os.path.join(HERE, 'profile_audit')
PROTOCOL = PM.PROTOCOL
KEY = 'decision_audit'
SELF = '../latency/profile_audit.py'             # this script, as protocol.json names it
BUILDS = ('resnet18_ts', 'vit_fp16_graph', 'vit_fp32')
SPACING_MS = 10
N_ROUNDS, CAL_ROUNDS = 15, 5
N_SLOTS = N_ROUNDS * len(BUILDS)
IDLE_S, WARMUP_S, WINDOW_S = 60, 60, 120
SMOKE_S = 2.0


class ResNetTS:
    """ResNet-18 in FP32, the path from the device frame to the index traced and frozen."""

    def __init__(self, torch, dev, ring_pinned):
        import torchvision
        import models as M
        from calibrate_opt2 import frozen
        self.name, self.torch, self.dev = 'resnet18_ts', torch, dev
        torch.manual_seed(PM.WEIGHT_SEED)
        net = torchvision.models.resnet18(weights=None, num_classes=64)
        M.glorot_(net, PM.WEIGHT_SEED)
        self.eager = net.eval().to(dev)
        self.weights_sha256 = PM.state_sha(self.eager)
        mean = torch.tensor(PM.MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(PM.STD, device=dev).view(1, 3, 1, 1)

        class Path(torch.nn.Module):               # device frame (H, W, 3) uint8 -> logits
            def __init__(s):
                super().__init__()
                s.net = net
                s.register_buffer('mean', mean)
                s.register_buffer('std', std)

            def forward(s, x):
                x = x.permute(2, 0, 1).unsqueeze(0).float().contiguous()
                return s.net((x / 255.0 - s.mean) / s.std)

        class Decide(torch.nn.Module):             # -> the beam index, on the device
            def __init__(s):
                super().__init__()
                s.p = Path()

            def forward(s, x):
                return torch.argmax(s.p(x)[0])

        self.ring, self.i = ring_pinned, 0
        self.dbuf = torch.empty((PM.IMG, PM.IMG, 3), dtype=torch.uint8, device=dev)
        self.dbuf.copy_(ring_pinned[0])
        self.f_idx = frozen(Decide(), (self.dbuf,))
        self.f_logits = frozen(Path(), (self.dbuf,))
        self.mean, self.std = mean, std

    def _next(self):
        f = self.ring[self.i]
        self.i = (self.i + 1) % PM.RING
        return f

    def call(self):
        self.dbuf.copy_(self._next(), non_blocking=True)
        return int(self.f_idx(self.dbuf))

    def check(self, frame_pinned):
        self.dbuf.copy_(frame_pinned, non_blocking=True)
        idx = int(self.f_idx(self.dbuf))
        return idx, self.f_logits(self.dbuf).float().cpu().clone()

    def eager_logits(self, frame_pinned):
        x = frame_pinned.to(self.dev).permute(2, 0, 1).unsqueeze(0).float()
        return self.eager((x / 255.0 - self.mean) / self.std).float().cpu()

    def free(self):
        for a in ('eager', 'f_idx', 'f_logits', 'dbuf'):
            if hasattr(self, a):
                delattr(self, a)


def make(name, torch, dev, ring):
    if name == 'resnet18_ts':
        return ResNetTS(torch, dev, ring)
    vit32 = PM.network(torch)
    pm_name = {'vit_fp16_graph': 'fp16_graph', 'vit_fp32': 'fp32'}[name]
    return PM.Build(pm_name, copy.deepcopy(vit32) if pm_name == 'fp32' else vit32, torch, dev,
                    ring)


def checks(b, check, torch):
    """PM.execution_checks, with the eager FP32 network as the recorded comparison."""
    import numpy as np
    if isinstance(b, ResNetTS):
        ref = np.stack([b.eager_logits(f)[0].numpy() for f in check])
    else:
        vit = PM.network(torch).to(b.dev).eval()
        mean = torch.tensor(PM.MEAN, device=b.dev).view(1, 3, 1, 1)
        std = torch.tensor(PM.STD, device=b.dev).view(1, 3, 1, 1)
        ref = np.stack([vit((f.to(b.dev).permute(2, 0, 1).unsqueeze(0).float() / 255.0 - mean)
                            / std)[0].float().cpu().numpy() for f in check])
        del vit
    out, _ = PM.execution_checks(b, check, ref)
    return out


def frozen_record():
    """The analysis entry's frozen record, its sha256 and the recorded decisions hash."""
    P = json.load(open(PROTOCOL))
    A = [a for a in P['analyses'] if a['key'] == KEY]
    assert len(A) == 1, f"protocol.json declares analysis '{KEY}' exactly once"
    fz = A[0].get('frozen')
    assert fz, "the session order is not frozen in protocol.json: run audit_screen.py --freeze"
    return fz, hashlib.sha256(json.dumps(fz, sort_keys=True).encode()).hexdigest(), \
        A[0].get('decisions_sha256')


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
    stem = 'smoke' if smoke else f"a{a.slot:02d}" + ('_a2' if a.attempt == 2 else '')
    out = os.path.join(out_dir, stem + '.json')
    for p in (out, out[:-5] + '.npz', out[:-5] + '_gpu.csv'):
        assert not os.path.exists(p), f"{p} exists: an output is written once"

    # checks that need no torch
    import trace_decisive as TD
    assert PM.sha_file(os.path.join(HERE, 'models.py')) == TD.SRC_SHA256['models.py'], \
        "models.py changed"
    assert PM.sha_file(os.path.join(HERE, 'calibrate_opt2.py')) == \
        TD.SRC_SHA256['calibrate_opt2.py'], "calibrate_opt2.py changed"
    assert sorted(os.sched_getaffinity(0)) == PM.AFFINITY, "run under taskset -c 0-15"
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(v) == '1', f"{v} must be 1"
    if smoke:
        fz, fz_sha, dec_sha = None, None, None
        plan = list(BUILDS)
        phase_s = (SMOKE_S, SMOKE_S, 2 * SMOKE_S)
        rnd = None
    else:
        fz, fz_sha, dec_sha = frozen_record()
        assert PM.sha_file(os.path.abspath(__file__)) == fz['scripts_sha256'][SELF], \
            "profile_audit.py differs from the frozen script"
        slot = [o for o in fz['order'] if o['slot'] == a.slot]
        assert len(slot) == 1, a.slot
        plan, rnd = [slot[0]['build']], slot[0]['round']
        phase_s = (fz['phases_s']['idle'], fz['phases_s']['warmup'], fz['phases_s']['window'])
        assert phase_s == (IDLE_S, WARMUP_S, WINDOW_S) and fz['spacing_ms'] == SPACING_MS
        if rnd > fz['calibration_rounds']:
            # the screening decisions are made from rounds 1-5 and recorded before round 6
            assert dec_sha, "the screening decisions are not recorded: run audit_screen.py --decide"
            dp = os.path.join(HERE, '..', 'scheduler', 'audit_decisions.json')
            assert PM.sha_file(dp) == dec_sha, "audit_decisions.json differs from its record"

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
    rec = {'stem': stem, 'mode': 'smoke' if smoke else 'session', 'slot': a.slot, 'round': rnd,
           'attempt': a.attempt, 'smoke': smoke, 'started': True, 'started_utc': PM.utc(),
           'frozen_sha256': fz_sha, 'decisions_sha256': dec_sha, 'frozen': fz,
           'compute_apps_at_start': apps0,
           'protocol': {'weight_seed': PM.WEIGHT_SEED, 'frame_seed': PM.FRAME_SEED,
                        'ring': PM.RING, 'check_frames': PM.N_CHECK,
                        'image': [PM.IMG, PM.IMG, 3], 'mean': PM.MEAN, 'std': PM.STD,
                        'spacing_ms': SPACING_MS, 'idle_s': phase_s[0], 'warmup_s': phase_s[1],
                        'window_s': phase_s[2]},
           'runs': [], 'quiet_log': qlog, 'env': env}
    raw, failed = {}, None
    try:
        ring_np, check_np = PM.frames()
        ring = [torch.from_numpy(f).pin_memory() for f in ring_np]
        check = [torch.from_numpy(f).pin_memory() for f in check_np]
        for name in plan:
            w = {'build': name, 'spacing_ms': SPACING_MS}
            rec['runs'].append(w)                  # kept, as far as it got, if a call fails
            torch.cuda.empty_cache()
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                t_init = time.time()
                b = make(name, torch, dev, ring)
                w['init_s'] = time.time() - t_init
            w['build_warnings'] = sorted({f"{x.category.__name__}: {x.message}"[:500]
                                          for x in wl})
            w['weights_sha256'] = b.weights_sha256
            if not smoke:
                assert b.weights_sha256 == fz['weights_sha256'][name], \
                    f"{name}'s weights differ from the frozen record"
            r = b.call()
            assert type(r) is int and 0 <= r < 64, (name, r)
            delta = int(SPACING_MS * 1e6)
            w['idle_start_time'] = time.time()
            time.sleep(phase_s[0])
            w['idle_end_time'] = time.time()
            w['state_before'] = PM.state(TD)
            s, e, epoch = PC.paced(b.call, phase_s[1] + phase_s[2], delta)
            w0 = int(phase_s[1] * 1e9)
            w['warmup_start_time'] = epoch
            w['window_start_time'] = epoch + phase_s[1]
            w['window_end_time'] = epoch + e[-1] / 1e9
            w['state_after'] = PM.state(TD)
            pre = f"{name}." if smoke else ''
            raw[pre + 'start_ns'], raw[pre + 'end_ns'] = s, e
            raw[pre + 'window_from_ns'] = np.array(w0, dtype=np.int64)
            win = s >= w0
            w['warmup'] = PC.summary(s[~win], e[~win], delta)
            w['window'] = PC.summary(s[win], e[win], delta)
            if not win.any() or not (~win).any() or not np.all(e > s):
                raise RuntimeError('missing or invalid timing records')
            chk = checks(b, check, torch)
            w['execution_checks'] = chk
            print(f"  {stem} {name}: {int(win.sum())} calls, mean "
                  f"{w['window']['mean_us']:.1f} us", flush=True)
            b.free()
            del b
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
