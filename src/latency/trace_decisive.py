#!/usr/bin/env python3
"""The three decisive implementations, timed again.

Declared in protocol.json (analysis 'trace_decisive') before it ran, with rule R20 for reading its
outputs: every number in Table I, rule R5 and the margins keeps coming from the stored timings
(p_*.json, p_gpu_heavy_full.json, o2_*.json), and this campaign is reported beside them.  Rule R21,
set later, reads these runs together with the stored timings as those inputs (reconcile.py).

WHY.  The three implementations on which the screen's verdicts turn are timed again, under
recorded operating conditions, with every call kept and with the beam index made available where
it is consumed.  The stored timings kept every call only for the uncompiled runs at reduced
counts, one run of three back-to-back weight seeds; the TorchScript and the full-count GPU runs
kept only summaries; and no GPU timing copied the beam index to the CPU.

WHAT.  One case per process (run_trace_decisive.sh sets the pinning and environment, as
run_all.sh did):
  A  ResNet-18 on eight P-core threads   taskset -c 0-15, 8 threads, no CUDA device visible
  B  ViT-B/16 on the GPU                 taskset -c 0-15, 1 thread
  C  ResNet-18 on the GPU                taskset -c 0-15, 1 thread
Runtimes: uncompiled (calibrate.build) and TorchScript of the whole decision (calibrate_opt2.build),
both imported unchanged, so the networks, weights (seeds 17, 29, 43), inputs and freezing are the
stored campaign's.  Boundaries: 'host', the beam index converted to a CPU integer inside the timed
call (int(), which on the GPU copies it from device memory); 'device', the index left in device
memory as the stored GPU timings left it (case C only, to measure the copy).  Per seed: 1,000
warm-up and 10,000 timed calls (the protocol's full counts), then the contention recheck with a
quarter of each (250 and 2,500); drift = |recheck p95 - p95| / p95.  A configuration whose drift
exceeds 10% for any seed is contended, and so is its process (exit status 2).

Before timing, the process waits for a quiet machine: ten samples over 5 s, each with GPU
utilization at most 10% and a one-minute load average at most 1.0, polled every 30 s for up to
60 min; if it never comes, it writes r<run>_<case>_notstarted.json and exits with status 3.  The
machine's state (load, CPU governor, energy preference, turbo, the pinned CPUs' frequencies, the
GPU's utilization, clocks, temperature, power and processes, the busiest CPU processes) is read
from /proc, sysfs, nvidia-smi and ps before and after every configuration, never during one.

Outputs, never overwritten: trace_decisive/r<run>_<case>[_a2].json (per seed: counts, median,
mean, quartiles, p95, p99, minimum, maximum, the recheck's p95, drift, lag-one autocorrelation)
and the .npz beside it holding every timed call in nanoseconds, in call order.

Gates: the sha256 of calibrate.py, calibrate_opt2.py and models.py equal the values below, those
of the code in this repository (the stored runs, like protocol.json's source_sha256, record the
code that ran, which differs from it only in calibrate.py's help text); affinity, thread counts and device as declared; a
'device' configuration returns a CUDA tensor and a 'host' one a Python int; the output does not
exist.  --smoke DIR runs every configuration with 3 warm-up and 10 timed calls (recheck 1 and 3)
into DIR, outside this repository's output directory, without waiting for a quiet machine; it was
run once before the campaign was declared, and its output discarded.

usage:  python3 trace_decisive.py --case A|B|C --run 1|2|3 [--attempt 2] [--smoke DIR]
"""
import argparse, hashlib, json, math, os, subprocess, sys, time, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'trace_decisive')
SRC_SHA256 = {
    'calibrate.py': 'b24299782672abb68eedeb86e01d9902c8611f2857b2aa12896b10fc271c2646',
    'calibrate_opt2.py': 'fb93f5489ffaa25f3bbd0f1f603e4ef65fae806b940bc38c1689e85762728e4b',
    'models.py': '1819703b5203e83bdce77667f27878b711c7cc585797eec2335d22d551d4d20f',
}
CASES = {   # case -> (component, profile, device, threads, boundaries)
    'A': ('resnet18_vision', 'cpu-8t-pcore', 'cpu', 8, ('host',)),
    'B': ('vit_b16_vision', 'gpu-resident', 'cuda', 1, ('host',)),
    'C': ('resnet18_vision', 'gpu-resident', 'cuda', 1, ('device', 'host')),
}
RUNTIMES = ('uncompiled', 'torchscript')
AFFINITY = list(range(16))                      # taskset -c 0-15: the eight P-cores, with SMT
WARMUP, TIMED = 1_000, 10_000
RECHECK = (WARMUP // 4, TIMED // 4)
SMOKE, SMOKE_RECHECK = (3, 10), (1, 3)
DRIFT_MAX = 0.10
QUIET_SAMPLES, QUIET_GAP_S, QUIET_UTIL_MAX, QUIET_LOAD_MAX = 10, 0.5, 10, 1.0
QUIET_POLL_S, QUIET_TIMEOUT_S = 30, 3600


def sha(p):
    return hashlib.sha256(open(os.path.join(HERE, p), 'rb').read()).hexdigest()


def _read(p):
    try:
        return open(p).read().strip()
    except OSError:
        return None


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:             # recorded, never fatal: the state is a record, not a gate
        return f"unavailable: {e!r}"


def gpu_util():
    out = _run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
    try:
        return int(out.split()[0])
    except (ValueError, IndexError):
        return None


def load1():
    return float(open('/proc/loadavg').read().split()[0])


def state():
    """The machine's state, from /proc, sysfs, nvidia-smi and ps only (no CUDA call)."""
    cpu = '/sys/devices/system/cpu'
    freqs = [_read(f'{cpu}/cpu{i}/cpufreq/scaling_cur_freq') for i in AFFINITY]
    return {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'loadavg': open('/proc/loadavg').read().split()[:3],
        'governor': _read(f'{cpu}/cpu0/cpufreq/scaling_governor'),
        'energy_performance_preference': _read(f'{cpu}/cpu0/cpufreq/energy_performance_preference'),
        'no_turbo': _read(f'{cpu}/intel_pstate/no_turbo'),
        'pinned_cpu_khz': [int(f) if f and f.isdigit() else None for f in freqs],
        'gpu': _run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,clocks.sm,'
                     'clocks.mem,temperature.gpu,power.draw,pstate',
                     '--format=csv,noheader,nounits']),
        'gpu_processes': _run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u']).splitlines(),
        'top_cpu': _run(['ps', '-eo', 'pid,comm,pcpu', '--sort=-pcpu']).splitlines()[:8],
    }


def quiet(enforce):
    """Wait for a quiet machine; returns (quiet, log)."""
    log, t0 = [], time.time()
    while True:
        s = []
        for _ in range(QUIET_SAMPLES):
            s.append((gpu_util(), load1()))
            time.sleep(QUIET_GAP_S)
        ok = all(u is not None and u <= QUIET_UTIL_MAX and l <= QUIET_LOAD_MAX for u, l in s)
        log.append({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'samples': s, 'quiet': ok})
        if ok or not enforce:
            return ok, log
        if time.time() - t0 + QUIET_POLL_S > QUIET_TIMEOUT_S:
            return False, log
        time.sleep(QUIET_POLL_S)


def nearest_rank(a, q):
    s = sorted(a)
    return int(s[int(math.ceil(q * len(s))) - 1])


def summary(a, b, calibrate):
    import numpy as np
    p95, rp95 = calibrate.nearest_rank_p95(a), calibrate.nearest_rank_p95(b)
    return {'n_timed': int(a.size), 'p95_ns': p95, 'median_ns': float(np.median(a)),
            'mean_ns': float(a.mean()), 'q1_ns': float(np.percentile(a, 25)),
            'q3_ns': float(np.percentile(a, 75)), 'p99_ns': nearest_rank(a, 0.99),
            'min_ns': int(a.min()), 'max_ns': int(a.max()),
            'lag1_autocorrelation': float(np.corrcoef(a[:-1], a[1:])[0, 1]),
            'recheck_n_timed': int(b.size), 'recheck_p95_ns': rp95,
            'drift': abs(rp95 - p95) / p95}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=sorted(CASES))
    ap.add_argument('--run', required=True, type=int, choices=(1, 2, 3))
    ap.add_argument('--attempt', type=int, default=1, choices=(1, 2))
    ap.add_argument('--smoke', default='')
    a = ap.parse_args()
    comp, profile, device, threads, bounds = CASES[a.case]
    smoke = bool(a.smoke)
    out_dir = os.path.abspath(a.smoke) if smoke else OUT
    if smoke:
        assert not out_dir.startswith(os.path.abspath(OUT)), "a smoke test never writes the outputs"
    stem = f"r{a.run}_{a.case}" + ('_a2' if a.attempt == 2 else '')
    out = os.path.join(out_dir, stem + '.json')
    assert not os.path.exists(out) and not os.path.exists(out[:-5] + '.npz'), \
        f"{out} exists: an output is written once"

    # gates that need no torch
    for f, h in SRC_SHA256.items():
        assert sha(f) == h, f"{f} changed: the stored campaign's code is what this re-times"
    assert sorted(os.sched_getaffinity(0)) == AFFINITY, "run under taskset -c 0-15"
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(v) == str(threads), f"{v} must be {threads}, as in run_all.sh"
    if device == 'cpu':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', "case A runs with no CUDA device"

    # set up as calibrate.main and calibrate_opt2 did, before any torch operation
    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    if threads == 1:
        torch.set_num_interop_threads(1)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    import numpy as np
    import calibrate, calibrate_opt2               # both silence warnings, as run_all.sh did
    assert torch.get_num_threads() == threads
    assert torch.cuda.is_available() == (device == 'cuda')

    ok, qlog = quiet(enforce=not smoke)
    if not ok:
        rec = {'case': a.case, 'run': a.run, 'attempt': a.attempt, 'started': False,
               'reason': 'the machine was never quiet', 'quiet_log': qlog, 'state': state()}
        os.makedirs(out_dir, exist_ok=True)
        json.dump(rec, open(os.path.join(out_dir, stem + '_notstarted.json'), 'w'), indent=1)
        print(f"{stem}: not started, machine never quiet", flush=True)
        sys.exit(3)

    w_warmup, w_timed = SMOKE if smoke else (WARMUP, TIMED)
    r_warmup, r_timed = SMOKE_RECHECK if smoke else RECHECK
    if device == 'cuda':
        assert torch.cuda.get_device_name(0) == 'Quadro RTX 8000'
    t_start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    configs, raw, caught = [], {}, []
    for rt in RUNTIMES:
        for bd in bounds:
            before = state()
            seeds = []
            for seed in calibrate.SEEDS:
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter('always')
                    if rt == 'uncompiled':
                        f = dict((c[0], c[1]) for c in calibrate.build(device=device, seed=seed))[comp]
                    else:
                        f = calibrate_opt2.build(seed=seed, dev=device)[comp]
                caught += [f"{rt} s{seed}: {w.category.__name__}: {w.message}" for w in wl]
                fn = (lambda f=f: int(f())) if bd == 'host' else f
                r = fn()
                if bd == 'host':
                    assert type(r) is int, (rt, bd, type(r))
                else:
                    assert isinstance(r, torch.Tensor) and r.is_cuda, (rt, bd, type(r))
                x = calibrate.timed(fn, w_warmup, w_timed)
                y = calibrate.timed(fn, r_warmup, r_timed)
                raw[f"{rt}.{bd}.s{seed}"] = x
                raw[f"{rt}.{bd}.s{seed}.recheck"] = y
                seeds.append({'seed': seed, 'n_warmup': w_warmup, 'recheck_n_warmup': r_warmup,
                              **summary(x, y, calibrate)})
                print(f"  {a.case} r{a.run} {rt:11s} {bd:6s} s{seed}: p95 "
                      f"{seeds[-1]['p95_ns'] / 1e3:9.1f} us, drift {seeds[-1]['drift']:.3f}",
                      flush=True)
            configs.append({'runtime': rt, 'boundary': bd, 'per_seed': seeds,
                            'charge_ns': max(s['p95_ns'] for s in seeds),
                            'contended': any(s['drift'] > DRIFT_MAX for s in seeds),
                            'state_before': before, 'state_after': state()})
    contended = any(c['contended'] for c in configs)
    rec = {'case': a.case, 'component': comp, 'profile': profile, 'device': device,
           'threads': threads, 'interop_threads': torch.get_num_interop_threads(),
           'run': a.run, 'attempt': a.attempt, 'smoke': smoke, 'started': True,
           'started_utc': t_start, 'ended_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'protocol': {'warmup': w_warmup, 'timed': w_timed, 'recheck': [r_warmup, r_timed],
                        'seeds': list(calibrate.SEEDS), 'percentile': 'nearest-rank p95',
                        'drift_max': DRIFT_MAX, 'boundaries': list(bounds),
                        'runtimes': list(RUNTIMES)},
           'contended': contended, 'configurations': configs, 'quiet_log': qlog,
           'build_warnings': caught,
           'env': {**calibrate.env_record(profile, threads), 'torchvision': __import__(
               'torchvision').__version__, 'cuda': torch.version.cuda,
               'cudnn': torch.backends.cudnn.version(), 'source_sha256': SRC_SHA256,
               'script_sha256': sha(os.path.basename(__file__)),
               'env_vars': {k: os.environ.get(k) for k in (
                   'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                   'CUDA_VISIBLE_DEVICES', 'PYTHONWARNINGS')}}}
    os.makedirs(out_dir, exist_ok=True)
    assert not os.path.exists(out), f"{out} appeared while timing"
    np.savez_compressed(out[:-5] + '.npz', **raw)
    json.dump(rec, open(out, 'w'), indent=1)
    print(f"wrote {out}{' (contended)' if contended else ''}", flush=True)
    sys.exit(2 if contended else 0)


if __name__ == '__main__':
    main()
