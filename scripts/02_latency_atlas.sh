#!/usr/bin/env bash
# Stage 2 -- the predictor latency atlas.
# TIMING STAGE: run on a quiet machine, nothing else on the pinned cores.
# A contended run inflated tau_1 from 13.8 us to 21.1 us in our own testing,
# which is larger than several of the effects the paper reports.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../src/latency"
./run_all.sh              # calibrate.py over 4 profiles -> p_*.json
python3 analyze.py        # -> atlas.json  (reads the four p_*.json profiles)

# TorchScript variants, on the three profiles they were measured on, pinned and threaded as
# run_all.sh pins and threads them: the network alone (calibrate_opt.py -> o_*.json) and the
# whole decision (calibrate_opt2.py PROFILE OUT DEVICE -> o2_*.json).
export PYTHONWARNINGS=ignore
pinned() {   # CPUS THREADS COMMAND...
  local cpus=$1 threads=$2; shift 2
  OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
    taskset -c "$cpus" "$@"
}
pinned 2    1 python3 calibrate_opt.py --profile cpu-1t-pcore --device cpu --threads 1 --out o_cpu1t_pcore.json
pinned 24   1 python3 calibrate_opt.py --profile cpu-1t-ecore --device cpu --threads 1 --out o_cpu1t_ecore.json
pinned 0-15 1 python3 calibrate_opt.py --profile gpu-resident --device cuda --threads 1 --out o_gpu.json
pinned 2    1 python3 calibrate_opt2.py cpu-1t-pcore o2_cpu1t_pcore.json cpu
pinned 24   1 python3 calibrate_opt2.py cpu-1t-ecore o2_cpu1t_ecore.json cpu
pinned 0-15 1 python3 calibrate_opt2.py gpu-resident o2_gpu.json cuda
