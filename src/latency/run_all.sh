#!/usr/bin/env bash
# Sequential, never concurrent: two profiles running at once would contaminate
# each other's timings.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
export PYTHONWARNINGS=ignore

echo "### cpu-1t-pcore  (pinned P-core, 1 thread) --- the G3 profile"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 2 python3 calibrate.py --profile cpu-1t-pcore --device cpu \
  --threads 1 --out "$D/p_cpu1t_pcore.json"

echo "### cpu-1t-ecore  (pinned E-core, 1 thread) --- UE-class proxy"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 24 python3 calibrate.py --profile cpu-1t-ecore --device cpu \
  --threads 1 --out "$D/p_cpu1t_ecore.json"

echo "### cpu-8t-pcore  (8 P-core threads) --- server CPU"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  taskset -c 0-15 python3 calibrate.py --profile cpu-8t-pcore --device cpu \
  --threads 8 --out "$D/p_cpu8t_pcore.json"

echo "### gpu  (RTX 8000, device-resident, includes synchronize)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0-15 python3 calibrate.py --profile gpu-resident --device cuda \
  --threads 1 --out "$D/p_gpu.json"

echo "### DONE"
