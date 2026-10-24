#!/usr/bin/env bash
# Sequential, never concurrent: two profiles at once contaminate each other.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"; export PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
echo "### cpu-1t-pcore (the G3 primary profile)"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 2 \
  python3 latency.py --profile cpu-1t-pcore --device cpu --threads 1 --out "$D/lat_cpu1t_pcore.json"
echo "### cpu-1t-ecore"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 24 \
  python3 latency.py --profile cpu-1t-ecore --device cpu --threads 1 --out "$D/lat_cpu1t_ecore.json"
echo "### cpu-8t-pcore"
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 0-15 \
  python3 latency.py --profile cpu-8t-pcore --device cpu --threads 8 --out "$D/lat_cpu8t_pcore.json"
echo "### gpu-resident"
unset CUDA_VISIBLE_DEVICES
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-15 \
  python3 latency.py --profile gpu-resident --device cuda --threads 1 --out "$D/lat_gpu.json"
echo "### DONE"
