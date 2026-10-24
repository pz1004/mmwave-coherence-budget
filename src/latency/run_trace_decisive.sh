#!/usr/bin/env bash
# The three decisive implementations, timed again in three runs, one case per process, sequential
# and never concurrent (two processes at once would contaminate each other's timings).  Pinning
# and environment as run_all.sh: taskset -c 0-15 throughout, eight threads for case A with no
# CUDA device visible, one thread on the GPU.  trace_decisive.py waits
# for a quiet machine and refuses to overwrite an output.  Exit status 2 marks a contended process,
# which is repeated once, immediately, as attempt 2; a contended second attempt is kept and
# reported, not repeated.  Status 3 means the machine never became quiet: the campaign stops.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
export PYTHONWARNINGS=ignore

one() {   # $1 case, $2 run, $3 attempt
  if [ "$1" = A ]; then
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= \
      taskset -c 0-15 python3 trace_decisive.py --case "$1" --run "$2" --attempt "$3"
  else
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      taskset -c 0-15 python3 trace_decisive.py --case "$1" --run "$2" --attempt "$3"
  fi
}

for r in 1 2 3; do
  for c in A B C; do
    echo "### run $r, case $c"
    one "$c" "$r" 1
    st=$?
    if [ "$st" = 2 ]; then
      echo "### run $r, case $c: contended; attempt 2"
      one "$c" "$r" 2
      st=$?
      [ "$st" = 2 ] && st=0
    fi
    if [ "$st" != 0 ]; then
      echo "### stopped at run $r, case $c (exit $st)"
      exit "$st"
    fi
  done
done
echo "### DONE"
