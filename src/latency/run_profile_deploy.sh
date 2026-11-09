#!/usr/bin/env bash
# The deployment-runtime study (protocol.json analysis 'deploy', rule R38).
#   ./run_profile_deploy.sh pilot   the three groups' pilots; then src/scheduler/deploy.py --freeze
#   ./run_profile_deploy.sh main    sessions 1, 2, 3, each the three groups in turn, one process each
# Pinning and environment as run_profile_matched.sh: taskset -c 0-15; the GPU groups with one
# thread, the CPU group with eight.  Exit status 2 (contended) or 4 (a CUDA error) repeats a
# session's group once, immediately, as attempt 2; a second attempt is kept and reported whatever its
# status, never repeated.  Status 3 (the machine never became quiet) or any other failure stops the
# campaign.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
export PYTHONWARNINGS=ignore

threads() { [ "$1" = rn_cpu ] && echo 8 || echo 1; }

one() {
  local t; t=$(threads "$1"); shift
  OMP_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t MKL_NUM_THREADS=$t \
    taskset -c 0-15 python3 profile_deploy.py "$@"
}

case "${1:-}" in
  pilot)
    for g in rn_gpu vit_gpu rn_cpu; do
      echo "### pilot $g"
      one "$g" --pilot "$g" || { st=$?; echo "### stopped at pilot $g (exit $st)"; exit "$st"; }
    done
    ;;
  main)
    for s in 1 2 3; do
      for g in rn_gpu vit_gpu rn_cpu; do
        echo "### session $s $g"
        one "$g" --session "$s" "$g" --attempt 1
        st=$?
        if [ "$st" = 2 ] || [ "$st" = 4 ]; then
          echo "### session $s $g: status $st; attempt 2"
          one "$g" --session "$s" "$g" --attempt 2
          st=$?
          { [ "$st" = 2 ] || [ "$st" = 4 ]; } && st=0
        fi
        if [ "$st" != 0 ]; then
          echo "### stopped at session $s $g (exit $st)"
          exit "$st"
        fi
      done
    done
    ;;
  *)
    echo "usage: $0 pilot|main" >&2
    exit 64
    ;;
esac
