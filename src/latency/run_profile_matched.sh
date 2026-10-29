#!/usr/bin/env bash
# The matched implementation-profiling study (protocol.json analysis 'matched', rule R22).
#   ./run_profile_matched.sh pilot   the pilot; then src/scheduler/matched.py --freeze
#   ./run_profile_matched.sh main    sessions 1, 2, 3, one process each, sequential
# Pinning and environment as run_trace_decisive.sh's case B: taskset -c 0-15, one thread.  Exit
# status 2 (contended) or 4 (a CUDA error) repeats a session once, immediately, as attempt 2; a
# second attempt is kept and reported whatever its status, never repeated.  Status 3 (the machine
# never became quiet) or any other failure stops the campaign.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
export PYTHONWARNINGS=ignore

one() {
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 0-15 python3 profile_matched.py "$@"
}

case "${1:-}" in
  pilot)
    one --pilot
    exit $?
    ;;
  main)
    for s in 1 2 3; do
      echo "### session $s"
      one --session "$s" --attempt 1
      st=$?
      if [ "$st" = 2 ] || [ "$st" = 4 ]; then
        echo "### session $s: status $st; attempt 2"
        one --session "$s" --attempt 2
        st=$?
        { [ "$st" = 2 ] || [ "$st" = 4 ]; } && st=0
      fi
      if [ "$st" != 0 ]; then
        echo "### stopped at session $s (exit $st)"
        exit "$st"
      fi
    done
    ;;
  *)
    echo "usage: $0 pilot|main" >&2
    exit 64
    ;;
esac
