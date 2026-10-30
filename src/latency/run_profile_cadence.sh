#!/usr/bin/env bash
# The cadence study (protocol.json analysis 'cadence', rule R24).
#   ./run_profile_cadence.sh main [first_slot]   slots 1-18 of the frozen order, one process each,
#                                                sequential (src/scheduler/cadence.py --freeze first)
# Pinning and environment as run_profile_matched.sh: taskset -c 0-15, one thread.  Exit status 4,
# an objective execution failure (a crash or CUDA error, failed execution checks or a missing
# timing record), repeats a slot once, immediately, as attempt 2; the failed attempt is kept, and a
# second attempt is kept whatever its status, never repeated.  Status 3 (the machine never became
# quiet; nothing was timed) or any other status stops the campaign, which resumes at that slot.  A
# slot with a record that did not fail, or with a second attempt, is skipped, so resuming never
# repeats a session.
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
export PYTHONWARNINGS=ignore

one() {
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 0-15 python3 profile_cadence.py "$@"
}

failed() {    # attempt 1's record exists and records a failure
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$1'))['failed'] else 1)"
}

case "${1:-}" in
  main)
    for k in $(seq "${2:-1}" 18); do
      kk=$(printf '%02d' "$k")
      [ -f "profile_cadence/c${kk}_a2.json" ] && continue
      if [ -f "profile_cadence/c${kk}.json" ]; then
        failed "profile_cadence/c${kk}.json" || continue
        st=4
      else
        echo "### slot $k"
        one --slot "$k" --attempt 1
        st=$?
      fi
      if [ "$st" = 4 ]; then
        echo "### slot $k: objective failure; attempt 2"
        one --slot "$k" --attempt 2
        st=$?
        [ "$st" = 4 ] && st=0
      fi
      if [ "$st" != 0 ]; then
        echo "### stopped at slot $k (exit $st)"
        exit "$st"
      fi
    done
    echo "### all 18 slots done"
    ;;
  *)
    echo "usage: $0 main [first_slot]" >&2
    exit 64
    ;;
esac
