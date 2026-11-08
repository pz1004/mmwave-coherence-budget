#!/usr/bin/env bash
# The decision audit's timing campaign (protocol.json analysis 'decision_audit', rule R26).
#   ./run_profile_audit.sh main FIRST LAST   slots FIRST..LAST of the frozen order, one process
#                                            each, sequential (audit_screen.py --freeze first;
#                                            slots 16-45 also need audit_screen.py --decide)
# Pinning and environment as run_profile_cadence.sh: taskset -c 0-15, one thread.  Exit status 4,
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
    taskset -c 0-15 python3 profile_audit.py "$@"
}

failed() {    # attempt 1's record exists and records a failure
  python3 -c "import json, sys; sys.exit(0 if json.load(open('$1'))['failed'] else 1)"
}

case "${1:-}" in
  main)
    first="${2:?first slot}"; last="${3:?last slot}"
    for k in $(seq "$first" "$last"); do
      kk=$(printf '%02d' "$k")
      [ -f "profile_audit/a${kk}_a2.json" ] && continue
      if [ -f "profile_audit/a${kk}.json" ]; then
        failed "profile_audit/a${kk}.json" || continue
        st=4
      else
        echo "### slot $k  $(date -u +%H:%M:%S)"
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
    echo "### slots $first-$last done  $(date -u +%H:%M:%S)"
    ;;
  *)
    echo "usage: $0 main FIRST LAST" >&2
    exit 64
    ;;
esac
