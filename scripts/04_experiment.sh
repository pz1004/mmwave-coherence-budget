#!/usr/bin/env bash
# Stage 4 -- the scheduler experiment. Needs stages 1-3 to have run.
#
# Safe to re-run end to end.
#   * protocol.py verifies the protocol record (protocol.json): the protocol frozen on
#     2026-09-15, the input files it hashed, and the stored files it lists.
#   * Training keeps every seed whose checkpoints exist; nothing makes CUDA
#     training bit-deterministic, so a retrained seed would silently replace
#     the weights the recorded numbers were computed with.
#   * The timing stage re-measures every charge.  It runs only with RETIME=1, on a quiet
#     machine; the charges it writes are listed with their hashes in protocol.json, so a
#     re-timing fails protocol.py until the record says so.
#   * The later analyses refuse to overwrite their outputs, and the long ones run only
#     when their flag is set (RERANK, EXACT, SCREEN, CALIB, NRCFG, LEVELS, TRACE,
#     MATCHED, CADENCE).
#
# The seed list (ten seeds) comes from protocol.json; `--seeds 17 29 43` re-runs the
# pre-registered three.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../src/scheduler"

python3 protocol.py            # verifies the protocol record (never re-freezes)
python3 build.py               # observations from src/sites -> dataset.json
python3 train.py               # exit predictors (existing seeds kept)
python3 train_base.py          # comparator baselines -> baselines.json
if [ "${RETIME:-0}" = 1 ]; then
  ./run_latency.sh             # TIMING STAGE -- quiet machine, see 02
fi
python3 profile_pred.py        # -> predictor_profile.json
python3 sched.py               # solve the controller -> scheduler.json, tables_*.npz

# Four evaluation runs.  The validation city goes FIRST: the other runs pick
# their deployable baseline from validation.json.
python3 evaluate.py --sites city_1_losangeles_28 --out validation.json
python3 evaluate.py --out results.json
python3 evaluate.py --sites city_4_phoenix_28 city_5_philadelphia_28 \
                            city_6_miami_28 city_7_sandiego_28 --out heldout.json
python3 evaluate.py --free-compute --out results_freecompute.json

python3 analyze.py             # variant selection on validation -> report.json
python3 window.py              # the operating window -> window.json
python3 rho.py                 # T* vs rho
python3 accounting.py          # Zhang et al. reproduction -> accounting.json
python3 selection_stability.py # leave-one-city-out -> selection_stability.json
python3 interleave.py          # how often a probe falls between two computes
python3 comparator.py          # why the published comparator loses rate as T grows

# The pre-registered analysis: seeds 17/29/43 and the frozen estimator.  It has its own
# validation run, so its deployable baseline and its matched selection are its own;
# report_3seed.json carries the pre-registered primary criterion, reported whatever the
# ten-seed run shows.
P3="--seeds 17 29 43"
python3 evaluate.py $P3 --sites city_1_losangeles_28 --out validation_3seed.json
python3 evaluate.py $P3 --validation validation_3seed.json --out results_3seed.json
python3 evaluate.py $P3 --validation validation_3seed.json --sites city_4_phoenix_28 \
    city_5_philadelphia_28 city_6_miami_28 city_7_sandiego_28 --out heldout_3seed.json
python3 analyze.py --suffix _3seed

# Sensitivity sweeps (descriptive): rho and t_p one at a time, each with its own solve and
# validation run.  About an hour and a half on the GPU; SWEEPS=0 skips them and reuses
# whatever sweep_*.json are on disk.
if [ "${SWEEPS:-1}" = 1 ]; then
  python3 sweeps.py
else
  python3 sweeps.py --summarize-only
fi

# Descriptive analyses: the predictor's quality as a sweep axis (predq.json), channel aging
# of the fixed menu (aging.json), and what the three solvers cost (complexity.json).
# PREDQ=0, AGING=0, COMPLEXITY=0 skip a stage and reuse what is on disk.
if [ "${PREDQ:-1}" = 1 ]; then
  python3 sweeps.py --predq
else
  python3 sweeps.py --predq --summarize-only
fi
if [ "${AGING:-1}" = 1 ]; then python3 aging.py; fi
if [ "${COMPLEXITY:-1}" = 1 ]; then python3 complexity.py; fi
# The reading rules run last: R6 and R7 read the analyses above.
python3 reading_rules.py       # R1-R7 -> rules.json

# Selection-aware uncertainty for Table I's changes of the best method, read by rule R8.
# It reads results*.json, heldout.json, aging.json and accounting.json, so it runs after
# them; WINNER=0 skips the run and keeps winner_uncertainty.json.
if [ "${WINNER:-1}" = 1 ]; then python3 winner_uncertainty.py; fi

# Proposition 1(c), the perfect-predictor bound under channel aging: aging_bound.py re-runs
# the aging simulation and must reproduce aging.json.  About seven minutes on the GPU;
# AGINGBOUND=0 skips it and keeps aging_bound.json.
if [ "${AGINGBOUND:-1}" = 1 ]; then python3 aging_bound.py; fi

# The first re-solve: the time for choosing the controller's next codeword after an
# inference call, counted on every probe of a sequence that invokes the predictor, with
# every controller result re-solved, and the adaptive baseline as published (rule R9).
# rerank.py --run writes only files tagged _rr and rerank.json, skipping each step whose
# output exists; a full run takes about 3.6 hours on the GPU.  It is skipped unless
# RERANK=1, and a re-run keeps rerank.json's gate record only if RERANK_DIR holds the
# gates.json that `rerank.py --check-only RERANK_DIR` wrote.
if [ "${RERANK:-0}" = 1 ]; then python3 rerank.py --run "${RERANK_DIR:?set RERANK_DIR}"; fi

# The second re-solve: the ranking time counted only from the first inference call on
# (kappa in the state, kappa-major tables whose lookup is timed on them), both adaptive
# baselines with a deadline guard, and the validation-frozen selector analysis (rules R10,
# R11).  exact.py writes only files tagged _rx, lat_lookup_aug.json and exact.json,
# skipping each step whose output exists; the lookup is timed once
# (`OMP_NUM_THREADS=1 taskset -c 2 python3 exact.py --time-lookup DIR`, after `--pilot DIR`)
# and never re-timed.  The runs take about four hours on the GPU and are skipped unless
# EXACT=1.
if [ "${EXACT:-0}" = 1 ]; then python3 exact.py --run "${EXACT_DIR:?set EXACT_DIR}"; [ -f selector.json ] || python3 selector.py; fi

# The sweep's efficiency under the CSI-RS procedure's own measurement (csirs.py, GPU,
# minutes), robust exclusion margins from the stored timing calls (margins.py, after
# csirs.json; nothing is timed) and the coverage of the two intervals under a crossed
# seed-by-block model (calib.py, about an hour of CPU time); rules R12 and R13.  Each
# refuses to overwrite its output; they are skipped unless SCREEN=1.
if [ "${SCREEN:-0}" = 1 ]; then python3 csirs.py; python3 margins.py; python3 calib.py; fi

# The paper's final tests at the levels Holm's procedure uses, simulated under the crossed
# seed-by-block model with each scenario's own blocks (calib_holm.py, about an hour on 24
# CPU workers); rules R14 and R15.  It refuses to overwrite its outputs and is skipped
# unless CALIB=1.
if [ "${CALIB:-0}" = 1 ]; then python3 calib_holm.py; fi

# The CSI-RS sweep as NR configures it, six one-slot resource sets and their L1-RSRP
# reports compared at 1 dB (csirs_cfg.py, GPU, minutes), and simultaneous lower bounds on
# the fast references' efficiencies (refbounds.py, GPU, a minute); rules R16 and R17.  Each
# refuses to overwrite its output; both are skipped unless NRCFG=1.
if [ "${NRCFG:-0}" = 1 ]; then [ -f csirs_cfg.json ] || python3 csirs_cfg.py; [ -f refbounds.json ] || python3 refbounds.py; fi

# The CSI-RS report timing under both computation-time conditions of TS 38.214 5.4
# (nrtime.py), and levels.py: the bounds at the level each claim needs, R12 at the
# corrected thresholds, eight resources per slot and the bootstrap's tails block by block;
# rules R18 and R19.  levels.py refuses to overwrite its output and is skipped unless
# LEVELS=1.
if [ "${LEVELS:-0}" = 1 ]; then [ -f levels.json ] || python3 levels.py; fi

# The three decisive implementations timed again under recorded conditions (rule R20).
# The campaign runs only with TRACE=1, on a quiet machine, and never twice: its outputs are
# written once.
if [ "${TRACE:-0}" = 1 ]; then [ -d ../latency/trace_decisive ] || ../latency/run_trace_decisive.sh; fi

# Rule R21: Table I and every number computed from it read every runtime measured, the
# stored timings and the repeated runs.  reconcile.py applies it; nothing is timed, and it
# refuses to overwrite its output.
[ -f reconcile.json ] || python3 reconcile.py

# The margins as rules R16, R19 and R21 read them: every CSI-RS threshold, margin and delay limit
# with nrtime.py's report times, and each exclusion's excess under aging, at reconcile.py's
# runtimes.  margins_post.py times nothing and refuses to overwrite its output.
[ -f margins_post.json ] || python3 margins_post.py

# The matched profiling study (rule R22): an image-only ViT-B/16 on the GPU, FP32 against the
# FP16 build its pilot chose.  The pilot, the freeze of its choices into protocol.json and the
# three sessions run only with MATCHED=1, on a quiet machine, and never twice; matched.py reads
# the sessions whenever its output is missing.
if [ "${MATCHED:-0}" = 1 ] && [ ! -d ../latency/profile_matched ]; then
  ../latency/run_profile_matched.sh pilot
  python3 matched.py --freeze
  ../latency/run_profile_matched.sh main
fi
[ -f matched.json ] || python3 matched.py

# Rule R23, set post hoc: the matched study's readings restated, and calculations on its records
# added after its outputs (per-session boundaries, uniform scaling factors).  matched_post.py
# times nothing and refuses to overwrite its output.
[ -f matched_post.json ] || python3 matched_post.py

# The cadence study (rule R24): the matched study's two builds at three invocation spacings,
# 18 sessions in an order cadence.py --freeze fixed before the first ran.  They run only with
# CADENCE=1, on a quiet machine, and never twice; cadence.py reads them whenever its output is
# missing.
if [ "${CADENCE:-0}" = 1 ] && [ ! -d ../latency/profile_cadence ]; then
  python3 cadence.py --freeze
  ../latency/run_profile_cadence.sh main
fi
[ -f cadence.json ] || python3 cadence.py

# Rule R25: the calibration audit of the sweep-efficiency bound, by repeated sampling of spatial
# blocks from the six evaluation scenarios.  calib_audit.py re-simulates the sweep's selections on
# the GPU, times nothing and refuses to overwrite its output.
[ -f calib_audit.json ] || python3 calib_audit.py

python3 make_manifest.py       # -> manifest.json: the datasets', models' and environment's hashes
