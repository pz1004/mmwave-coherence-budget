#!/usr/bin/env bash
# Stage 3 -- the accuracy-free compute budget and the T* crossover (closed form).
# Cheap: no data, no GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../src/budget"
python3 theory.py         # tau* = (M-K)t_p, T*, the upper convex hull envelope
python3 dp.py             # exact DP reference for the structural claims
python3 verify.py         # independent re-derivation -> verification.json
