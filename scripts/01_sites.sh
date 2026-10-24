#!/usr/bin/env bash
# Stage 1 -- acquire and qualify the DeepMIMO city sites.
# Runs in the DeepMIMO loader environment (src/sites/requirements-pinned.txt),
# NOT the environment the rest of the pipeline uses: the loader needs numpy>=2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../src/sites"
python3 acquire.py        # download + extract the 28 GHz city scenarios
python3 qualify.py        # outcome-blind qualification -> qualification.json
python3 fixtures.py       # channel-unit fixtures -> fixtures_city_*.json
python3 make_manifest.py  # catalog cross-check -> site-manifest.json
