#!/usr/bin/env bash
# Fixed, measured stage1 run. Does not install packages or download sources.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE"
source scripts/env.sh
mkdir -p results/stage1
bash scripts/stage1_preflight.sh full
python scripts/build_manifest.py --root "$TSP_ROOT" --tsplib-archive "$TSP_ROOT/data/raw/ALL_tsp.tar.gz"
python -m pytest tests/test_solver.py -q --junitxml=results/stage1/correctness.xml
python scripts/evaluate_baselines.py --root "$TSP_ROOT"
