#!/usr/bin/env bash
# Fresh-source replay against existing verified inputs, with new result files.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE"
export TSP_ROOT="${1:?verified project root required}"
OUT="${2:?new replay output directory required}"
source scripts/env.sh
mkdir -p "$OUT"
python -m pytest tests/test_solver.py -q --junitxml="$OUT/correctness.xml"
python scripts/evaluate_baselines.py --root "$TSP_ROOT" --output "$OUT"
python scripts/compare_baselines.py results/stage1/baseline_runs.jsonl \
    "$OUT/baseline_runs.jsonl" --output "$OUT/comparison.json"
