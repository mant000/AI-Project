#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/tsp-state-value
while tmux has-session -t tsp-milestone-timing-figures 2>/dev/null; do sleep 10; done
test -f "$ROOT/code/results/milestone-fragment-expression-v1/timing/completed.json"
export GIT_TERMINAL_PROMPT=0 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
unset GIT_ASKPASS SSH_ASKPASS
DEST="$ROOT/operations/milestone-anonymous-clone-v1"
git -c credential.helper= clone --quiet --depth 1 https://github.com/mant000/AI-Project.git "$DEST"
cd "$DEST"
test "$(git rev-parse HEAD)" = 110fba1ed847509910f74ebaa4c2053ac76c4282
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="$ROOT/envs/milestone-cpu-repro/bin/python"
OUT="$ROOT/operations/milestone-cpu-reproduction-v1"
mkdir -p "$OUT"
"$PY" -m pip check > "$OUT/pip_check.txt"
"$PY" -m unittest discover -s tests -p test_fragment_core.py > "$OUT/tests.txt" 2>&1
"$PY" scripts/milestone_analyze.py --output "$OUT/analysis" > "$OUT/analysis.log" 2>&1
"$PY" scripts/milestone_replay.py --workers 10 --output "$OUT/replay" > "$OUT/replay.log" 2>&1
printf '%s\n' 'Anonymous clone, separate CPU environment analysis and solver replay passed'
"$PY" scripts/milestone_release_audit.py --analysis-proof "$OUT/analysis/analysis_verification.json" --replay-proof "$OUT/replay/replay_verification.json" --output results/milestone-reproduction-v1 > "$OUT/release_audit.log" 2>&1
cp "$OUT/tests.txt" "$OUT/pip_check.txt" results/milestone-reproduction-v1/