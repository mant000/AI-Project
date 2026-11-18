#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/tsp-state-value/code
source scripts/env.sh
while tmux has-session -t tsp-solver-controls 2>/dev/null; do sleep 20; done
test -f results/eoh-control-v1/completed.json
python scripts/solver_dev_benchmark.py --root "$TSP_ROOT" > "$TSP_ROOT/logs/solver-dev-benchmark-v1.log" 2>&1
