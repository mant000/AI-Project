#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/tsp-state-value/code
source scripts/env.sh
while tmux has-session -t tsp-after-r3 2>/dev/null; do sleep 20; done
test -f results/main/discovery-v1/completed.json
python scripts/eoh_control.py --root "$TSP_ROOT" > "$TSP_ROOT/logs/eoh-control-v1.log" 2>&1
