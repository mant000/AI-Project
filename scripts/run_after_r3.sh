#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/tsp-state-value/code
source scripts/env.sh
# Dependent work is allowed only after the observed R3 tmux job ends successfully.
while tmux has-session -t tsp-r3-runner 2>/dev/null; do sleep 20; done
for optseed in 701 702 703; do
    for rewardarm in terminal state; do
        test -f "results/training/grpo-v1/${rewardarm}-seed${optseed}/completed.json"
    done
done
python scripts/grpo_verify.py --root "$TSP_ROOT" > "$TSP_ROOT/logs/grpo-verify-v1.log" 2>&1
python scripts/main_control.py --root "$TSP_ROOT" > "$TSP_ROOT/logs/main-control-v1.log" 2>&1
