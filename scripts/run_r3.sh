#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/tsp-state-value/code
source scripts/env.sh
# Wait for the confirmed existing SFT tmux job; never restart it on an observation timeout.
while tmux has-session -t tsp-common-sft 2>/dev/null; do sleep 15; done
test -f results/training/sft-v1/completed.json
python scripts/common_sft_verify.py --root "$TSP_ROOT" > "$TSP_ROOT/logs/common-sft-verify-v1.log" 2>&1
for optseed in 701 702 703; do
    for rewardarm in terminal state; do
        python scripts/controlled_grpo.py --root "$TSP_ROOT" --arm "$rewardarm" --seed "$optseed" > "$TSP_ROOT/logs/grpo-${rewardarm}-${optseed}-v1.log" 2>&1
    done
done
