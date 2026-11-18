#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/tsp-state-value/code
source scripts/env.sh
while tmux has-session -t tsp-dev-benchmark 2>/dev/null; do sleep 20; done
python scripts/dev_benchmark_verify.py --root /root/autodl-tmp/tsp-state-value
python scripts/dev_benchmark_analyze.py
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python scripts/final_fixture.py
date -Is
