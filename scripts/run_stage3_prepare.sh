#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/stage3_prepare.py --root "$TSP_ROOT" --output "${1:-results/stage3/prepare-v1}"
