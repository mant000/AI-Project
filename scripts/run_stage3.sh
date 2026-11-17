#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/stage3_discover.py --root "$TSP_ROOT" --prepare "${1:-results/stage3/prepare-v1}" --output "${2:-results/stage3/discovery-v1}"
