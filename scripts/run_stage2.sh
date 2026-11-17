#!/usr/bin/env bash
# Replay the frozen stage2 protocol, preserving all attempts in a fresh directory.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env.sh"
cd "$CODE"
OUT="${1:?Supply a new results directory}"
CKPT="${2:?Supply a new checkpoint directory}"
[[ ! -e "$OUT" && ! -e "$CKPT" ]] || { echo 'Output/checkpoint directory already exists'; exit 1; }
mkdir -p "$OUT"
python -m pytest tests/test_solver.py -q
for spec in diagnosis-v1:stage2_diagnosis diagnosis-v2:stage2_diagnosis_v2 diagnosis-v3:stage2_diagnosis_v3 pilot:stage2_pilot; do
    python scripts/stage2_labels.py --root "$TSP_ROOT" --config "configs/${spec#*:}.json" --output "$OUT/${spec%:*}"
done
python scripts/stage2_labels.py --root "$TSP_ROOT" --config configs/stage2_train_labels.json --reuse "$OUT/pilot/states.jsonl.gz" --output "$OUT/train"
python scripts/stage2_labels.py --root "$TSP_ROOT" --config configs/stage2_val_labels.json --output "$OUT/val"
python scripts/stage2_train.py --train "$OUT/train/states.jsonl.gz" --val "$OUT/val/states.jsonl.gz" --config configs/stage2_training.json --output "$OUT/training-v1" --checkpoints "$CKPT"
python scripts/stage2_replay.py --dataset "$OUT/val/states.jsonl.gz" --training "$OUT/training-v1" --output "$OUT/action-replay"
python scripts/stage2_plot.py --results "$OUT"
