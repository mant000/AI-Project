#!/usr/bin/env bash
# Run after copying this repository to the data disk. Does not edit base Python.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env.sh"
mountpoint -q /root/autodl-tmp || { echo 'Data disk is not mounted'; exit 1; }
mkdir -p "$TSP_ROOT"/{envs,cache/pip,cache/huggingface,cache/torch,tmp,data/raw,logs,artifacts,checkpoints}
BASE_PYTHON="${BASE_PYTHON:-/root/miniconda3/bin/python}"
if [[ ! -x "$TSP_ROOT/envs/milestone/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$TSP_ROOT/envs/milestone"
fi
# The venv inherits the image CUDA wheel; overlays install only into the venv.
python -m pip install -r "$CODE/requirements-torch.txt" -r "$CODE/requirements.txt"
command -v tmux >/dev/null || { apt-get update && apt-get install -y tmux; }
python -m pip check
python -m pip freeze > "$TSP_ROOT/artifacts/environment.freeze.txt"
python -m pip list --format=json > "$TSP_ROOT/artifacts/environment.packages.json"
python "$CODE/scripts/system_probe.py" --output "$TSP_ROOT/artifacts/system_probe.json"
echo 'Environment verified. Source scripts/env.sh before subsequent commands.'
