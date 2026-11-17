#!/usr/bin/env bash
# This is an executable environment/data gate, not a claimed solver implementation.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env.sh"
python - "$TSP_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
probe = json.loads((root/'artifacts/system_probe.json').read_text())
assert probe['cuda_smoke']['success'], 'CUDA smoke has not passed'
assert all(m['verified_mount'] for m in probe['mounts'][1:]), 'Storage not verified'
p = root/'artifacts/joshi_download.json'
assert p.exists(), 'No Joshi download record; inspect logs before launching another job'
d = json.loads(p.read_text())
assert d['status'] == 'completed_verified', f"Joshi download is {d['status']}: {d.get('error', '')}"
assert pathlib.Path(d['path']).exists(), 'Verified archive missing'
print('Stage 1 data/solver implementation may proceed.')
print('Verified author archive:', d['path'])
print('Archive SHA256:', d['sha256'])
print('Do not use author test / INViT / TSPLIB for milestone tuning.')
PY
