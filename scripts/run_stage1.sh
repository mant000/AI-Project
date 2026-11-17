#!/usr/bin/env bash
# Fixed, measured stage1 run. Does not install packages or download sources.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE"
source scripts/env.sh
mkdir -p results/stage1
bash scripts/stage1_preflight.sh full
if [[ "${1:-}" == --reuse-manifest ]]; then
    python - "$TSP_ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
audit=json.loads(pathlib.Path('results/stage1/manifest_audit.json').read_text())
assert audit['status']=='passed'
for path,key in [(pathlib.Path('splits/manifest.csv'),'manifest_sha256'),
                 (root/'data/prepared/stage1/project_instances.jsonl','project_instances_sha256')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==audit[key]
print('Reusing hash-verified stage1 instance manifest; no source preparation repeated')
PY
else
    python scripts/build_manifest.py --root "$TSP_ROOT" --tsplib-archive "$TSP_ROOT/data/raw/ALL_tsp.tar.gz"
fi
python -m pytest tests/test_solver.py -q --junitxml=results/stage1/correctness.xml
python scripts/evaluate_baselines.py --root "$TSP_ROOT"
