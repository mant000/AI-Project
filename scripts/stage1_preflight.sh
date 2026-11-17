#!/usr/bin/env bash
# This is an executable environment/data gate, not a claimed solver implementation.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
source "$CODE/scripts/env.sh"
python - "$TSP_ROOT" "${1:-full}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
probe = json.loads((root/'artifacts/system_probe.json').read_text())
assert probe['cuda_smoke']['success'], 'CUDA smoke has not passed'
assert all(m['verified_mount'] for m in probe['mounts'][1:]), 'Storage not verified'
if sys.argv[2] == 'bootstrap':
    a = json.loads((root/'data/bootstrap/bootstrap_audit.json').read_text())
    assert a['status'] == 'bootstrap_rows_validated' and a['original_split'] == 'train'
    data = pathlib.Path(a['data_file'])
    assert hashlib.sha256(data.read_bytes()).hexdigest() == a['data_sha256']
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    assert len(rows) == a['instance_count']
    assert len({x['coordinate_set_sha256'] for x in rows}) == len(rows)
    print('Stage 1 bootstrap READY:',len(rows),'complete author-train TSP50 instances')
    print('Scope: solver implementation/engineering checks only; full train/val/TSP100 split remains pending.')
    print('Full author archive SHA256 and gzip CRC are NOT verified in bootstrap mode.')
    raise SystemExit(0)
assert sys.argv[2] == 'full', 'Usage: stage1_preflight.sh [full|bootstrap]'
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
