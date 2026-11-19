"""Download or verify the pinned Qwen snapshot used by the frozen generator."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

CODE = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads((CODE/'configs/sources.json').read_text())['qwen']
    hashes = json.loads((CODE/'configs/qwen_weight_hashes.json').read_text())
    snapshot = Path(snapshot_download(repo_id=cfg['repository'], revision=cfg['revision'],
        allow_patterns=['*.json','*.safetensors','*.txt','*.md','LICENSE'], max_workers=2))
    files = [dict(name=str(p.relative_to(snapshot)),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(snapshot.rglob('*')) if p.is_file()]
    expected = {r['rfilename']:r['lfs']['sha256'] for r in hashes['safetensors']}
    assert {r['name']:r['sha256'] for r in files if r['name'].endswith('.safetensors')} == expected
    target = a.root/'artifacts/qwen_download.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(status='completed',config=cfg,snapshot_path=str(snapshot),
        files=files,author_weight_hashes_verified=True),indent=2)+'\n')
    print('Verified pinned model at',snapshot)

if __name__ == '__main__':main()
