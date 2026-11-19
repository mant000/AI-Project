"""Download or verify the pinned Qwen snapshot used by the frozen generator."""
import argparse
import hashlib
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--snapshot', type=Path, help='Register existing local safetensors; no download')
    ap.add_argument('--model', help='Hugging Face repository; defaults to the reported Qwen model')
    ap.add_argument('--revision', help='Pinned model revision; required for a different repository')
    ap.add_argument('--cache-dir', type=Path, help='Hub cache; defaults to ROOT/cache/huggingface/hub')
    a = ap.parse_args()
    reported = json.loads((CODE/'configs/sources.json').read_text())['qwen']
    cfg = dict(reported)
    if a.model and a.model != reported['repository']:
        if not a.revision:
            ap.error('--revision is required with a different --model')
        cfg = dict(repository=a.model, revision=a.revision)
    elif a.revision:
        cfg['revision'] = a.revision
    hashes = json.loads((CODE/'configs/qwen_weight_hashes.json').read_text())
    if a.snapshot:
        snapshot = a.snapshot.resolve()
    else:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(repo_id=cfg['repository'], revision=cfg['revision'],
            cache_dir=str(a.cache_dir or a.root/'cache/huggingface/hub'),
            allow_patterns=['*.json','*.safetensors','*.txt','*.md','LICENSE'], max_workers=2))
    assert (snapshot/'config.json').is_file(), 'Snapshot must include config.json'
    assert any(snapshot.glob('*.safetensors')), 'Snapshot has no safetensors weights'
    for index in snapshot.glob('*.safetensors.index.json'):
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            assert (snapshot/name).is_file(), f'Missing weight shard: {name}'
    files = [dict(name=str(p.relative_to(snapshot)),bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(snapshot.rglob('*')) if p.is_file()]
    expected = {r['rfilename']:r['lfs']['sha256'] for r in hashes['safetensors']}
    author_verified = (cfg['repository'],cfg['revision']) == (reported['repository'],reported['revision'])
    if author_verified:
        assert {r['name']:r['sha256'] for r in files if r['name'].endswith('.safetensors')} == expected
    target = a.root/'artifacts/model.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(status='completed',config=cfg,snapshot_path=str(snapshot),
        files=files,author_weight_hashes_verified=author_verified),indent=2)+'\n')
    print('Model snapshot:',snapshot)
    print('Manifest:',target)

if __name__ == '__main__':main()
