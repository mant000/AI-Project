"""Archive the completed common SFT without touching concurrently running RL checkpoints."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone


def stamp(): return datetime.now(timezone.utc).isoformat()


def read(path): return json.loads(Path(path).read_text())


def write(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''): value.update(chunk)
    return value.hexdigest()


def verify(archive, destination):
    assert not destination.exists(); destination.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers(); meta = json.load(tf.extractfile('sft_backup_manifest.json'))
        for member in members:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            target = destination/path; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tf.extractfile(member).read())
    for name, expected in meta['files_sha256'].items(): assert digest(destination/name) == expected, name
    return dict(status='passed', at_utc=stamp(), archive_sha256=digest(archive), files_verified=len(meta['files_sha256']),
        selected_adapter_sha256=meta['selected_adapter_sha256'], restore_path=str(destination), scope=meta['scope'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path); ap.add_argument('--archive', type=Path)
    ap.add_argument('--restore', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
    if a.root:
        root = a.root; completed = read(root/'code/results/training/sft-v1/verification.json'); assert completed['status'] == 'passed'
        prefix = ['code/scripts', 'code/src/tsp_value', 'code/configs', 'code/results/training/sft-v1',
            'code/results/training/sft-analysis-v1', 'code/results/training/grpo-prepare-v1', 'code/results/main/prepare-v1',
            'checkpoints/sft-v1']
        files = set()
        for name in prefix:
            files.update(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
        files.update(root/'code'/name for name in ['requirements.txt', 'requirements-training.txt', 'docs/COMMON_SFT_RESULTS.md',
            'docs/R3_PROTOCOL_CN.md', 'docs/R3_REPETITION_AMENDMENT.json'])
        files.add(root/'operations/r3-code.bundle')
        manifest = dict(at_utc=stamp(), scope='completed common SFT + observable banks only; active GRPO excluded',
            selected_adapter_sha256=read(root/'code/results/training/sft-v1/checkpoint_frozen.json')['adapter_sha256'],
            files_sha256={str(p.relative_to(root)): digest(p) for p in sorted(files)})
        archive = a.archive
        if archive is None:
            name = datetime.now(timezone.utc).strftime('common-sft-%Y%m%dT%H%M%SZ.tar.gz')
            archive = Path('/root/autodl-fs/tsp-state-value/backups')/name
        assert not archive.exists()
        with tarfile.open(archive, 'w:gz') as tf:
            raw = (json.dumps(manifest, indent=2)+'\n').encode(); info = tarfile.TarInfo('sft_backup_manifest.json'); info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))
            for p in sorted(files): tf.add(p, arcname=str(p.relative_to(root)), recursive=False)
        for p in files: assert digest(p) == manifest['files_sha256'][str(p.relative_to(root))], 'Source changed during archive'
    else:
        archive = a.archive; assert archive is not None
    result = verify(archive, a.restore); result.update(archive_path=str(archive), archive_bytes=archive.stat().st_size)
    if a.root:
        import torch
        state = torch.load(a.restore/'checkpoints/sft-v1/last.pt', map_location='cpu')
        assert len(state['optimizer']['state']) == 112 and state['progress']['logical_updates'] == 154
        result.update(restored_populated_optimizer_states=112, restored_logical_updates=154)
    write(a.output, result); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
