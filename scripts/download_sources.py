"""Download fixed author releases or Qwen weights; record status and file hashes.

Run each source in a separate tmux job. A failed or incomplete download is never
reported as verified. Archives are inspected without executing author code.
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    """Hash a large file with bounded memory."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_archive(path):
    """List archive metadata only; extraction and instance parsing are stage 1."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            rows = [{'name': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size}
                    for x in z.infolist() if not x.is_dir()]
    else:
        with tarfile.open(path, 'r:*') as t:
            rows = [{'name': x.name, 'bytes': x.size} for x in t if x.isfile()]
    return {'member_count': len(rows), 'uncompressed_bytes': sum(x['bytes'] for x in rows), 'members': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', choices=['joshi', 'invit', 'qwen'])
    p.add_argument('--root', type=Path, default=Path(os.getenv('TSP_ROOT', '/root/autodl-tmp/tsp-state-value')))
    p.add_argument('--file', type=Path, help='Audit an already downloaded author archive')
    p.add_argument('--attempts', type=int, default=3, help='Bounded resumable retries for Drive downloads')
    a = p.parse_args()
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/sources.json').read_text())[a.source]
    dest = a.root / 'data/raw'
    dest.mkdir(parents=True, exist_ok=True)
    artifacts = a.root / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    status_path = artifacts / (a.source + '_download.json')
    info = {'source': a.source, 'config': config, 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
            'download_endpoint': os.getenv('HF_ENDPOINT', 'https://huggingface.co') if a.source == 'qwen' else config['url']}

    def save():
        tmp = status_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(info, indent=2) + '\n')
        tmp.replace(status_path)

    save()
    start = time.perf_counter()
    try:
        if a.source == 'qwen':
            from huggingface_hub import snapshot_download
            # Avoid .bin, GGUF, and other duplicate formats. Hub cache holds one copy.
            assert shutil.disk_usage(a.root).free > 19 * 1024**3, 'Need 19 GiB free before 7B download'
            target = Path(snapshot_download(repo_id=config['repository'], revision=config['revision'],
                            allow_patterns=['*.json','*.safetensors','*.txt','*.md','LICENSE'], max_workers=2))
            files = [{'name': str(f.relative_to(target)), 'bytes': f.stat().st_size, 'sha256': sha256(f)}
                     for f in sorted(target.rglob('*')) if f.is_file()]
            assert any(x['name'].endswith('.safetensors') for x in files)
            reference = json.loads((Path(__file__).resolve().parents[1] / 'configs/qwen_weight_hashes.json').read_text())
            expected = {x['rfilename']: x['lfs']['sha256'] for x in reference['safetensors']}
            actual = {x['name']: x['sha256'] for x in files if x['name'].endswith('.safetensors')}
            assert actual == expected, 'Weights do not match author-published LFS SHA256 hashes'
            info['author_weight_hashes_verified'] = True
            info.update(snapshot_path=str(target), files=files)
        else:
            target = a.file or dest / config['filename']
            if not a.file and not target.exists():
                import gdown
                import requests
                assert shutil.disk_usage(a.root).free > 6 * 1024**3, 'Need 6 GiB free for archive download'
                original_request = requests.sessions.Session.request

                def bounded_request(session, method, url, **kwargs):
                    kwargs.setdefault('timeout', (15, 45))
                    response = original_request(session, method, url, **kwargs)
                    requested = kwargs.get('headers', {}).get('Range')
                    if requested:
                        start_byte = requested.split('=')[1].split('-')[0]
                        if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith('bytes ' + start_byte + '-'):
                            response.close()
                            raise RuntimeError('Server did not honor resume Range; refused to append')
                    return response

                requests.sessions.Session.request = bounded_request
                try:
                    for attempt in range(1, a.attempts + 1):
                        info['attempt'] = attempt
                        save()
                        try:
                            print(f'Download {a.source}, attempt {attempt}', flush=True)
                            gdown.download(config['url'], str(target), quiet=True, resume=True, use_cookies=False)
                            if target.exists():
                                break
                        except Exception as e:
                            info['last_retry_error'] = repr(e)
                            info['partial_bytes'] = sum(f.stat().st_size for f in dest.glob(target.name + '*.part'))
                            save()
                            print(f'Retry after {type(e).__name__}; partial_bytes={info["partial_bytes"]}', flush=True)
                            if 'Too many users' in str(e) or 'quota' in str(e).lower():
                                info['blocking_reason'] = 'Author Google Drive download quota; stop retrying and check later'
                                raise
                            if attempt == a.attempts:
                                raise
                            time.sleep(min(30, 3 * attempt))
                finally:
                    requests.sessions.Session.request = original_request
            assert target.exists(), 'Download did not create archive'
            info.update(path=str(target.resolve()), bytes=target.stat().st_size, sha256=sha256(target),
                        archive=inspect_archive(target))
        info['status'] = 'completed_verified'
    except BaseException as e:
        info.update(status='interrupted' if isinstance(e, KeyboardInterrupt) else 'failed', error=repr(e))
        raise
    finally:
        info['elapsed_seconds'] = time.perf_counter() - start
        info['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({k:v for k,v in info.items() if k not in ('archive','files')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
