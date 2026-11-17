"""Fetch an author Google Drive release in small, validated, resumable ranges.

The public download domain is the one used by Google's confirmation page. Each
response must match its exact requested range. Completed chunks have hashes;
the assembled gzip/tar stream must pass gzip CRC before it is marked complete.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def audit_tar_gzip(path):
    """Inspect metadata and consume the entire gzip stream to verify its CRC."""
    members = []
    with gzip.open(path, 'rb') as gz:
        with tarfile.open(fileobj=gz, mode='r|') as archive:
            for member in archive:
                if member.isfile():
                    members.append({'name': member.name, 'bytes': member.size})
        while gz.read(8*1024**2):
            pass
    return {'member_count': len(members), 'uncompressed_bytes': sum(m['bytes'] for m in members),
            'gzip_crc_verified': True, 'members': members}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', choices=['joshi'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--chunk-mib', type=int, default=8)
    a = p.parse_args()
    assert 1 <= a.workers <= 8 and 1 <= a.chunk_mib <= 16
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/sources.json').read_text(encoding='utf-8'))[a.source]
    file_id = parse_qs(urlparse(config['url']).query)['id'][0]
    url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    raw = a.root/'data/raw'
    chunk_dir = raw/'chunks'/config['filename']
    chunk_dir.mkdir(parents=True, exist_ok=True)
    target = raw/config['filename']
    status = a.root/'artifacts'/f'{a.source}_download.json'
    status.parent.mkdir(parents=True, exist_ok=True)
    info = {'source': a.source, 'config': config, 'download_endpoint': url,
            'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
            'method': 'validated_small_http_ranges', 'pid': os.getpid()}
    started = time.perf_counter()

    def save():
        info['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = status.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(info, indent=2)+'\n', encoding='utf-8')
        tmp.replace(status)

    save()
    try:
        with requests.get(url, headers={'Range':'bytes=0-0'}, timeout=(15, 30)) as r:
            assert r.status_code == 206, f'Expected range response, got HTTP {r.status_code}'
            match = re.fullmatch(r'bytes 0-0/(\d+)', r.headers.get('Content-Range',''))
            assert match, 'Invalid probe Content-Range'
            total = int(match.group(1))
            assert len(r.content) == 1
            info['release_response_headers'] = {k:r.headers[k] for k in ('ETag','Last-Modified') if k in r.headers}
        info['total_bytes'] = total
        info['chunk_bytes'] = step = a.chunk_mib*1024**2
        assert shutil.disk_usage(a.root).free > total*2 + 1024**3, 'Insufficient room for chunks plus assembled archive'
        pairs = [(start, min(start+step,total)-1) for start in range(0,total,step)]
        info['total_chunks'] = len(pairs)
        save()

        def fetch(pair):
            start, end = pair
            size = end-start+1
            path = chunk_dir/f'{start:012d}-{end:012d}.chunk'
            side = path.with_suffix('.sha256')
            if path.exists() and path.stat().st_size == size and side.exists() and digest(path) == side.read_text(encoding='ascii').strip():
                return size
            for attempt in range(1,9):
                try:
                    with requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=(15,40)) as r:
                        assert r.status_code == 206, f'HTTP {r.status_code}'
                        assert r.headers.get('Content-Range') == f'bytes {start}-{end}/{total}', 'Range mismatch'
                        content = r.content
                        assert len(content) == size, 'Truncated range'
                    temp = path.with_suffix('.tmp')
                    temp.write_bytes(content)
                    temp.replace(path)
                    side.write_text(hashlib.sha256(content).hexdigest()+'\n', encoding='ascii')
                    return size
                except Exception as e:
                    print(f'Range {start}-{end}, attempt {attempt}: {type(e).__name__}', flush=True)
                    if attempt == 8:
                        raise
                    time.sleep(min(10,attempt*2))

        done = count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(fetch,pair) for pair in pairs]):
                done += future.result()
                count += 1
                info.update(downloaded_chunk_bytes=done, completed_chunks=count)
                save()
                print(f'Validated {count}/{len(pairs)} chunks, {done}/{total} bytes', flush=True)
        info['phase'] = 'assembling_and_crc_audit'
        save()
        assembled = target.with_suffix(target.suffix+'.assembling')
        with assembled.open('wb') as out:
            for start,end in pairs:
                with (chunk_dir/f'{start:012d}-{end:012d}.chunk').open('rb') as part:
                    shutil.copyfileobj(part,out,8*1024**2)
        assert assembled.stat().st_size == total
        sha = digest(assembled)
        inventory = audit_tar_gzip(assembled)
        assembled.replace(target)
        info.update(status='completed_verified', phase='complete', path=str(target.resolve()),
                    bytes=total, sha256=sha, archive=inventory)
    except BaseException as e:
        info.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed', error=repr(e))
        raise
    finally:
        info['elapsed_seconds'] = time.perf_counter()-started
        save()
        print(json.dumps({k:v for k,v in info.items() if k!='archive'},indent=2),flush=True)


if __name__ == '__main__':
    main()
