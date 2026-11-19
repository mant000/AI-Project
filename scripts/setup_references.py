"""Portable pinned reference setup; reuse existing verified assets when present.

Source and model bytes are checked against the previously frozen download
manifest. No historical raw metadata are overwritten and no datasets downloaded.
"""
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request
from experiment_io import CODE,read,write,digest,stamp


def fetch(url,path,sha):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(url,timeout=60) as f:data=f.read()
        assert hashlib.sha256(data).hexdigest()==sha,url
        path.write_bytes(data)
    assert digest(path)==sha,path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();start=time.perf_counter()
    old=read(CODE/'configs/reference_asset_hashes.json');cfg=read(CODE/'configs/reference_assets.json')
    meta=a.root/'operations/reference-assets';meta.mkdir(parents=True,exist_ok=True)
    lehd=[]
    prefix='/single_objective/LEHD/'
    for record in old['lehd_files']:
        relative=record['url'].split(prefix,1)[1];path=a.root/'external/lehd'/relative
        fetch(record['url'],path,record['sha256']);lehd.append({**record,'path':str(path)})
    write(meta/'lehd.json',dict(files=lehd,commit=cfg['lehd']['commit']))
    source=a.root/'external/lkh3';archive=a.root/'external/lkh-source.tar.gz'
    fetch(f"https://codeload.github.com/cerebis/LKH3/tar.gz/{cfg['lkh']['mirror_commit']}",archive,old['lkh_mirror_archive_sha256'])
    expected={r['path']:r['sha256'] for r in old['lkh_source']};source.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            relative=Path(*Path(member.name).parts[1:]);name=relative.as_posix()
            if name not in expected:continue
            assert member.isfile() and not member.issym() and '..' not in relative.parts
            data=tf.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==expected[name]
            dst=source/relative
            if dst.exists():assert digest(dst)==expected[name]
            else:dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
    binary=source/'LKH'
    if not binary.exists():
        proc=subprocess.run(cfg['lkh']['compile'],cwd=source,capture_output=True,text=True)
        (meta/'lkh_build_stdout.txt').write_text(proc.stdout);(meta/'lkh_build_stderr.txt').write_text(proc.stderr)
        assert proc.returncode==0,proc.stderr
    for relative,sha in expected.items():assert digest(source/relative)==sha
    write(meta/'lkh.json',dict(status='completed',at_utc=stamp(),milestone_portable_setup=True,
          lkh_binary=str(binary),lkh_binary_sha256=digest(binary),lkh_mirror_commit=old['lkh_mirror_commit'],
          source_matches_frozen_manifest=True,lehd_files=lehd,source_manifest_sha256=digest(CODE/'configs/reference_asset_hashes.json'),
          wall_seconds=time.perf_counter()-start,source_sha256=digest(Path(__file__))))
    print('Pinned reference assets verified at',a.root)


if __name__=='__main__':main()
