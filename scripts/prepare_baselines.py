"""Pinned author LEHD files/weights and original LKH source mirror, no benchmark data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request
from stage3_common import CODE, read, write, append, digest, stamp


def fetch(url, path, log, expected_blob=None, expected_bytes=None):
    def check(raw):
        if expected_bytes is not None: assert len(raw)==expected_bytes
        if expected_blob is not None:
            actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();assert actual==expected_blob,(url,actual,expected_blob)
    if path.exists():
        raw=path.read_bytes();check(raw);return raw
    for attempt in range(3):
        begin=time.perf_counter()
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'TSP-course-research/1.0'})
            with urllib.request.urlopen(req,timeout=20) as f:raw=f.read()
            check(raw);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(raw);tmp.replace(path)
            append(log,dict(at_utc=stamp(),url=url,attempt=attempt,status='downloaded',bytes=len(raw),wall_seconds=time.perf_counter()-begin,sha256=digest(path)))
            return raw
        except Exception as exc:
            append(log,dict(at_utc=stamp(),url=url,attempt=attempt,error=repr(exc),wall_seconds=time.perf_counter()-begin))
            if attempt==2:raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    out=CODE/'results/baselines/prepare-v1';out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists()
    cfgpath=CODE/'configs/external_baselines_v1.json';cfg=read(cfgpath);log=out/'download_attempts.jsonl';start=time.perf_counter()
    ext=a.root/'external';ext.mkdir(exist_ok=True)
    write(out/'protocol_frozen.json',dict(at_utc=stamp(),protocol=cfg,protocol_sha256=digest(cfgpath),source_sha256=digest(Path(__file__))))
    lehd=cfg['lehd'];repo=ext/'lehd';repo.mkdir(exist_ok=True)
    treeurl=f'https://api.github.com/repos/CIAM-Group/NCO_code/git/trees/{lehd["commit"]}?recursive=1'
    tree=json.loads(fetch(treeurl,out/'lehd_tree.json',log));blobs={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    records=[]
    for name in lehd['files']+[lehd['checkpoint']]:
        full=lehd['prefix']+name;blob=blobs[full]
        url=f'https://raw.githubusercontent.com/CIAM-Group/NCO_code/{lehd["commit"]}/{full}'
        path=repo/name;fetch(url,path,log,blob['sha'],blob['size'])
        records.append(dict(path=str(path),url=url,sha256=digest(path),git_blob_sha1=blob['sha'],bytes=path.stat().st_size))
    write(out/'lehd_completed.json',dict(at_utc=stamp(),commit=lehd['commit'],files=records,baseline_data_downloaded=False))
    lkh=cfg['lkh'];archive=ext/'lkh-source.tar.gz'
    # The checked mirror is used consistently; official failures are retained as provenance.
    try:
        fetch(lkh['official_download_attempt'],ext/'lkh-official-probe.tgz',log)
        official_available=True
    except Exception:official_available=False
    url=f'https://codeload.github.com/cerebis/LKH3/tar.gz/{lkh["mirror_commit"]}'
    fetch(url,archive,log);dest=ext/'lkh3';dest.mkdir(exist_ok=True);source=[]
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            relative=Path(*Path(member.name).parts[1:])
            if not member.isfile() or not relative.parts:continue
            if relative.as_posix() not in ('Makefile','README.txt') and relative.parts[0]!='SRC':continue
            assert not relative.is_absolute() and '..' not in relative.parts and not member.issym()
            path=dest/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tf.extractfile(member).read())
            source.append(dict(path=relative.as_posix(),sha256=digest(path)))
    proc=subprocess.run(lkh['compile'],cwd=dest,capture_output=True,text=True)
    (out/'lkh_build_stdout.txt').write_text(proc.stdout);(out/'lkh_build_stderr.txt').write_text(proc.stderr)
    assert proc.returncode==0,proc.stderr[-2000:]
    result=dict(status='completed',at_utc=stamp(),lehd_files=records,lkh_source=source,lkh_mirror_commit=lkh['mirror_commit'],
        official_archive_available_on_probe=official_available,official_archive_used=False,lkh_mirror_archive_sha256=digest(archive),
        lkh_binary=str(dest/'LKH'),lkh_binary_sha256=digest(dest/'LKH'),compile_argv=lkh['compile'],
        compiler=subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],wall_seconds=time.perf_counter()-start,
        no_environment_packages_installed=True,final_opened=False)
    write(out/'completed.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('lehd_files','lkh_source')},indent=2))


if __name__=='__main__':main()
