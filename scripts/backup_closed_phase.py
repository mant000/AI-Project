"""Snapshot completed training and preparation while excluding active program discovery."""
import argparse
import io
import json
from pathlib import Path
import tarfile
from datetime import datetime,timezone
from backup_sft import read,write,digest,stamp


def restore_verify(archive,destination):
    assert not destination.exists();destination.mkdir(parents=True)
    with tarfile.open(archive,'r:gz') as tf:
        manifest=json.load(tf.extractfile('phase_backup_manifest.json'))
        for member in tf.getmembers():
            name=Path(member.name);assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tf.extractfile(member).read())
            target.chmod(member.mode & 0o777)
    for name,expected in manifest['files_sha256'].items():assert digest(destination/name)==expected,name
    return dict(status='passed',at_utc=stamp(),files_verified=len(manifest['files_sha256']),
        archive_sha256=digest(archive),restore_path=str(destination),scope=manifest['scope'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--archive',type=Path)
    ap.add_argument('--restore',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.root:
        root=a.root;code=root/'code'
        assert read(code/'results/training/grpo-v1/verification.json')['status']=='passed'
        assert read(code/'results/history-ablation-v1/verification.json')['status']=='passed'
        prefix=['code','checkpoints','data/prepared/stage1','external/lehd','external/lkh3','logs','operations/core-handoff']
        excluded=['code/.git/','code/results/main/discovery-v1/','code/results/eoh-control-v1/','code/results/solver-controls/dev-v1/']
        active_logs={'main-control-v1.log','after-r3-v1.log','solver-controls-v1.log','eoh-control-v1.log','solver-dev-v1.log'}
        files=[]
        for base in prefix:
            for p in (root/base).rglob('*'):
                name=p.relative_to(root).as_posix()
                if not p.is_file() or any(part in p.parts for part in ('__pycache__','.pytest_cache')) or p.suffix=='.pyc':continue
                if any(name.startswith(e) for e in excluded) or (base=='logs' and p.name in active_logs):continue
                assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve());files.append(p)
        files.extend([root/'artifacts/qwen_download.json',root/'external/lkh-source.tar.gz',root/'operations/r3-r4-prep-code.bundle'])
        files=sorted(set(files));manifest=dict(at_utc=stamp(),scope='closed R3/SFT/history/external preparation; active main/EoH outputs and mutable logs excluded',
            files_sha256={p.relative_to(root).as_posix():digest(p) for p in files},final_opened=False)
        archive=a.archive or Path('/root/autodl-fs/tsp-state-value/backups')/datetime.now(timezone.utc).strftime('r3-closed-%Y%m%dT%H%M%SZ.tar.gz')
        assert not archive.exists()
        with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
            raw=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('phase_backup_manifest.json');info.size=len(raw);tf.addfile(info,io.BytesIO(raw))
            for p in files:tf.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
        for p in files:assert digest(p)==manifest['files_sha256'][p.relative_to(root).as_posix()],p
    else:archive=a.archive
    result=restore_verify(archive,a.restore);result.update(archive_path=str(archive),archive_bytes=archive.stat().st_size)
    if a.root:
        import torch
        checks=[]
        for seed in (701,702,703):
            for arm in ('terminal','state'):
                state=torch.load(a.restore/f'checkpoints/grpo-v1/{arm}-seed{seed}/last.pt',map_location='cpu')
                assert len(state['optimizer']['state'])==112 and state['progress']['logical_updates']==12
                checks.append(dict(arm=arm,seed=seed,optimizer_states=112,logical_updates=12))
        for seed in (101,202,303):
            for arm in ('full','no_history'):
                state=torch.load(a.restore/f'checkpoints/history-ablation-v1/{arm}-seed{seed}/best.pt',map_location='cpu')
                assert state['optimizer']['state'] and state['step']>0
        result.update(restored_grpo_optimizer_checks=checks,restored_history_optimizers=6)
    write(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
