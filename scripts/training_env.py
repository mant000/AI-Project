"""Add pinned adapter-training packages without changing the verified base environment."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;out=root/'operations/training-environment';out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    assert not (out/'completed.json').exists(),'Training additions already installed; do not reinstall'
    packages=('torch','transformers','accelerate','numpy','safetensors','huggingface-hub');before={k:importlib.metadata.version(k) for k in packages}
    (out/'before.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    wheels=root/'cache/training-wheels';wheels.mkdir(parents=True,exist_ok=True)
    commands=[[sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:','--dest',str(wheels),'peft==0.13.2','bitsandbytes==0.44.1'],
        [sys.executable,'-m','pip','install','--no-deps','--no-index','--find-links',str(wheels),'peft==0.13.2','bitsandbytes==0.44.1'],
        [sys.executable,'-m','pip','check']]
    for i,cmd in enumerate(commands):
        with (out/f'command{i}.log').open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,text=True)
        assert result.returncode==0,(cmd,result.returncode)
    after={k:importlib.metadata.version(k) for k in packages};assert before==after
    (out/'after.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    import peft,bitsandbytes
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels.glob('*.whl')}
    result=dict(at_utc=datetime.now(timezone.utc).isoformat(),base_packages_before=before,base_packages_after=after,
        base_versions_unchanged=True,additions=dict(peft=peft.__version__,bitsandbytes=bitsandbytes.__version__),
        wheel_sha256=hashes,pip_check_passed=True,full_model_or_data_downloaded=False,
        cuda_training_not_yet_validated=True,wall_seconds=time.perf_counter()-start)
    (out/'completed.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
