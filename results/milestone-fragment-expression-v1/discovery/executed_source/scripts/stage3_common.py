"""Shared stage3 IO, isolation, and immutable provenance helpers."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def stamp(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p,obj): Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def append(p,obj):
    with Path(p).open('a') as f: f.write(json.dumps(obj,allow_nan=False)+'\n'); f.flush()


def choose(rows,pool,n):
    selected=[]; counts=Counter()
    for row in rows:
        if row['project_use']==pool and counts[row['nodes']]<n:
            selected.append(row); counts[row['nodes']]+=1
    assert dict(counts)=={50:n,100:n}
    return selected


def worker(request,cfg):
    begin=time.perf_counter(); before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        proc=subprocess.run([sys.executable,str(CODE/'scripts/stage3_worker.py')],
            input=json.dumps(dict(**request,limits=cfg)),capture_output=True,text=True,
            timeout=cfg['worker_wall_timeout_seconds'])
        try: result=json.loads(proc.stdout)
        except Exception: result=dict(ok=False,error='Worker returned no valid JSON',stdout=proc.stdout)
        result.update(returncode=proc.returncode,stderr=proc.stderr)
        if proc.returncode: result['ok']=False
    except subprocess.TimeoutExpired as e:
        result=dict(ok=False,error='Isolated worker wall timeout',stdout=str(e.stdout),stderr=str(e.stderr))
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(subprocess_wall_seconds=time.perf_counter()-begin,
        child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime))
    return result


def evaluate(expression,items,seeds,cfg,method='gls'):
    return worker(dict(mode='endpoint',expression=expression,items=items,seeds=seeds,
                       budget_by_nodes=cfg['delta_budget_by_nodes'],method=method),cfg)
