"""External solver functional checks on hand-written geometry, never final instances."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import time
import numpy as np
import torch
from stage3_common import CODE,write,read,append,digest,stamp
import traceback
from external_reference import LEHDReference,lkh_solve
from tsp_value.data import distances,length


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    out=CODE/'results/baselines/fixture-v1';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'verification.json').exists();torch.set_num_threads(1)
    xy=np.asarray([[0.,0.],[.9,.1],[.8,.8],[.3,.9],[.15,.35],[.6,.4]])
    start,cpu=time.perf_counter(),time.process_time();model=LEHDReference(a.root,'cpu')
    first=model.solve(xy,31,10);second=model.solve(xy,31,10,placeholder=np.arange(6)[::-1].copy())
    assert first['best_tour']==second['best_tour'] and first['original_length']==second['original_length']
    greedy=model.solve(xy,31,0);lkh=lkh_solve(a.root,xy,31)
    rounded=xy*100+np.asarray([1000.,2500.]);tsplib=model.solve(rounded,31,10,'tsplib_euc_2d');lkh_rounded=lkh_solve(a.root,rounded,31,'tsplib_euc_2d')
    for r,c,mode in [(first,xy,'continuous_euclidean'),(greedy,xy,'continuous_euclidean'),(lkh,xy,'continuous_euclidean'),
                     (tsplib,rounded,'tsplib_euc_2d'),(lkh_rounded,rounded,'tsplib_euc_2d')]:
        assert abs(length(r['best_tour'],distances(c,mode))-r['original_length'])<1e-10
    write(out/'results.json',dict(lehd_rrc10=first,lehd_greedy=greedy,lkh=lkh,lehd_rounded=tsplib,lkh_rounded=lkh_rounded))
    result=dict(status='passed',at_utc=stamp(),fixture='hand-written six-city geometry and translated/scaled integer coordinates, not experimental data',
        exact_author_search_methods_called=True,reference_placeholder_invariance=True,valid_routes_checked=5,
        same_model_loaded_from_author_weights=True,environment_reinstalled=False,final_opened=False,
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        source_sha256={p.name:digest(p) for p in [Path(__file__),CODE/'scripts/external_reference.py']})
    write(out/'verification.json',result);print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':
    began,cpu=time.perf_counter(),time.process_time()
    directory=CODE/'results/baselines/fixture-v1';directory.mkdir(parents=True,exist_ok=True)
    attempt=stamp();snapshot=directory/('attempt-'+attempt.replace(':','-'));snapshot.mkdir()
    for p in [Path(__file__),CODE/'scripts/external_reference.py']:(snapshot/p.name).write_bytes(p.read_bytes())
    try:main()
    except Exception:
        append(directory/'failures.jsonl',dict(at_utc=stamp(),attempt=attempt,error=traceback.format_exc(),
            wall_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu));raise
