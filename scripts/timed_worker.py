"""Isolated arithmetic-program timing on shared inputs; model-free deployment only."""
import json
from pathlib import Path
import resource
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'src'))
from tsp_value.programs import compile_rule
from timed_engine import timed_solve


def main():
    request=json.load(sys.stdin);limits=request['limits']
    resource.setrlimit(resource.RLIMIT_AS,(limits['worker_address_space_gib']*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(limits['worker_cpu_limit_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32));resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,)*2)
    method=request.get('method','gls');rule=compile_rule(request['expression']) if method=='gls' else None
    start,cpu=time.perf_counter(),time.process_time();results=[]
    for item in request['items']:
        for seed in request['seeds']:
            result=timed_solve(item['coords'],seed,request['deadlines'],method,rule)
            result.update(instance_id=item['instance_id'],nodes=item['nodes'],reference_length=item['reference_length'])
            for row in result['checkpoints']:row['gap_percent']=100*(row['original_length']/item['reference_length']-1)
            results.append(result)
    print(json.dumps(dict(ok=True,results=results,wall_seconds=time.perf_counter()-start,
        cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),allow_nan=False))


if __name__=='__main__':main()
