"""Isolated, resumable-chunk final arithmetic evaluation; reference costs only score output."""
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'src'))
from final_state import create_evaluation_state,timed_evaluation,delta_budget
from tsp_value.programs import compile_rule
from tsp_value.solver import run
from tsp_value.data import distances,length,validate_tour


def fallback(item,seed,kind,cfg,error,elapsed,cpu):
    state=create_evaluation_state(item['coords'],seed,'nn',item['distance_convention'])
    base=dict(best_tour=state['best_tour'].tolist(),original_length=state['best_length'],delta_evaluations=0)
    if kind=='timing':
        result=dict(checkpoints=[dict(**base,deadline_seconds=d,available_seconds=None,available_by_deadline=False,
            deadline_observed_seconds=elapsed,atomic_overshoot_seconds=max(0.,elapsed-d)) for d in cfg['deadline_seconds']],
            error=error,actual_delta_evaluations=0,wall_seconds=elapsed,process_cpu_seconds=cpu,failed_fallback_nn=True)
    else:result=dict(**base,wall_seconds=elapsed,process_cpu_seconds=cpu)
    return dict(**result,solver_failed=True,error_detail=error)


def solve(item,seed,kind,method,rule,cfg):
    begin,cpu=time.perf_counter(),time.process_time()
    try:
        if kind=='timing':
            result=timed_evaluation(item['coords'],seed,cfg['deadline_seconds'],method,rule,item['distance_convention'])
            if result['error']:raise RuntimeError(result['error'])
        else:
            state=create_evaluation_state(item['coords'],seed,method,item['distance_convention'])
            if method!='nn':run(state,delta_budget(item['nodes'],cfg['exact_work_sweeps']),rule)
            validate_tour(state['best_tour'],item['nodes']);actual=length(state['best_tour'],state['distance'])
            assert abs(actual-state['best_length'])<1e-8
            result=dict(best_tour=state['best_tour'].tolist(),original_length=actual,delta_evaluations=state['delta_evaluations'],
                penalty_events=state['penalty_events'],wall_seconds=time.perf_counter()-begin,process_cpu_seconds=time.process_time()-cpu)
        result['solver_failed']=False
    except Exception:
        result=fallback(item,seed,kind,cfg,traceback.format_exc(),time.perf_counter()-begin,time.process_time()-cpu)
    result.update(instance_id=item['instance_id'],nodes=item['nodes'],group=item['group'],seed=seed,method=method,
        distance_convention=item['distance_convention'],reference_length=item['reference_length'])
    for q in result['checkpoints'] if kind=='timing' else [result]:
        q['gap_percent']=100*(q['original_length']/item['reference_length']-1)
    return result


def main():
    req=json.load(sys.stdin);cfg=req['config']
    resource.setrlimit(resource.RLIMIT_AS,(cfg['worker_address_space_gib']*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['worker_cpu_limit_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32));resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,)*2)
    rule=compile_rule(req['expression']) if req['method']=='gls' else None
    begin,cpu=time.perf_counter(),time.process_time()
    result=[solve(item,s,req['kind'],req['method'],rule,cfg) for item in req['items'] for s in cfg['solver_seeds']]
    print(json.dumps(dict(ok=True,results=result,wall_seconds=time.perf_counter()-begin,
        process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),allow_nan=False))


if __name__=='__main__':main()
