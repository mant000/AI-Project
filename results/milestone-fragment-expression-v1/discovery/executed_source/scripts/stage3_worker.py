"""One isolated arithmetic program evaluation; communicates only JSON stdin/stdout."""
import json
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from tsp_value.programs import compile_rule
from tsp_value.solver import create_state,run,select_edge,clone,ARRAY_TYPES,paired_action_set
from tsp_value.data import length,validate_tour


def restore_state(snapshot):
    state=create_state(snapshot['coords'],17)
    state.update(snapshot)
    for k,dtype in ARRAY_TYPES.items():
        if k in state: state[k]=np.asarray(state[k],dtype=dtype)
    return state


def main():
    request=json.load(sys.stdin); cfg=request['limits']
    resource.setrlimit(resource.RLIMIT_AS,(cfg['worker_address_space_gib']*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['worker_cpu_limit_seconds'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,)*2)
    resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
    start,cpu=time.perf_counter(),time.process_time(); results=[]
    try:
        if request['mode']=='probe':
            for row in request['states']:
                s=restore_state(row['state'])
                result=paired_action_set(s,tuple(row['reference_edge']),row['probe_edges'],
                    request['seeds'],request['horizon_by_nodes'][str(row['nodes'])],trace=True)
                results.append(dict(bank_id=row['bank_id'],instance_id=row['instance_id'],nodes=row['nodes'],**result))
        else:
            rule=compile_rule(request['expression'])
            if request['mode']=='decisions':
                for row in request['states']:
                    s=restore_state(row['state']); edge=select_edge(s,rule)
                    results.append(dict(bank_id=row['bank_id'],edge=list(edge)))
            else:
                for item in request['items']:
                    for seed in request['seeds']:
                        t,c=time.perf_counter(),time.process_time()
                        method=request.get('method','gls')
                        s=create_state(item['coords'],seed,mode='two_opt' if method=='two_opt' else 'gls')
                        budget=request['budget_by_nodes'][str(item['nodes'])]
                        if method!='nn': run(s,budget,rule=rule if method=='gls' else None)
                        validate_tour(s['best_tour'],item['nodes']); actual=length(s['best_tour'],s['distance'])
                        assert abs(actual-s['best_length'])<1e-10
                        assert method!='gls' or s['delta_evaluations']==budget
                        results.append(dict(instance_id=item['instance_id'],nodes=item['nodes'],project_use=item['project_use'],seed=seed,
                            original_length=actual,reference_length=item['reference_length'],gap_percent=100*(actual/item['reference_length']-1),
                            best_tour=s['best_tour'].tolist(),delta_evaluations=s['delta_evaluations'],penalty_events=s['penalty_events'],
                            method=method,wall_seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-c))
        result=dict(ok=True,results=results)
    except Exception:
        result=dict(ok=False,error=traceback.format_exc(),results=results)
    result.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
