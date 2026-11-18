"""Fixed manual and author-solver references on the new milestone dev pool.

Quality comparisons share originals and seeds, not equal CPU/GPU budgets. Every
returned tour is validated and priced on original continuous Euclidean costs.
Run after generation to avoid GPU contention. Each completed method is retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import resource
import time
import numpy as np
from stage3_common import CODE,read,write,digest,stamp
from tsp_value.programs import canonical_program, compile_rule
from tsp_value.solver import create_state,run
from tsp_value.data import length,validate_tour


def solve(job):
    root,item,seed,label,expr,budgets=job
    start,cpu=time.perf_counter(),time.process_time()
    if label=='LKH':
        from milestone_external import lkh_solve
        result=lkh_solve(root,item['coords'],seed,max_trials=1000)
    else:
        state=create_state(item['coords'],seed);run(state,budgets[str(item['nodes'])],compile_rule(expr))
        result=dict(best_tour=state['best_tour'].tolist(),original_length=length(state['best_tour'],state['distance']),delta_evaluations=state['delta_evaluations'])
    validate_tour(result['best_tour'],item['nodes'])
    return dict(**result,instance_id=item['instance_id'],nodes=item['nodes'],solver_seed=seed,label=label,
                gap_percent=100*(result['original_length']/item['reference_length']-1),
                service_wall_seconds=time.perf_counter()-start,service_parent_cpu_seconds=time.process_time()-cpu)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=CODE/'results/milestone-fragment-expression-v1/references')
    ap.add_argument('--prepare',type=Path,default=CODE/'results/milestone-fragment-v1/prepare');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfgpath=CODE/'configs/milestone_reference_v1.json';cfg=read(cfgpath);base=read(a.prepare/'protocol_frozen.json')['protocol']
    assert not (a.output/'completed.json').exists();start=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(a.output/'protocol_frozen.json',dict(at_utc=stamp(),protocol=cfg,config_sha256=digest(cfgpath),source_sha256=digest(Path(__file__)),inputs_sha256=digest(a.prepare/'program_dev_eval_instances.json')))
    items=read(a.prepare/'program_dev_eval_instances.json');assert len(items)==64
    for label,expr in [*cfg['manual_rules'].items(),('LKH',None)]:
        path=a.output/f'{label}.json'
        if path.exists():continue
        jobs=[(a.root,item,seed,label,expr,base['delta_budget_by_nodes']) for item in items for seed in cfg['solver_seeds']]
        begin=time.perf_counter()
        with ProcessPoolExecutor(max_workers=10) as pool:rr=list(pool.map(solve,jobs,chunksize=2))
        write(path,dict(label=label,results=rr,mean_gap_percent=float(np.mean([r['gap_percent'] for r in rr])),wall_seconds=time.perf_counter()-begin,parallel_workers=10,expression=expr))
        print(label,'completed',len(rr),flush=True)
    # All CPU workers have exited before CUDA is initialized.
    import torch
    from milestone_external import LEHDReference
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();model=LEHDReference(a.root,'cuda')
    for rrc in cfg['lehd']['rrc_budgets']:
        label=f'LEHD_RRC{rrc}';path=a.output/f'{label}.json'
        if path.exists():continue
        rr=[];begin=time.perf_counter()
        for item in items:
            for seed in cfg['solver_seeds']:
                obj=model.solve(item['coords'],seed,rrc_budget=rrc)
                rr.append(dict(**obj,instance_id=item['instance_id'],nodes=item['nodes'],gap_percent=100*(obj['original_length']/item['reference_length']-1)))
        write(path,dict(label=label,results=rr,mean_gap_percent=float(np.mean([r['gap_percent'] for r in rr])),wall_seconds=time.perf_counter()-begin,serial_inference=True))
        print(label,'completed',len(rr),flush=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(a.output/'completed.json',dict(status='completed',at_utc=stamp(),wall_seconds=time.perf_counter()-start,
          waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          final_opened=False,quality_comparison_not_equal_compute=True))


if __name__=='__main__':main()
