"""Isolated service-time sample, fixed before results, after all discovery jobs.

Sixteen originals, seed0, three repetitions. Each solve runs serially; no
multi-second repeated searches per program. This measures the implemented fixed
work budgets, not anytime quality or hardware-independent speed rankings.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import resource
import time
import numpy as np
from stage3_common import CODE,read,write,digest,stamp
from tsp_value.programs import canonical_program,compile_rule
from tsp_value.solver import create_state,run
from tsp_value.data import length


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--study',type=Path,default=CODE/'results/milestone-fragment-expression-v1/discovery')
    ap.add_argument('--prepare',type=Path,default=CODE/'results/milestone-fragment-v1/prepare')
    ap.add_argument('--output',type=Path,default=CODE/'results/milestone-fragment-expression-v1/timing');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert not (a.output/'completed.json').exists();cfg=read(CODE/'configs/milestone_reference_v1.json');base=read(a.study/'protocol_frozen.json')['protocol']
    items=read(a.prepare/'program_dev_eval_instances.json');chosen=[r for n in (50,100) for r in [x for x in items if x['nodes']==n][:cfg['timing']['instances_per_size']]]
    programs={};aliases=defaultdict(list)
    for label,expr in cfg['manual_rules'].items():
        p=canonical_program(expr);programs[p['program_sha256']]=p;aliases[p['program_sha256']].append(label)
    for f in read(a.study/'frozen_programs.json')['programs']:
        p=f['selected_program'];programs[p['program_sha256']]=p;aliases[p['program_sha256']].append(f"{f['arm']}/{f['discovery_seed']}")
    protocol=dict(at_utc=stamp(),source_sha256=digest(Path(__file__)),reference_config_sha256=digest(CODE/'configs/milestone_reference_v1.json'),
                  items=[r['instance_id'] for r in chosen],programs=programs,aliases=aliases,repetitions=3,seed=0,
                  concurrency=1,scope='fixed-work service time, startup/compile excluded; explicit model load and warmup retained separately')
    write(a.output/'protocol_frozen.json',protocol);start=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN);rr=[]
    # Warm both sizes, without recording them as new independent observations.
    for item in [next(r for r in chosen if r['nodes']==n) for n in (50,100)]:
        state=create_state(item['coords'],0);run(state,base['delta_budget_by_nodes'][str(item['nodes'])])
    methods=[('nn',None),('two_opt',None),('native_gls',None),*[(key,compile_rule(p['expression'])) for key,p in programs.items()]]
    for label,rule in methods:
        for repeat in range(3):
            for item in chosen:
                t,c=time.perf_counter(),time.process_time();s=create_state(item['coords'],0,mode='two_opt' if label=='two_opt' else 'gls')
                if label!='nn':run(s,base['delta_budget_by_nodes'][str(item['nodes'])],rule=rule)
                wall,cpu=time.perf_counter()-t,time.process_time()-c
                rr.append(dict(label=label,repeat=repeat,instance_id=item['instance_id'],nodes=item['nodes'],seed=0,
                               wall_seconds=wall,parent_cpu_seconds=cpu,best_tour=s['best_tour'].tolist(),original_length=length(s['best_tour'],s['distance']),delta_evaluations=s['delta_evaluations']))
    write(a.output/'records.json',rr)
    summary={}
    for label,_ in methods:
        summary[label]={str(n):dict(mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in rr if r['label']==label and r['nodes']==n])),
                                    median_wall_seconds=float(np.median([r['wall_seconds'] for r in rr if r['label']==label and r['nodes']==n]))) for n in (50,100)}
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    write(a.output/'completed.json',dict(status='completed',at_utc=stamp(),summary=summary,unique_originals=16,repetitions=3,
          solves=len(rr),wall_seconds=time.perf_counter()-start,waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
          scope='CPU GLS arithmetic/native/NN/2opt; author LEHD runs were serial in references, LKH parallel service times not promoted as isolated'))
    print('Isolated timing completed',len(rr),'solves',flush=True)


if __name__=='__main__':main()
