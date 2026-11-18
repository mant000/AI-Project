"""Frozen exact-work solver controls, bounded timing pilot and strong external references."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from stage3_common import CODE,read,write,append,digest,stamp,worker
from core_mechanism import select_slice
from timed_engine import timed_solve
from training_study import monitor_gpu
from tsp_value.programs import canonical_program,INITIAL
from tsp_value.solver import create_state,run
from tsp_value.data import length,validate_tour,distances
from tsp_value.neural_direct import NeuralDirect
from external_reference import LEHDReference,lkh_solve


def collect_programs(controls,cfg):
    programs={};aliases=[]
    def add(label,program):
        p=canonical_program(program['source']);assert p['program_sha256']==program['program_sha256']
        programs[p['program_sha256']]=p;aliases.append(dict(label=label,program_sha256=p['program_sha256']))
    add('conventional GLS',canonical_program(INITIAL))
    for r in read(CODE/'results/main/discovery-v1/frozen_programs.json')['programs']:
        add(f"main/{r['generator']}/{r['feedback_arm']}/{r['discovery_seed']}",r['selected_program'])
    for r in read(CODE/'results/eoh-control-v1/frozen_programs.json')['programs']:add(f"eoh/{r['discovery_seed']}",r['program'])
    for prefix in ('results/stage3','results/core'):
        for file in (CODE/prefix).rglob('frozen_programs.json'):
            for i,row in enumerate(read(file)['programs']):
                p=row.get('selected_program',row.get('program'));assert p is not None
                add(f'legacy/{prefix}/{i}',p)
    for name,expr in controls['manual_programs'].items():
        implementation=cfg.get('manual_implementations',{}).get(name,expr)
        add('manual/'+name,canonical_program(implementation))
    return programs,aliases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    cfgpath=CODE/'configs/dev_benchmark_v1.json';cfg=read(cfgpath);controls=read(CODE/'configs/solver_controls_v1.json')
    assert read(CODE/'results/main/discovery-v1/completed.json')['status']=='completed'
    assert read(CODE/'results/eoh-control-v1/completed.json')['status']=='completed'
    assert read(CODE/'results/baselines/fixture-v2/verification.json')['status']=='passed'
    assert read(CODE/'results/solver-controls/fixture-v1/verification.json')['status']=='passed'
    out=CODE/'results/solver-controls/dev-v1';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (out/'completed.json').exists()
    for name in ('exact','timing','external'):(out/name).mkdir(exist_ok=True)
    programs,aliases=collect_programs(controls,cfg);data=root/'data/prepared/stage1/project_instances.jsonl'
    assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    items=select_slice(rows,cfg['pool'],cfg['offset_per_size'],cfg['instances_per_size'])
    setting=cfg['timing_subset'];timing_items=select_slice(rows,cfg['pool'],setting['offset_per_size'],setting['instances_per_size'])
    protocol=dict(protocol=cfg,controls_sha256=digest(CODE/'configs/solver_controls_v1.json'),data_sha256=digest(data),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in [Path(__file__),CODE/'scripts/timed_engine.py',CODE/'scripts/timed_worker.py',
            CODE/'scripts/external_reference.py',CODE/'src/tsp_value/neural_direct.py',CODE/'src/tsp_value/solver.py',CODE/'src/tsp_value/programs.py']},
        program_hashes=sorted(programs),aliases=aliases,all_programs_frozen_before_this_dev=True)
    if (out/'protocol_frozen.json').exists():
        old=read(out/'protocol_frozen.json')
        for k,v in protocol.items():assert old[k]==v
    else:
        write(out/'protocol_frozen.json',dict(at_utc=stamp(),**protocol))
        write(out/'selection.json',dict(exact=[r['instance_id'] for r in items],timing=[r['instance_id'] for r in timing_items]))
    start,cpu=time.perf_counter(),time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();stop,thread=monitor_gpu(out,stamp(),start)
    try:
        for key,p in programs.items():
            path=out/'exact'/f'gls-{key}.json'
            if not path.exists():
                result=worker(dict(mode='endpoint',method='gls',expression=p['expression'],items=items,seeds=cfg['solver_seeds'],budget_by_nodes=cfg['delta_budget_by_nodes']),cfg)
                assert result['ok'];write(path,dict(program_sha256=key,**result))
            print('exact gls '+key,flush=True)
        initial=canonical_program(INITIAL)
        for method in cfg['traditional_baselines']:
            path=out/'exact'/f'{method}.json'
            if not path.exists():
                result=worker(dict(mode='endpoint',method=method,expression=initial['expression'],items=items,seeds=cfg['solver_seeds'],budget_by_nodes=cfg['delta_budget_by_nodes']),cfg)
                assert result['ok'];write(path,result)
        native=read(out/'exact/native_gls.json')['results']
        compiled=read(out/'exact'/f'gls-{initial["program_sha256"]}.json')['results']
        assert all(a['instance_id']==b['instance_id'] and a['seed']==b['seed'] and a['best_tour']==b['best_tour']
            and a['delta_evaluations']==b['delta_evaluations'] for a,b in zip(native,compiled))
        for label,method,p in [(key,'gls',p) for key,p in programs.items()]+[(m,m,initial) for m in cfg['traditional_baselines']]:
            path=out/'timing'/f'{label}.json'
            if path.exists():continue
            request=dict(method=method,expression=p['expression'],items=timing_items,seeds=cfg['solver_seeds'],deadlines=cfg['deadline_seconds'],limits=cfg)
            proc=subprocess.run([sys.executable,str(CODE/'scripts/timed_worker.py')],input=json.dumps(request),capture_output=True,text=True,timeout=cfg['timing_worker_wall_timeout_seconds'])
            result=json.loads(proc.stdout);assert proc.returncode==0 and result['ok'],proc.stderr
            write(path,result);print('timing '+label,flush=True)
        teacher=NeuralDirect(root,cfg['neural_direct_device'])
        path=out/'exact/neural-direct.json'
        if not path.exists():
            result=[]
            for item in items:
                for seed in cfg['solver_seeds']:
                    teacher.reset_profile();begin,begin_cpu=time.perf_counter(),time.process_time();state=create_state(item['coords'],seed)
                    run(state,cfg['delta_budget_by_nodes'][str(item['nodes'])],rule=teacher)
                    validate_tour(state['best_tour'],item['nodes']);cost=length(state['best_tour'],state['distance']);assert abs(cost-state['best_length'])<1e-10
                    result.append(dict(instance_id=item['instance_id'],nodes=item['nodes'],seed=seed,best_tour=state['best_tour'].tolist(),
                        original_length=cost,reference_length=item['reference_length'],gap_percent=100*(cost/item['reference_length']-1),
                        delta_evaluations=state['delta_evaluations'],profile=dict(teacher.profile),
                        wall_seconds=time.perf_counter()-begin,parent_cpu_seconds=time.process_time()-begin_cpu))
            write(path,dict(ok=True,results=result,checkpoints=teacher.checkpoints))
        path=out/'timing/neural-direct.json'
        if not path.exists():
            result=[]
            for item in timing_items:
                for seed in cfg['solver_seeds']:
                    teacher.reset_profile();r=timed_solve(item['coords'],seed,cfg['deadline_seconds'],'gls',teacher)
                    for checkpoint in r['checkpoints']:checkpoint['gap_percent']=100*(checkpoint['original_length']/item['reference_length']-1)
                    r.update(instance_id=item['instance_id'],nodes=item['nodes'],profile=dict(teacher.profile));result.append(r)
            write(path,dict(ok=True,results=result))
        del teacher;torch.cuda.empty_cache()
        lehd=LEHDReference(root,cfg['external_device'])
        for label in cfg['external']:
            path=out/'external'/f'{label}.json'
            if path.exists():continue
            result=[]
            for item in items:
                for seed in cfg['solver_seeds']:
                    r=(lkh_solve(root,item['coords'],seed) if label=='lkh1000' else lehd.solve(item['coords'],seed,0 if label=='lehd_greedy' else 10))
                    r.update(instance_id=item['instance_id'],nodes=item['nodes'],gap_percent=100*(r['original_length']/item['reference_length']-1));result.append(r)
            write(path,dict(ok=True,results=result));print('external '+label,flush=True)
        write(out/'completed.json',dict(status='completed',at_utc=stamp(),unique_programs=len(programs),logical_program_aliases=len(aliases),
            exact_original_instances=len(items),timing_pilot_original_instances=len(timing_items),solver_seeds=cfg['solver_seeds'],
            wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),final_opened=False))
    except Exception:
        append(out/'failures.jsonl',dict(at_utc=stamp(),error=traceback.format_exc(),wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu));raise
    finally:
        stop.set();thread.join(timeout=5);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        append(out/'attempt_costs.jsonl',dict(at_utc=stamp(),wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
            waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)))


if __name__=='__main__':main()
