"""Serial final evaluation, immutable chunk recovery, all logical seeds retained."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
import torch
from stage3_common import CODE,read,write,append,digest,stamp
from final_prepare import verify_freeze
from final_worker import solve,fallback
from final_state import delta_budget
from training_study import monitor_gpu
from tsp_value.programs import INITIAL,canonical_program
from tsp_value.neural_direct import NeuralDirect
from tsp_value.data import distances
from external_reference import LEHDReference,lkh_solve


def atomic(path,value):
    temp=path.with_suffix('.tmp');write(temp,value);temp.replace(path)


def signature(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def isolated(request):
    begin,cpu=time.perf_counter(),time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        proc=subprocess.run([sys.executable,str(CODE/'scripts/final_worker.py')],input=json.dumps(request),
            capture_output=True,text=True,timeout=request['config']['worker_wall_timeout_seconds'])
        result=json.loads(proc.stdout);assert proc.returncode==0 and result['ok'],proc.stderr
        result['stderr']=proc.stderr
    except Exception:
        # A crashed/timed-out solver consumed its budget. Its predeclared NN
        # fallback is flagged; successful outputs are never retried for quality.
        error=traceback.format_exc();result=dict(ok=False,error=error,results=[])
        if 'proc' in locals():result.update(stdout=proc.stdout,stderr=proc.stderr,returncode=proc.returncode)
        for item in request['items']:
            for seed in request['config']['solver_seeds']:
                r=fallback(item,seed,request['kind'],request['config'],error,time.perf_counter()-begin,None)
                r.update(instance_id=item['instance_id'],nodes=item['nodes'],group=item['group'],seed=seed,
                    method=request['method'],distance_convention=item['distance_convention'],reference_length=item['reference_length'])
                for q in r['checkpoints'] if request['kind']=='timing' else [r]:q['gap_percent']=100*(q['original_length']/item['reference_length']-1)
                result['results'].append(r)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(subprocess_wall_seconds=time.perf_counter()-begin,parent_cpu_seconds=time.process_time()-cpu,
        waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--freeze',type=Path,default=CODE/'results/final/freeze-v1/protocol_frozen.json')
    ap.add_argument('--output',type=Path,default=CODE/'results/final/evaluation-v1');a=ap.parse_args()
    freeze=read(a.freeze);verify_freeze(freeze);cfg=freeze['protocol'];out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'completed.json').exists():print('Final already complete; no evaluation rerun',flush=True);return
    data=a.root/'data/prepared/final-v1/instances.jsonl';prepared=read(data.parent/'completed.json')
    assert digest(data)==prepared['data_sha256'] and prepared['freeze_sha256']==digest(a.freeze)
    items=[json.loads(x) for x in data.read_text().splitlines()];assert len(items)==cfg['instances']
    prohibited={'main_control.py','controlled_grpo.py','common_sft.py','eoh_control.py','solver_dev_benchmark.py','history_ablation.py'}
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=proc.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        assert not (len(parts)>1 and Path(parts[1]).name in prohibited),parts[:2]
    available=sorted(os.sched_getaffinity(0));selected_cpu=available[0];os.sched_setaffinity(0,{selected_cpu})
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    programs=freeze['programs'];methods=[(k,'gls',p['expression']) for k,p in sorted(programs.items())]
    methods.extend((m,m,canonical_program(INITIAL)['expression']) for m in cfg['traditional_baselines'])
    rng=np.random.default_rng(cfg['schedule_seed']);order=rng.permutation(len(items)).tolist()
    chunks=[[items[i] for i in order[k:k+cfg['chunk_instances']]] for k in range(0,len(items),cfg['chunk_instances'])]
    schedule=dict(freeze_sha256=digest(a.freeze),data_sha256=digest(data),original_available_cpus=available,
        selected_cpu=selected_cpu,method_order=[x[0] for x in methods],chunks=[[r['instance_id'] for r in c] for c in chunks],
        config=cfg,hardware_cpu=Path('/proc/cpuinfo').read_text().split('model name',1)[1].splitlines()[0],
        software=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__),
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    if (out/'schedule_frozen.json').exists():
        previous=read(out/'schedule_frozen.json')
        for key in ('freeze_sha256','data_sha256','selected_cpu','method_order','chunks','config','software','threads'):assert previous[key]==schedule[key],key
    else:write(out/'schedule_frozen.json',dict(at_utc=stamp(),**schedule))
    for phase in ('exact','external','timing'):(out/phase).mkdir(exist_ok=True)
    attempt=str(uuid.uuid4());begin,cpu=time.perf_counter(),time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    append(out/'attempts.jsonl',dict(attempt=attempt,status='started',at_utc=stamp(),pid=os.getpid()))
    stop,thread=monitor_gpu(out,stamp(),begin);teacher=None;lehd=None;status='failed'
    def save_task(phase,label,index,request,execute):
        path=out/phase/f'{index:04d}-{label}.json';request_sha=signature(request)
        if path.exists():
            old=read(path);assert old['request_sha256']==request_sha and old['status']=='completed'
            return
        start,host=time.perf_counter(),time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN)
        append(out/'task_events.jsonl',dict(attempt=attempt,phase=phase,label=label,chunk=index,status='started',at_utc=stamp()))
        result=execute();after=resource.getrusage(resource.RUSAGE_CHILDREN)
        atomic(path,dict(status='completed',at_utc=stamp(),attempt=attempt,request_sha256=request_sha,result=result,
            command_wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-host,
            waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(child.ru_utime+child.ru_stime)))
        print(phase,index,label,'saved',flush=True)
    try:
        t,c=time.perf_counter(),time.process_time();teacher=NeuralDirect(a.root,cfg['neural_direct_device'])
        append(out/'model_load_costs.jsonl',dict(attempt=attempt,model='neural_direct',wall_seconds=time.perf_counter()-t,
            parent_cpu_seconds=time.process_time()-c,checkpoints=teacher.checkpoints,amortized_not_in_per_instance_clock=True))
        for phase in ('exact','external','timing'):
            verify_freeze(freeze)
            if (out/f'{phase}_completed.json').exists():continue
            if phase=='external':
                t,c=time.perf_counter(),time.process_time();lehd=LEHDReference(a.root,cfg['external_device'])
                append(out/'model_load_costs.jsonl',dict(attempt=attempt,model='lehd',wall_seconds=time.perf_counter()-t,
                    parent_cpu_seconds=time.process_time()-c,amortized_not_in_per_instance_clock=True))
            for index,chunk in enumerate(chunks):
                if phase!='external':
                    rotation=index%len(methods);ordered=methods[rotation:]+methods[:rotation]
                    for label,method,expression in ordered:
                        req=dict(kind=phase,method=method,expression=expression,items=chunk,config=cfg)
                        save_task(phase,label,index,req,lambda req=req:isolated(req))
                    def neural():
                        rr=[]
                        for item in chunk:
                            for seed in cfg['solver_seeds']:
                                teacher.reset_profile();r=solve(item,seed,phase,'gls',teacher,cfg)
                                r['profile']=dict(teacher.profile);rr.append(r)
                        return dict(ok=True,results=rr)
                    save_task(phase,'neural-direct',index,dict(items=chunk,kind=phase,config=cfg,checkpoints=teacher.checkpoints),neural)
                else:
                    for label in cfg['external']:
                        def external(label=label):
                            rr=[]
                            for item in chunk:
                                for seed in cfg['solver_seeds']:
                                    t,c=time.perf_counter(),time.process_time()
                                    try:
                                        if label=='lkh1000':
                                            d=distances(item['coords'],item['distance_convention']);scale=cfg['lkh_integer_scale'][item['distance_convention']]
                                            assert float(np.floor(d*scale+.5).max())*100<2**31-1,'LKH integer guard'
                                            r=lkh_solve(a.root,item['coords'],seed,item['distance_convention'])
                                        else:r=lehd.solve(item['coords'],seed,0 if label=='lehd_greedy' else 10,item['distance_convention'])
                                        r['solver_failed']=False
                                    except Exception:r=fallback(item,seed,'exact',cfg,traceback.format_exc(),time.perf_counter()-t,time.process_time()-c)
                                    r.update(instance_id=item['instance_id'],nodes=item['nodes'],group=item['group'],seed=seed,
                                        distance_convention=item['distance_convention'],reference_length=item['reference_length'])
                                    r['gap_percent']=100*(r['original_length']/item['reference_length']-1);rr.append(r)
                            return dict(ok=True,results=rr)
                        save_task(phase,label,index,dict(label=label,items=chunk,config=cfg),external)
            atomic(out/f'{phase}_completed.json',dict(status='completed',at_utc=stamp(),original_instances=len(items),solver_seeds=cfg['solver_seeds']))
            if phase=='external':del lehd;lehd=None;torch.set_default_tensor_type('torch.FloatTensor');torch.cuda.empty_cache()
        verify_freeze(freeze);status='completed'
        atomic(out/'completed.json',dict(status=status,at_utc=stamp(),freeze_sha256=digest(a.freeze),
            data_sha256=digest(data),original_instances=len(items),unique_programs=len(programs),logical_program_aliases=len(freeze['aliases']),
            phases=['exact','external','timing'],cost_records='attempts.jsonl; per-task chunk costs; model_load_costs.jsonl',
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            no_method_or_program_selection_from_final=True))
    except Exception:append(out/'failures.jsonl',dict(attempt=attempt,at_utc=stamp(),error=traceback.format_exc()));raise
    finally:
        stop.set();thread.join(timeout=5);after=resource.getrusage(resource.RUSAGE_CHILDREN)
        append(out/'attempts.jsonl',dict(attempt=attempt,status=status,at_utc=stamp(),wall_seconds=time.perf_counter()-begin,
            parent_cpu_seconds=time.process_time()-cpu,waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':main()
