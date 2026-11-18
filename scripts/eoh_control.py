"""Code-only EoH-inspired population/variation control on the common exact-work engine."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from transformers import AutoTokenizer
from stage3_common import CODE,read,write,append,digest,stamp
from stage3_discover import SYSTEM,TASK
from training_study import TrainingStudy,monitor_gpu
from qlora_pilot import load_base
from tsp_value.programs import canonical_program,INITIAL

OPERATORS={
    'e1':'Explore a substantially different arithmetic mechanism from the listed parents.',
    'e2':'Combine useful structure from the listed parents into a concise new rule.',
    'm1':'Change one parent with a meaningful local modification to its ranking mechanism.',
    'm2':'Modify numerical constants of one parent while preserving its broad structure.',
    'm3':'Simplify one parent by removing redundant terms while preserving a useful ranking.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    cfgpath=CODE/'configs/solver_controls_v1.json';allcfg=read(cfgpath);cfg=allcfg['eoh_style']
    out=CODE/'results/eoh-control-v1';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (out/'completed.json').exists()
    assert a.resume or not (out/'protocol_frozen.json').exists()
    study=TrainingStudy(a.root,out,CODE/'results/main/prepare-v1');maincfg=study.cfg
    provenance=dict(protocol=allcfg,protocol_sha256=digest(cfgpath),main_protocol_sha256=digest(CODE/'configs/main_control_v1.json'),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in [Path(__file__),CODE/'scripts/training_study.py',
            CODE/'scripts/qlora_pilot.py',CODE/'scripts/stage3_discover.py',CODE/'scripts/stage3_worker.py',CODE/'src/tsp_value/solver.py',CODE/'src/tsp_value/programs.py']})
    if (out/'protocol_frozen.json').exists():
        prior=read(out/'protocol_frozen.json')
        for k,v in provenance.items():assert prior[k]==v
    else:write(out/'protocol_frozen.json',dict(at_utc=stamp(),**provenance))
    start,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
    stop,thread=monitor_gpu(out,stamp(),start);manifest=read(a.root/'artifacts/qwen_download.json')
    try:
        study.tokenizer=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True)
        study.model=load_base(manifest['snapshot_path'],False).eval().requires_grad_(False);study.model.config.use_cache=True
        initial=canonical_program(INITIAL);study.save_program(initial)
        initial_eval,_=study.endpoint(initial,'discovery',maincfg['endpoint_solver_seeds']);assert initial_eval['ok']
        populations={};all_candidates=[]
        for seed in cfg['discovery_seeds']:
            rng=np.random.default_rng(seed+7331);population=[initial];history=[]
            def gap(p):return study.endpoint(p,'discovery',maincfg['endpoint_solver_seeds'])[0]['mean_gap_percent']
            for i in range(cfg['outputs_per_seed']):
                operator=cfg['operators_cycle'][i%len(cfg['operators_cycle'])];population=sorted(population,key=gap)
                weights=1/(np.arange(len(population))+1);weights=weights/weights.sum()
                count=min(2 if operator in ('e1','e2') else 1,len(population))
                indices=rng.choice(len(population),count,replace=False,p=weights);parents=[population[j] for j in indices]
                feedback=[]
                for p in parents:
                    ev,_=study.endpoint(p,'discovery',maincfg['endpoint_solver_seeds'])
                    feedback.append(dict(program_sha256=p['program_sha256'],source=p['source'],real_mean_gap_percent=ev['mean_gap_percent'],by_nodes=ev['by_nodes_mean_gap_percent']))
                messages=[dict(role='system',content=SYSTEM),dict(role='user',content=TASK+'\nPOPULATION OPERATOR: '+operator+'\n'+OPERATORS[operator]+
                    '\nREAL PARENT ENDPOINT FEEDBACK:\n'+json.dumps(feedback,indent=2)+'\nRECENT ATTEMPTS:\n'+json.dumps(history[-8:],indent=2))]
                cid=f'eoh-seed{seed}-candidate{i:03d}';path=out/'candidates'/f'{cid}.json'
                if path.exists():candidate=read(path)
                else:
                    rawpath=out/'generations'/f'{cid}.json';raw=read(rawpath) if rawpath.exists() else study.generate(messages,seed*1000+i,cid)
                    assert raw['messages']==messages
                    candidate=dict(candidate_id=cid,at_utc=stamp(),discovery_seed=seed,index=i,operator=operator,
                        parent_sha256=[p['program_sha256'] for p in parents],valid=False)
                    try:
                        p=canonical_program(raw['response']);candidate.update(**p,duplicate_program=p['program_sha256'] in study.programs);study.save_program(p)
                        ev,reused=study.endpoint(p,'discovery',maincfg['endpoint_solver_seeds'])
                        if not ev['ok']:raise ValueError(ev.get('error','Endpoint failure'))
                        candidate.update(valid=True,mean_discovery_gap_percent=ev['mean_gap_percent'],evaluation_reused=reused)
                    except Exception:candidate['error']=traceback.format_exc()
                    write(path,candidate)
                if candidate['valid']:
                    p={k:candidate[k] for k in ('expression','source','program_sha256')};study.save_program(p)
                    if p['program_sha256'] not in {x['program_sha256'] for x in population}:population.append(p)
                    population=sorted(population,key=gap)[:cfg['population']]
                history.append(dict(expression=candidate.get('expression'),valid=candidate['valid'],real_mean_gap_percent=candidate.get('mean_discovery_gap_percent')))
                all_candidates.append(candidate);write(out/'progress.json',dict(at_utc=stamp(),candidate_id=cid,outputs=len(all_candidates)))
                print(json.dumps(dict(candidate_id=cid,valid=candidate['valid'],incumbent_gap=gap(population[0]))),flush=True)
            populations[seed]=population
        frozen=[]
        for seed,population in populations.items():
            results=[study.endpoint(p,'program_select',maincfg['program_select_solver_seeds'])[0] for p in population]
            gaps=[r['mean_gap_percent'] if r['ok'] else 1e100 for r in results];chosen=population[int(np.argmin(gaps))]
            frozen.append(dict(discovery_seed=seed,program=chosen,elites=[p['program_sha256'] for p in population],selection_gaps=gaps))
        if not (out/'frozen_programs.json').exists():write(out/'frozen_programs.json',dict(at_utc=stamp(),programs=frozen,all_frozen_before_own_dev=True,final_opened=False))
        dev=[]
        for row in frozen:
            result,_=study.endpoint(row['program'],'program_dev_eval',maincfg['program_dev_eval_solver_seeds']);assert result['ok']
            dev.append(dict(discovery_seed=row['discovery_seed'],program_sha256=row['program']['program_sha256'],evaluation=result))
        write(out/'dev_evaluation.json',dict(at_utc=stamp(),frozen_programs_sha256=digest(out/'frozen_programs.json'),runs=dev,final_opened=False))
        calls=[read(p) for p in (out/'generations').glob('*.json')];evaluations=[read(p) for p in (out/'evaluations').glob('*.json')]
        write(out/'completed.json',dict(status='completed',at_utc=stamp(),outputs=len(calls),valid=sum(r['valid'] for r in all_candidates),
            input_tokens=sum(r['input_tokens'] for r in calls),output_tokens=sum(r['output_tokens'] for r in calls),
            wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
            endpoint_routes=sum(len(r.get('results',[])) for r in evaluations),worker_cpu_seconds=sum(r['cpu_seconds'] for r in evaluations),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,final_opened=False))
    finally:stop.set();thread.join(timeout=5)


if __name__=='__main__':main()
