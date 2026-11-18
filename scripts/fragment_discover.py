"""A/P/F frozen-original-generator study; matched instances, budgets, seeds and real selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback
import numpy as np
from stage3_common import CODE,read,write,append,digest,stamp,worker
from fragment_core import restore
from fragment_feedback_worker import initialize,ready,measure
from tsp_value.programs import canonical_program,INITIAL


class FragmentStudy:
    def __init__(self,root,out,prepare,cfg):
        self.root,self.out,self.prepare,self.cfg=root,out,prepare,cfg
        for name in ('programs','candidates','generations','evaluations','feedback','rollouts'):(out/name).mkdir(parents=True,exist_ok=True)
        wanted=set(read(prepare/'feedback_selection.json')['bank_ids'])
        with gzip.open(prepare/'states.jsonl.gz','rt') as f:self.bank=[r for r in map(json.loads,f) if r['bank_id'] in wanted]
        assert len(self.bank)==128 and len({r['instance_id'] for r in self.bank})==128
        self.pools={p:read(prepare/f'{p}_instances.json') for p in ('discovery','program_select','program_dev_eval')}
        self.programs={};self.cache={};self.score_cache={};self.candidates=[]
        # Fork only pure NumPy workers, before importing/initializing CUDA.
        self.pool=ProcessPoolExecutor(max_workers=cfg['workers'],initializer=initialize,initargs=(self.bank,cfg))
        assert list(self.pool.map(ready,range(cfg['workers'])))==list(range(cfg['workers']))
        self.threads=ThreadPoolExecutor(max_workers=cfg['workers'])

    def save_program(self,p):
        self.programs[p['program_sha256']]=p;path=self.out/'programs'/f'{p["program_sha256"]}.py'
        if not path.exists():path.write_text(p['source'])

    def endpoint(self,p,pool,seeds,method='gls'):
        key=f'{pool}-{method}-{p["program_sha256"]}-'+','.join(map(str,seeds));path=self.out/'evaluations'/f'{key}.json'
        if key in self.cache:return self.cache[key],True
        if path.exists():result=read(path);cached=True
        else:
            begin=time.perf_counter();items=self.pools[pool];chunks=[items[i:i+4] for i in range(0,len(items),4)]
            requests=[dict(mode='endpoint',method=method,expression=p['expression'],items=rr,seeds=seeds,budget_by_nodes=self.cfg['delta_budget_by_nodes']) for rr in chunks]
            results=list(self.threads.map(lambda request:worker(request,self.cfg),requests))
            result=dict(ok=all(r['ok'] for r in results),results=[r for obj in results for r in obj.get('results',[])],
                program_sha256=p['program_sha256'],pool=pool,seeds=seeds,method=method,at_utc=stamp(),
                wall_seconds=time.perf_counter()-begin,overlapping_child_cpu_deltas_not_additive=sum(r.get('child_cpu_seconds',0) for r in results),
                cpu_scope='Concurrent subprocess child deltas overlap; authoritative total waited-child CPU is process_resources.json; worker internal cpu sum below',
                worker_internal_cpu_seconds=sum(r.get('cpu_seconds',0) for r in results),subprocess_records=results)
            if not result['ok']:result['error']='One or more isolated endpoint partitions failed'
            if result['ok']:
                result['mean_gap_percent']=float(np.mean([r['gap_percent'] for r in result['results']]))
                result['by_nodes_mean_gap_percent']={str(n):float(np.mean([r['gap_percent'] for r in result['results'] if r['nodes']==n])) for n in (50,100)}
            write(path,result);cached=False
        self.cache[key]=result;return result,cached

    def score(self,p,mode):
        key=mode+'-'+p['program_sha256'];path=self.out/'rollouts'/f'{key}.json'
        if key in self.score_cache:return self.score_cache[key]
        if path.exists():result=read(path)
        else:
            begin=time.perf_counter();rr=list(self.pool.map(measure,[(i,p['expression'],mode) for i in range(len(self.bank))],chunksize=4))
            result=dict(mode=mode,program_sha256=p['program_sha256'],results=rr,physical_branches=len(rr),
                wall_seconds=time.perf_counter()-begin,worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rr),bank_sha256=digest(self.prepare/'states.jsonl.gz'))
            write(path,result)
        self.score_cache[key]=result;return result

    def feedback(self,parent,child,arm,seed,index):
        mode='single' if arm=='P' else 'fragment';pa=self.score(parent,mode);ch=pa if parent['program_sha256']==child['program_sha256'] else self.score(child,mode)
        details=[]
        for row,p,c in zip(self.bank,pa['results'],ch['results']):
            assert p['bank_id']==c['bank_id']==row['bank_id']
            details.append(dict(bank_id=row['bank_id'],instance_id=row['instance_id'],nodes=row['nodes'],policy_index=row['policy_index'],phase_index=row['phase_index'],
                child_minus_parent_benefit_pp=100*(p['best_length']-c['best_length'])/row['state']['best_length'],
                same_first_action=p['first_edge']==c['first_edge'],same_controlled_actions=p['controlled_action_sha256']==c['controlled_action_sha256'],
                parent_controlled_decisions=p['controlled_decisions'],child_controlled_decisions=c['controlled_decisions']))
        chosen=[(seed+13*index)%len(details),(seed+13*index+len(details)//2)%len(details)];cases=[]
        for j in chosen:
            row=self.bank[j];r=details[j];s=restore(row);d=s['distance'];lengths=d[s['tour'],np.roll(s['tour'],-1)]
            def features(edge):
                a,b=edge
                return dict(distance_over_current_mean=float(d[a,b]/lengths.mean()),penalty=int(s['penalties'][a,b]))
            cases.append(dict(nodes=r['nodes'],work_fraction=row['actual_work_fraction'],progress=float(np.log1p(s['penalty_events'])),
                stagnation=float(np.log1p(s['stagnation_sweeps'])),parent_first_choice=features(pa['results'][j]['first_edge']),
                child_first_choice=features(ch['results'][j]['first_edge']),same_first_choice=r['same_first_action'],
                parent_controlled_decisions=r['parent_controlled_decisions'],child_controlled_decisions=r['child_controlled_decisions'],
                measured_child_minus_parent_benefit_pp=r['child_minus_parent_benefit_pp']))
        values=np.array([r['child_minus_parent_benefit_pp'] for r in details])
        display=dict(assessment='one rule-selected penalty then standard continuation' if mode=='single' else 'rule controls25%remaining work then standard continuation',
            compared_parent_code=parent['source'],compared_child_code=child['source'],original_bank_instances=128,
            mean_measured_child_minus_parent_benefit_pp=float(values.mean()),child_better=int((values>1e-7).sum()),
            tied=int((np.abs(values)<=1e-7).sum()),parent_better=int((values< -1e-7).sum()),
            same_first_action_fraction=float(np.mean([r['same_first_action'] for r in details])),fixed_schedule_examples=cases,
            use='A conditional measured hint, not whole-run proof. Actual complete discovery tours determine selection.')
        return display,details

    def prompt(self,ctx,arm,seed,index):
        p,ev,last=ctx['parent'],ctx['parent_eval'],ctx['last']
        endpoint=dict(parent_code=p['source'],parent_gap_percent=ev['mean_gap_percent'],gap_by_size=ev['by_nodes_mean_gap_percent'],
            original_instances=len(self.pools['discovery']),solver_seeds=self.cfg['endpoint_solver_seeds'],
            work_caps=self.cfg['delta_budget_by_nodes'],recent_attempts=[dict(source=r.get('source'),valid=r['valid'],
                gap_percent=r.get('mean_discovery_gap_percent'),error=(r.get('error') or '').splitlines()[-1:]) for r in ctx['history'][-4:]])
        user=TASK+'\nREAL ENDPOINT FEEDBACK:\n'+json.dumps(endpoint,separators=(',',':'))
        if arm!='A':
            parent=self.programs[last['parent_sha256']] if last and last['valid'] else p
            child=self.programs[last['program_sha256']] if last and last['valid'] else p
            path=self.out/'feedback'/f'{arm}-seed{seed}-candidate{index:03d}.json'
            try:display,details=self.feedback(parent,child,arm,seed,index);record=dict(status='available',display=display,states_detail=details)
            except Exception:record=dict(status='unavailable',display=dict(status='unavailable',reason='Local rollout failed; use real endpoint results'),error=traceback.format_exc())
            if path.exists():assert read(path)==record
            else:write(path,record)
            user+='\nADDITIONAL MEASURED STATE FEEDBACK:\n'+json.dumps(record['display'],separators=(',',':'))
        return [dict(role='system',content=SYSTEM),dict(role='user',content=user)]

    def generate(self,messages,seed,cid):
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=self.tokenizer(prompt,return_tensors='pt').to('cuda');tokens=int(batch['input_ids'].shape[1]);assert tokens+self.cfg['generation']['max_new_tokens']<=self.cfg['max_context_tokens']
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.synchronize();begin,cpu=time.perf_counter(),time.process_time()
        with torch.inference_mode():output=self.model.generate(**batch,**self.cfg['generation'],pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize();ids=output[0,tokens:].cpu().tolist()
        result=dict(candidate_id=cid,at_utc=stamp(),sampling_seed=seed,messages=messages,rendered_prompt=prompt,input_token_ids=batch['input_ids'][0].cpu().tolist(),
            output_token_ids=ids,response=self.tokenizer.decode(ids,skip_special_tokens=True),input_tokens=tokens,output_tokens=len(ids),
            generation_wall_seconds=time.perf_counter()-begin,parent_cpu_seconds=time.process_time()-cpu,hit_output_cap=len(ids)==self.cfg['generation']['max_new_tokens'])
        write(self.out/'generations'/f'{cid}.json',result);return result


def main():
    global torch,SYSTEM,TASK
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,default=CODE/'results/milestone-fragment-v1/discovery')
    ap.add_argument('--prepare',type=Path,default=CODE/'results/milestone-fragment-v1/prepare');a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (out/'completed.json').exists()
    cfg=read(CODE/'configs/milestone_fragment_v1.json');assert read(a.prepare.parent/'mechanism/summary.json')['signal_gate_passed']
    source=[Path(__file__),*[CODE/f'scripts/{p}.py' for p in ('fragment_core','fragment_feedback_worker','stage3_common','stage3_discover','stage3_worker','qlora_pilot')],CODE/'src/tsp_value/solver.py',CODE/'src/tsp_value/programs.py']
    provenance=dict(protocol=cfg,config_sha256=digest(CODE/'configs/milestone_fragment_v1.json'),bank_sha256=digest(a.prepare/'states.jsonl.gz'),
        source_sha256={p.relative_to(CODE).as_posix():digest(p) for p in source},parameter_training=False,final_opened=False)
    if (out/'protocol_frozen.json').exists():
        old=read(out/'protocol_frozen.json')
        for k,v in provenance.items():assert old[k]==v,k
    else:
        write(out/'protocol_frozen.json',dict(at_utc=stamp(),**provenance))
        for p in source:
            target=out/'executed_source'/p.relative_to(CODE);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    begin,cpu=time.perf_counter(),time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    study=FragmentStudy(a.root,out,a.prepare,cfg)
    import torch
    from transformers import AutoTokenizer
    from qlora_pilot import load_base
    from training_study import monitor_gpu
    from stage3_discover import SYSTEM,TASK
    torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();manifest=read(a.root/'artifacts/qwen_download.json')
    assert manifest['config']['revision']==cfg['qwen_revision']
    load=time.perf_counter();study.tokenizer=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True)
    study.model=load_base(manifest['snapshot_path'],False).eval().requires_grad_(False);study.model.config.use_cache=True
    assert not any(p.requires_grad for p in study.model.parameters());load_seconds=time.perf_counter()-load
    stop,monitor=monitor_gpu(out,stamp(),begin);initial=canonical_program(INITIAL);study.save_program(initial)
    evaluation,_=study.endpoint(initial,'discovery',cfg['endpoint_solver_seeds']);assert evaluation['ok']
    contexts={(s,arm):dict(parent=initial,parent_eval=evaluation,last=None,population=[initial],history=[]) for s in cfg['discovery_seeds'] for arm in cfg['arms']}
    try:
        for seed in cfg['discovery_seeds']:
            for index in range(cfg['candidates_per_arm_per_seed']):
                for arm in cfg['arms']:
                    assert time.perf_counter()-begin<cfg['budgets']['discovery_wall_max_seconds'],'Predeclared wall budget exceeded'
                    cid=f'{arm}-seed{seed}-candidate{index:03d}';ctx=contexts[seed,arm];parent=ctx['parent'];path=out/'candidates'/f'{cid}.json'
                    if path.exists():candidate=read(path)
                    else:
                        messages=study.prompt(ctx,arm,seed,index);rawpath=out/'generations'/f'{cid}.json'
                        raw=read(rawpath) if rawpath.exists() else study.generate(messages,seed*1000+index,cid);assert raw['messages']==messages
                        candidate=dict(candidate_id=cid,arm=arm,discovery_seed=seed,index=index,parent_sha256=parent['program_sha256'],valid=False,at_utc=stamp())
                        try:
                            p=canonical_program(raw['response']);duplicate=p['program_sha256'] in study.programs;study.save_program(p);candidate.update(**p,duplicate_program=duplicate)
                            result,reused=study.endpoint(p,'discovery',cfg['endpoint_solver_seeds']);assert result['ok'],result.get('error')
                            candidate.update(valid=True,mean_discovery_gap_percent=result['mean_gap_percent'],gap_by_size=result['by_nodes_mean_gap_percent'],evaluation_reused=reused,
                                same_endpoint_routes_as_parent=all(x['best_tour']==y['best_tour'] for x,y in zip(result['results'],ctx['parent_eval']['results'])))
                        except Exception:candidate['error']=traceback.format_exc()
                        write(path,candidate)
                    if candidate['valid']:
                        p={k:candidate[k] for k in ('expression','source','program_sha256')};study.save_program(p)
                        if p['program_sha256'] not in {q['program_sha256'] for q in ctx['population']}:ctx['population'].append(p)
                        if candidate['mean_discovery_gap_percent']<ctx['parent_eval']['mean_gap_percent']-1e-12:ctx['parent']=p;ctx['parent_eval']=study.endpoint(p,'discovery',cfg['endpoint_solver_seeds'])[0]
                    ctx['last']=candidate;ctx['history'].append(candidate);study.candidates.append(candidate)
                    write(out/'progress.json',dict(at_utc=stamp(),outputs=len(study.candidates),total=cfg['total_outputs'],candidate_id=cid,wall_seconds=time.perf_counter()-begin))
                    print(cid,candidate['valid'],candidate.get('mean_discovery_gap_percent'),flush=True)
                    if len(study.candidates)==cfg['budgets']['nested_generation_projection_check_after_outputs']:
                        seconds=time.perf_counter()-begin-load_seconds;projected=load_seconds+seconds*cfg['total_outputs']/len(study.candidates)
                        write(out/'nested_timing_pilot.json',dict(at_utc=stamp(),outputs_included=len(study.candidates),projection_seconds=projected,selection_based_only_on_runtime=True))
                        assert projected<cfg['budgets']['discovery_wall_max_seconds'],'Timing projection exceeds budget; retain incomplete record'
        frozen=[]
        for (seed,arm),ctx in contexts.items():
            elite=sorted(ctx['population'],key=lambda p:study.endpoint(p,'discovery',cfg['endpoint_solver_seeds'])[0]['mean_gap_percent'])[:cfg['elite_count']]
            scores=[study.endpoint(p,'program_select',cfg['program_select_solver_seeds'])[0]['mean_gap_percent'] for p in elite]
            frozen.append(dict(arm=arm,discovery_seed=seed,selected_program=elite[int(np.argmin(scores))],elite_hashes=[p['program_sha256'] for p in elite],program_select_gaps=scores,outputs=len(ctx['history'])))
        write(out/'frozen_programs.json',dict(at_utc=stamp(),programs=frozen,all_frozen_before_dev=True,config_sha256=provenance['config_sha256']))
        dev=[]
        for method in ('nn','two_opt','gls'):
            result,_=study.endpoint(initial,'program_dev_eval',cfg['program_dev_eval_solver_seeds'],method);assert result['ok'];dev.append(dict(label=method,evaluation=result))
        for chosen in frozen:
            result,_=study.endpoint(chosen['selected_program'],'program_dev_eval',cfg['program_dev_eval_solver_seeds']);assert result['ok']
            dev.append(dict(arm=chosen['arm'],discovery_seed=chosen['discovery_seed'],program_sha256=chosen['selected_program']['program_sha256'],evaluation=result))
        write(out/'dev_evaluation.json',dict(at_utc=stamp(),frozen_programs_sha256=digest(out/'frozen_programs.json'),runs=dev,final_opened=False))
        calls=[read(p) for p in (out/'generations').glob('*.json')]
        write(out/'completed.json',dict(status='completed',at_utc=stamp(),outputs=len(calls),valid=sum(r['valid'] for r in study.candidates),
            input_tokens=sum(r['input_tokens'] for r in calls),output_tokens=sum(r['output_tokens'] for r in calls),frozen_lineages=len(frozen),parameter_training=False,final_opened=False))
    finally:
        stop.set();monitor.join(timeout=4);study.threads.shutdown();study.pool.shutdown();after=resource.getrusage(resource.RUSAGE_CHILDREN)
        append(out/'attempt_costs.jsonl',dict(at_utc=stamp(),wall_seconds=time.perf_counter()-begin,parent_cpu_seconds=time.process_time()-cpu,
            waited_child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),model_load_seconds=load_seconds,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_seconds_scope='synchronized generation wall seconds in every raw call; gpu_samples.jsonl utilization/power'))


if __name__=='__main__':main()
