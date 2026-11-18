"""Four paired frozen-generator repetitions: endpoint / predicted / measured feedback."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import subprocess
import threading
import time
import traceback
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from stage3_common import CODE,read,write,digest,stamp
from stage3_discover import Study,SYSTEM,TASK
from core_mechanism import select_slice
from tsp_value.programs import canonical_program,INITIAL
from stage3_common import worker


class CoreStudy(Study):
    def __init__(self,root,out,prepare):
        self.root,self.out,self.prepare=root,out,prepare
        self.cfg=read(prepare/'protocol_frozen.json')['protocol'];assert self.cfg==read(CODE/'configs/core_abc_v1.json')
        self.bankpath=prepare/'bank_observables.jsonl';self.bankhash=digest(self.bankpath)
        assert self.bankhash==read(prepare/'completed.json')['bank_sha256']
        self.bank=[json.loads(x) for x in self.bankpath.read_text().splitlines()]
        self.true={read(p)['bank_id']:read(p) for p in (prepare/'exact_states').glob('*.json')}
        assert self.true.keys()=={r['bank_id'] for r in self.bank}
        rows=[json.loads(x) for x in (root/'data/prepared/stage1/project_instances.jsonl').read_text().splitlines()]
        self.pools={pool:select_slice(rows,pool,self.cfg[pool+'_offset_per_size'],self.cfg[pool+'_instances_per_size']) for pool in ('discovery','program_select','program_dev_eval')}
        self.programs={};self.cache={};self.decisions={};self.candidates=[];self.calls=[];self.contexts={}

    def decision(self,program):
        key=program['program_sha256'];path=self.out/'decisions'/f'{key}.json'
        if key not in self.decisions:
            if path.exists():result=read(path);assert result['bank_sha256']==self.bankhash
            else:
                result=worker(dict(mode='decisions',expression=program['expression'],states=self.bank),self.cfg)
                result.update(program_sha256=key,bank_sha256=self.bankhash);write(path,result)
            if not result['ok']:raise ValueError(result.get('error','Decision worker failed'))
            self.decisions[key]=result
        return self.decisions[key]['results']

    def feedback(self,parent,child,arm='B'):
        aa,bb=self.decision(parent),self.decision(child);rows=[]
        for state,pa,ch in zip(self.bank,aa,bb):
            assert state['bank_id']==pa['bank_id']==ch['bank_id']
            i,j=state['edges'].index(pa['edge']),state['edges'].index(ch['edge'])
            if arm=='C':scores=np.asarray(self.true[state['bank_id']]['true_scores_pp'])[None,:]
            else:scores=np.asarray(state['model_scores_pp'])
            delta=scores[:,j]-scores[:,i];mean,std=float(delta.mean()),float(delta.std())
            threshold=max(self.cfg['predicted_preference_threshold_pp'],std) if arm=='B' else 1e-7
            rows.append(dict(bank_id=state['bank_id'],nodes=state['nodes'],penalty_events=state['state_index'],
                parent_edge=pa['edge'],child_edge=ch['edge'],same_action=pa['edge']==ch['edge'],
                parent_value_pp=float(scores[:,i].mean()),child_value_pp=float(scores[:,j].mean()),
                child_minus_parent_pp=mean,optimization_variation_std_pp=std if arm=='B' else None,
                indicated_preference='none' if abs(mean)<=threshold else ('child' if mean>0 else 'parent')))
        warning=('Model predictions are provisional. Across-model std measures optimization variation, not calibrated uncertainty. Values assess a single action followed by conventional GLS, not the whole generated rule.' if arm=='B' else
            'Measured conditional branch outcomes at fixed H and common seed71, with conventional subsequent GLS. These are neither optimal Q values nor a stochastic expectation. Full-rule quality is still determined by real endpoints.')
        return dict(source='Transformer ensemble predictions' if arm=='B' else 'measured single-action branch rollouts',
            bank_sha256=self.bankhash,states=len(rows),same_action_fraction=float(np.mean([r['same_action'] for r in rows])),
            mean_child_minus_parent_pp=float(np.mean([r['child_minus_parent_pp'] for r in rows])),
            indicated_preference_counts=dict(Counter(r['indicated_preference'] for r in rows)),states_detail=rows,warning=warning)

    def prompt_for(self,ctx,arm):
        parent,ev,last=ctx['parent'],ctx['parent_eval'],ctx['last']
        endpoint=dict(parent_program_sha256=parent['program_sha256'],parent_code=parent['source'],
            parent_mean_discovery_gap_percent=ev['mean_gap_percent'],parent_gap_by_size=ev['by_nodes_mean_gap_percent'],
            original_instances=len(self.pools['discovery']),paired_solver_seeds=self.cfg['endpoint_solver_seeds'],
            work_budget_by_nodes=self.cfg['delta_budget_by_nodes'])
        endpoint['recent_attempts']=[dict(expression=c.get('expression'),valid=c['valid'],
            mean_gap_percent=c.get('mean_discovery_gap_percent'),same_endpoint_routes_as_parent=c.get('same_endpoint_routes_as_parent'),
            error=c.get('error','').splitlines()[-1] if c.get('error') else None) for c in ctx['history'][-8:]]
        if last:endpoint['previous_attempt']=dict(candidate_id=last['candidate_id'],previous_parent_sha256=last['parent_sha256'],
            child_sha256=last.get('program_sha256'),child_code=last.get('source'),valid=last['valid'],
            mean_gap_percent=last.get('mean_discovery_gap_percent'),gap_by_size=last.get('gap_by_size'))
        extra='\nAvoid repeating recent expressions or changes that leave every edge ranking unchanged (for example adding the same scalar to all priorities). Use endpoint outcomes to judge progress. Return exactly one function with a single return; numerical calls have bare names such as log1p, never np.log1p.\n'
        user=TASK+extra+'\nREAL ENDPOINT FEEDBACK:\n'+json.dumps(endpoint,indent=2)
        if arm in ('B','C'):
            feedback=last.get('value_feedback') if last and last['valid'] else self.feedback(parent,parent,arm)
            display={k:v for k,v in feedback.items() if k!='states_detail'}
            display['same_state_examples']=sorted(feedback['states_detail'],key=lambda r:(-abs(r['child_minus_parent_pp']),r['bank_id']))[:self.cfg['auxiliary_examples']]
            user+='\nADDITIONAL STATE VALUE FEEDBACK:\n'+json.dumps(display,indent=2)
        return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]

    def run_discovery(self):
        initial=canonical_program(INITIAL);self.save_program(initial)
        ev,_=self.endpoint(initial,'discovery',self.cfg['endpoint_solver_seeds']);assert ev['ok'];self.decision(initial)
        self.contexts={(seed,arm):dict(parent=initial,parent_eval=ev,last=None,population=[initial],history=[]) for seed in self.cfg['discovery_seeds'] for arm in self.cfg['arms']}
        for seed in self.cfg['discovery_seeds']:
            for index in range(self.cfg['candidates_per_arm_per_seed']):
                for arm in self.cfg['arms']:
                    ctx=self.contexts[seed,arm];parent=ctx['parent'];cid=f'{arm}-seed{seed}-candidate{index:03d}'
                    path=self.out/'candidates'/f'{cid}.json'
                    if path.exists():candidate=read(path)
                    else:
                        messages=self.prompt_for(ctx,arm);rawpath=self.out/'generations'/f'{cid}.json'
                        raw=read(rawpath) if rawpath.exists() else self.generate(messages,seed*1000+index,cid)
                        assert raw['messages']==messages,'Resume changed the original prompt'
                        candidate=dict(candidate_id=cid,arm=arm,discovery_seed=seed,index=index,parent_sha256=parent['program_sha256'],valid=False,at_utc=stamp())
                        try:
                            p=canonical_program(raw['response']);duplicate=p['program_sha256'] in self.programs;self.save_program(p)
                            candidate.update(**p,duplicate_program=duplicate)
                            outcome,cached=self.endpoint(p,'discovery',self.cfg['endpoint_solver_seeds']);candidate['evaluation_reused']=cached
                            if not outcome['ok']:raise ValueError(outcome.get('error','Endpoint worker failed'))
                            parent_results=ctx['parent_eval']['results']
                            same=all(a['best_tour']==b['best_tour'] for a,b in zip(outcome['results'],parent_results))
                            candidate.update(valid=True,mean_discovery_gap_percent=outcome['mean_gap_percent'],gap_by_size=outcome['by_nodes_mean_gap_percent'],
                                same_endpoint_routes_as_parent=same)
                            if arm in ('B','C'):candidate['value_feedback']=self.feedback(parent,p,arm)
                        except Exception:candidate['error']=traceback.format_exc()
                        write(path,candidate)
                    if candidate['valid']:
                        p={k:candidate[k] for k in ('expression','source','program_sha256')};self.save_program(p)
                        if p['program_sha256'] not in {r['program_sha256'] for r in ctx['population']}:ctx['population'].append(p)
                        if candidate['mean_discovery_gap_percent']<ctx['parent_eval']['mean_gap_percent']-1e-12:
                            ctx['parent']=p;ctx['parent_eval']=self.endpoint(p,'discovery',self.cfg['endpoint_solver_seeds'])[0]
                    ctx['last']=candidate;ctx['history'].append(candidate);self.candidates.append(candidate)
                    print(json.dumps(dict(candidate_id=cid,valid=candidate['valid'],gap=candidate.get('mean_discovery_gap_percent'),
                        incumbent_gap=ctx['parent_eval']['mean_gap_percent'],duplicate=candidate.get('duplicate_program'))),flush=True)
                if seed==self.cfg['discovery_seeds'][0] and index==1:
                    write(self.out/'first_closed_loop.json',dict(at_utc=stamp(),outputs=6,nested_in_total_budget=True,
                        protocol_changed=False,next_action='Continue the predeclared fixed study; no outcome-dependent expansion'))
        return self.contexts

    def select_and_dev(self,contexts):
        frozen=[]
        for (seed,arm),ctx in contexts.items():
            actual=[c for c in self.candidates if c['discovery_seed']==seed and c['arm']==arm]
            if not actual:continue
            ranked=sorted(ctx['population'],key=lambda p:self.endpoint(p,'discovery',self.cfg['endpoint_solver_seeds'])[0]['mean_gap_percent'])
            elites=ranked[:self.cfg['elite_count']];scores=[]
            for p in elites:
                result,_=self.endpoint(p,'program_select',self.cfg['program_select_solver_seeds'])
                scores.append(result['mean_gap_percent'] if result['ok'] else 1e100)
            selected=elites[int(np.argmin(scores))]
            frozen.append(dict(arm=arm,discovery_seed=seed,actual_outputs=len(actual),elites=[p['program_sha256'] for p in elites],
                select_mean_gaps_percent=scores,selected_program=selected,frozen_at_utc=stamp()))
        write(self.out/'frozen_programs.json',dict(at_utc=stamp(),programs=frozen,all_programs_frozen_before_dev_outcomes=True,
            protocol_sha256=digest(CODE/'configs/core_abc_v1.json')))
        baseline=canonical_program(INITIAL);dev=[]
        for method in ('nn','two_opt','gls'):
            result,reused=self.endpoint(baseline,'program_dev_eval',self.cfg['program_dev_eval_solver_seeds'],method=method)
            assert result['ok'];dev.append(dict(label=method,method=method,evaluation=result,reused=reused))
        for chosen in frozen:
            result,reused=self.endpoint(chosen['selected_program'],'program_dev_eval',self.cfg['program_dev_eval_solver_seeds'])
            assert result['ok'];dev.append(dict(label=f"{chosen['arm']}-seed{chosen['discovery_seed']}",arm=chosen['arm'],
                discovery_seed=chosen['discovery_seed'],program_sha256=chosen['selected_program']['program_sha256'],evaluation=result,reused=reused))
        write(self.out/'dev_evaluation.json',dict(frozen_programs_sha256=digest(self.out/'frozen_programs.json'),completed_at_utc=stamp(),runs=dev,final_opened=False))
        return frozen,dev


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prepare',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists()
    import fcntl
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('programs','candidates','evaluations','decisions','generations'):(out/name).mkdir(exist_ok=True)
    start,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1);study=CoreStudy(a.root,out,a.prepare)
    manifest=read(a.root/'artifacts/qwen_download.json');assert manifest['config']['revision']==study.cfg['qwen_revision']
    source=[Path(__file__),CODE/'scripts/core_prepare.py',CODE/'scripts/stage3_discover.py',CODE/'scripts/stage3_common.py',CODE/'scripts/stage3_worker.py',CODE/'src/tsp_value/programs.py',CODE/'src/tsp_value/solver.py',CODE/'src/tsp_value/value_model.py']
    provenance=dict(started_at_utc=stamp(),protocol_sha256=digest(CODE/'configs/core_abc_v1.json'),bank_sha256=study.bankhash,
        qwen_revision=study.cfg['qwen_revision'],precision='bfloat16',parameters_trained=False,source_sha256={str(p.relative_to(CODE)):digest(p) for p in source})
    if (out/'provenance.json').exists():
        old=read(out/'provenance.json');assert old['source_sha256']==provenance['source_sha256'];assert old['protocol_sha256']==provenance['protocol_sha256']
    else:write(out/'provenance.json',provenance)
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            try:samples.append(dict(seconds=time.perf_counter()-start,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()))
            except Exception:pass
            stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();torch.cuda.reset_peak_memory_stats();load=time.perf_counter()
    study.tokenizer=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True,trust_remote_code=False)
    study.model=AutoModelForCausalLM.from_pretrained(manifest['snapshot_path'],local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval().requires_grad_(False)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load
    try:
        contexts=study.run_discovery();frozen,dev=study.select_and_dev(contexts)
    finally:
        stop.set();thread.join(timeout=3);write(out/'gpu_samples.json',samples)
        write(out/'process_resources.json',dict(started_at_utc=provenance['started_at_utc'],finished_at_utc=stamp(),wall_seconds=time.perf_counter()-start,
            process_cpu_seconds=time.process_time()-cpu,child_user_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
            child_system_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,model_load_wall_seconds=load_seconds,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            generation_timing='synchronized wall time, not GPU-kernel-only time'))
    write(out/'completed.json',dict(at_utc=stamp(),logical_outputs=len(study.candidates),frozen_lineages=len(frozen),
        final_opened=False,parameter_training_started=False,predeclared_budget_complete=True))
    print(json.dumps(read(out/'completed.json'),indent=2),flush=True)


if __name__=='__main__':main()
