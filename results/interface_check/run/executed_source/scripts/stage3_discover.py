"""Matched-budget frozen local 7B discovery, select, then independent dev evaluation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import threading
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from stage3_common import CODE,read,write,digest,stamp,choose,worker,evaluate,append
from tsp_value.programs import canonical_program,INITIAL,ARGUMENTS

SYSTEM='You design a pure numeric edge-priority rule for guided local search for Euclidean TSP. Obey the supplied interface. Return only Python code. No imports, comments, prose, assignments or extra functions.'
TASK='''Improve the given parent using real endpoint results on fixed discovery instances.
The solver penalizes one maximum-priority edge of the CURRENT tour then performs budget-exact best-improvement 2-opt. Quality always uses original Euclidean route length. Lower gap-to-reference is better; reference routes are not certified optima. Only the priority rule may change. Search uses a fixed work cap; do not infer improvement from a predicted value alone.
Return exactly:
def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return ONE_ARITHMETIC_EXPRESSION
d is the array of current edge lengths, p their nonnegative penalty counts. mean_d/max_d are current-tour distance summaries, lam is the frozen GLS penalty weight, progress=log1p(number of penalty events), stagnation=log1p(stagnation sweeps), best_ratio=best/current route length. Arithmetic + - * / ** and bare sqrt, log1p, exp, minimum, maximum, abs, clip are allowed. No np prefix or attributes, indexing, conditionals, loops or assignments. Max 96 AST nodes, finite constants magnitude <=100, literal powers magnitude <=4. Use numerically safe expressions. Scalar constants broadcast. Standard GLS is d/(1+p). Prefer a concise meaningful change to the parent. Every output, including invalid or duplicate code, consumes the budget.
'''


class Study:
    def __init__(self,root,out,prepare):
        self.root,self.out,self.prepare=root,out,prepare
        self.cfg=read(prepare/'protocol_frozen.json')['protocol']
        assert self.cfg==read(CODE/'configs/stage3_protocol.json')
        assert digest(prepare/'bank.jsonl')==read(prepare/'bank_frozen.json')['sha256']
        self.bank=[json.loads(x) for x in (prepare/'bank.jsonl').read_text().splitlines()]
        rows=[json.loads(x) for x in (root/'data/prepared/stage1/project_instances.jsonl').read_text().splitlines()]
        self.pools={pool:choose(rows,pool,self.cfg[key]) for pool,key in [('discovery','discovery_instances_per_size'),
            ('program_select','program_select_instances_per_size'),('program_dev_eval','program_dev_eval_instances_per_size')]}
        self.programs={}; self.cache={}; self.decisions={}; self.candidates=[]; self.calls=[]
        self.resources=[]

    def endpoint(self,program,pool,seeds,method='gls'):
        key=pool+'-'+method+'-'+program['program_sha256']+'-'+'-'.join(map(str,seeds))
        path=self.out/'evaluations'/f'{key}.json'
        if key in self.cache: return self.cache[key],True
        if path.exists(): result=read(path); cached=True
        else:
            result=evaluate(program['expression'],self.pools[pool],seeds,self.cfg,method=method)
            result.update(program_sha256=program['program_sha256'],pool=pool,seeds=seeds,method=method,at_utc=stamp())
            write(path,result); cached=False
        if result['ok']:
            result['mean_gap_percent']=float(np.mean([r['gap_percent'] for r in result['results']]))
            result['by_nodes_mean_gap_percent']={str(n):float(np.mean([r['gap_percent'] for r in result['results'] if r['nodes']==n])) for n in (50,100)}
        self.cache[key]=result
        return result,cached

    def decision(self,program):
        key=program['program_sha256']; path=self.out/'decisions'/f'{key}.json'
        if key not in self.decisions:
            if path.exists(): result=read(path)
            else:
                result=worker(dict(mode='decisions',expression=program['expression'],states=self.bank),self.cfg)
                result.update(program_sha256=key,bank_sha256=digest(self.prepare/'bank.jsonl'))
                write(path,result)
            if not result['ok']: raise ValueError(result.get('error','Decision worker failed'))
            self.decisions[key]=result
        return self.decisions[key]['results']

    def feedback(self,parent,child):
        a,b=self.decision(parent),self.decision(child); rows=[]
        for state,pa,ch in zip(self.bank,a,b):
            assert pa['bank_id']==ch['bank_id']==state['bank_id']
            i,j=state['edges'].index(pa['edge']),state['edges'].index(ch['edge'])
            scores=np.asarray(state['model_scores_pp']); delta=scores[:,j]-scores[:,i]
            mean,std=float(delta.mean()),float(delta.std())
            s=state['state']; d=np.linalg.norm(np.asarray(s['coords'])[ch['edge'][0]]-np.asarray(s['coords'])[ch['edge'][1]])
            rows.append(dict(bank_id=state['bank_id'],nodes=state['nodes'],penalty_events=state['state_index'],
                parent_edge=pa['edge'],child_edge=ch['edge'],same_action=pa['edge']==ch['edge'],
                parent_predicted_pp=float(scores[:,i].mean()),child_predicted_pp=float(scores[:,j].mean()),
                predicted_child_minus_parent_pp=mean,across_model_delta_std_pp=std,
                qualified_preference='none' if abs(mean)<=max(.001,std) else ('child_predicted' if mean>0 else 'parent_predicted'),
                child_distance=float(d),child_penalty=s['penalties'][ch['edge'][0]][ch['edge'][1]]))
        return dict(bank_sha256=digest(self.prepare/'bank.jsonl'),states=len(rows),same_action_fraction=float(np.mean([r['same_action'] for r in rows])),
            mean_predicted_child_minus_parent_pp=float(np.mean([r['predicted_child_minus_parent_pp'] for r in rows])),
            qualified_preference_counts=dict(Counter(r['qualified_preference'] for r in rows)),states_detail=rows,
            warning='Provisional predictions, not measured action outcomes or whole-solver gains. Three predeclared optimization seeds averaged; no accuracy calibration from value_val or heldout program pools.')

    def save_program(self,program):
        key=program['program_sha256']; self.programs[key]=program
        p=self.out/'programs'/f'{key}.py'
        if not p.exists(): p.write_text(program['source'])

    def prompt(self,parent,parent_eval,last,arm):
        endpoint=dict(parent_program_sha256=parent['program_sha256'],parent_code=parent['source'],
            parent_mean_discovery_gap_percent=parent_eval['mean_gap_percent'],parent_gap_by_size=parent_eval['by_nodes_mean_gap_percent'],
            instances=16,paired_solver_seeds=self.cfg['endpoint_solver_seeds'],work_budget_by_nodes=self.cfg['delta_budget_by_nodes'])
        if last:
            endpoint['previous_attempt']=dict(candidate_id=last['candidate_id'],program_sha256=last.get('program_sha256'),
                previous_parent_sha256=last['parent_sha256'],code=last.get('source'),valid=last['valid'],error=last.get('error'),
                mean_discovery_gap_percent=last.get('mean_discovery_gap_percent'),gap_by_size=last.get('gap_by_size'))
        user=TASK+'\nREAL ENDPOINT FEEDBACK:\n'+json.dumps(endpoint,indent=2)
        if arm=='B':
            feedback=last.get('value_feedback') if last and last['valid'] else self.feedback(parent,parent)
            display={k:v for k,v in feedback.items() if k!='states_detail'}
            display['same_state_examples']=sorted(feedback['states_detail'],key=lambda r:(-abs(r['predicted_child_minus_parent_pp']),r['bank_id']))[:2]
            user+='\nADDITIONAL STATE VALUE FEEDBACK:\n'+json.dumps(display,indent=2)
        return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]

    def generate(self,messages,seed,candidate_id):
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=self.tokenizer(prompt,return_tensors='pt').to('cuda')
        tokens=int(batch['input_ids'].shape[1]); assert tokens+self.cfg['generation']['max_new_tokens']<=self.cfg['max_context_tokens']
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(); t,c=time.perf_counter(),time.process_time()
        with torch.inference_mode(): output=self.model.generate(**batch,**self.cfg['generation'],pad_token_id=self.tokenizer.eos_token_id)
        torch.cuda.synchronize(); duration=time.perf_counter()-t
        ids=output[0,tokens:].cpu().tolist(); response=self.tokenizer.decode(ids,skip_special_tokens=True)
        record=dict(candidate_id=candidate_id,at_utc=stamp(),sampling_seed=seed,messages=messages,rendered_prompt=prompt,
            input_token_ids=batch['input_ids'][0].cpu().tolist(),output_token_ids=ids,response=response,
            input_tokens=tokens,output_tokens=len(ids),generation_wall_seconds=duration,process_cpu_seconds=time.process_time()-c,
            hit_output_cap=len(ids)==self.cfg['generation']['max_new_tokens'],
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        # Persist raw response BEFORE parsing or evaluating. A crash never grants a free reroll.
        write(self.out/'generations'/f'{candidate_id}.json',record); self.calls.append(record)
        return record

    def run_discovery(self):
        initial=canonical_program(INITIAL); self.save_program(initial)
        evaluation,_=self.endpoint(initial,'discovery',self.cfg['endpoint_solver_seeds']); assert evaluation['ok']
        contexts={(seed,arm):dict(parent=initial,parent_eval=evaluation,last=None,population=[initial])
            for seed in self.cfg['discovery_seeds'] for arm in ('A','B')}
        self.decision(initial)
        loop_start=time.perf_counter(); done=0; gate=None
        for seed in self.cfg['discovery_seeds']:
            for index in range(self.cfg['candidates_per_arm_per_seed']):
                for arm in ('A','B'):
                    ctx=contexts[seed,arm]; parent=ctx['parent']; candidate_id=f'{arm}-seed{seed}-candidate{index:03d}'
                    resultpath=self.out/'candidates'/f'{candidate_id}.json'
                    if resultpath.exists(): candidate=read(resultpath)
                    else:
                        messages=self.prompt(parent,ctx['parent_eval'],ctx['last'],arm)
                        rawpath=self.out/'generations'/f'{candidate_id}.json'
                        raw=read(rawpath) if rawpath.exists() else self.generate(messages,seed*1000+index,candidate_id)
                        assert raw['messages']==messages,'Resume must reproduce exact prompt'
                        candidate=dict(candidate_id=candidate_id,arm=arm,discovery_seed=seed,index=index,
                            parent_sha256=parent['program_sha256'],at_utc=stamp(),valid=False)
                        try:
                            program=canonical_program(raw['response']); duplicate=program['program_sha256'] in self.programs
                            self.save_program(program); candidate.update(**program,duplicate_program=duplicate)
                            ev,cached=self.endpoint(program,'discovery',self.cfg['endpoint_solver_seeds'])
                            candidate['evaluation_reused']=cached
                            if not ev['ok']: raise ValueError(ev.get('error','Endpoint worker failed'))
                            # Same-state values are measured from precomputed live, label-free predictions.
                            feedback=self.feedback(parent,program)
                            candidate.update(valid=True,mean_discovery_gap_percent=ev['mean_gap_percent'],
                                gap_by_size=ev['by_nodes_mean_gap_percent'],value_feedback=feedback)
                        except Exception: candidate['error']=traceback.format_exc()
                        write(resultpath,candidate)
                    if candidate['valid']:
                        program={k:candidate[k] for k in ('expression','source','program_sha256')}; self.save_program(program)
                        if program['program_sha256'] not in {p['program_sha256'] for p in ctx['population']}: ctx['population'].append(program)
                        if candidate['mean_discovery_gap_percent']<ctx['parent_eval']['mean_gap_percent']-1e-12:
                            ctx['parent']=program; ctx['parent_eval']=self.endpoint(program,'discovery',self.cfg['endpoint_solver_seeds'])[0]
                    ctx['last']=candidate; self.candidates.append(candidate); done+=1
                    print(json.dumps(dict(candidate_id=candidate_id,valid=candidate['valid'],
                        gap=candidate.get('mean_discovery_gap_percent'),incumbent_gap=ctx['parent_eval']['mean_gap_percent'],
                        duplicate=candidate.get('duplicate_program'))),flush=True)
                if seed==self.cfg['discovery_seeds'][0] and index==1:
                    elapsed=time.perf_counter()-loop_start; projection=elapsed/4*64
                    gate=dict(at_utc=stamp(),nested_first_outputs=4,elapsed_seconds=elapsed,projected_64_output_seconds=projection,
                        severe_probe_mismatch=read(self.prepare/'probe_summary.json')['severe_mismatch_gate'],
                        continue_predeclared_small_study=projection<7200 and not read(self.prepare/'probe_summary.json')['severe_mismatch_gate'],
                        no_240_output_expansion=True)
                    write(self.out/'first_closed_loop.json',gate)
                    if not gate['continue_predeclared_small_study']: return contexts,gate
        return contexts,gate

    def select_and_dev(self,contexts):
        frozen=[]
        for (seed,arm),ctx in contexts.items():
            # If gate stopped early, untouched seeds are reported as unrun, never extra repetitions.
            actual=[c for c in self.candidates if c['discovery_seed']==seed and c['arm']==arm]
            if not actual: continue
            ranked=sorted(ctx['population'],key=lambda p:self.endpoint(p,'discovery',self.cfg['endpoint_solver_seeds'])[0]['mean_gap_percent'])
            elites=ranked[:self.cfg['elite_count']]; scores=[]
            for p in elites:
                result,_=self.endpoint(p,'program_select',self.cfg['program_select_solver_seeds'])
                scores.append(result['mean_gap_percent'] if result['ok'] else 1e100)
            choice=int(np.argmin(scores)); selected=elites[choice]
            frozen.append(dict(arm=arm,discovery_seed=seed,actual_outputs=len(actual),elites=[p['program_sha256'] for p in elites],
                select_mean_gaps_percent=scores,selected_program=selected,frozen_at_utc=stamp()))
        write(self.out/'frozen_programs.json',dict(at_utc=stamp(),programs=frozen,
            all_programs_frozen_before_dev_outcomes=True,protocol_sha256=digest(CODE/'configs/stage3_protocol.json')))
        baseline=canonical_program(INITIAL)
        dev=[]
        for method in ('nn','two_opt','gls'):
            result,reused=self.endpoint(baseline,'program_dev_eval',self.cfg['program_dev_eval_solver_seeds'],method=method)
            assert result['ok']; dev.append(dict(label=method,method=method,evaluation=result,reused=reused))
        for f in frozen:
            result,reused=self.endpoint(f['selected_program'],'program_dev_eval',self.cfg['program_dev_eval_solver_seeds'])
            assert result['ok']; dev.append(dict(label=f"{f['arm']}-seed{f['discovery_seed']}",arm=f['arm'],
                discovery_seed=f['discovery_seed'],program_sha256=f['selected_program']['program_sha256'],evaluation=result,reused=reused))
        write(self.out/'dev_evaluation.json',dict(frozen_programs_sha256=digest(self.out/'frozen_programs.json'),
            completed_at_utc=stamp(),runs=dev,final_opened=False))
        return frozen,dev


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prepare',type=Path,required=True); a=ap.parse_args()
    out=a.output; out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists(): raise FileExistsError('Already completed; do not rerun')
    for name in ('programs','candidates','evaluations','decisions','generations'): (out/name).mkdir(exist_ok=True)
    begin,cpu=time.perf_counter(),time.process_time(); started=stamp(); torch.set_num_threads(1)
    study=Study(a.root,out,a.prepare)
    assert read(a.prepare/'feature_equivalence.json')['prediction_max_abs_error_pp']==0
    assert not read(a.prepare/'probe_summary.json')['severe_mismatch_gate'],'Severe discovery probe mismatch; stop before generation'
    manifest=read(a.root/'artifacts/qwen_download.json'); assert manifest['config']['revision']==study.cfg['qwen_revision']
    provenance=dict(started_at_utc=started,protocol_sha256=digest(CODE/'configs/stage3_protocol.json'),
        bank_sha256=digest(a.prepare/'bank.jsonl'),qwen_snapshot=manifest['snapshot_path'],qwen_revision=study.cfg['qwen_revision'],
        precision='bfloat16',torch=torch.__version__,transformers=__import__('transformers').__version__,
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in sorted(CODE.glob('scripts/stage3_*.py'))},
        gpu_name=torch.cuda.get_device_name(),training=False)
    if not (out/'provenance.json').exists(): write(out/'provenance.json',provenance)
    else: append(out/'resume_events.jsonl',provenance)
    samples=[]; stop=threading.Event()
    def sample():
        while not stop.is_set():
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
                samples.append(dict(seconds=time.perf_counter()-begin,raw_gpu_utilization_memory_power=s))
            except Exception: pass
            stop.wait(1)
    monitor=threading.Thread(target=sample,daemon=True); monitor.start()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); load=time.perf_counter()
    study.tokenizer=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True,trust_remote_code=False)
    study.model=AutoModelForCausalLM.from_pretrained(manifest['snapshot_path'],local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True).eval().requires_grad_(False)
    torch.cuda.synchronize(); load_seconds=time.perf_counter()-load
    write(out/'model_loaded.json',dict(at_utc=stamp(),load_seconds=load_seconds,
        parameter_count=sum(p.numel() for p in study.model.parameters()),all_parameters_frozen=all(not p.requires_grad for p in study.model.parameters()),
        dtype=str(next(study.model.parameters()).dtype)))
    try:
        contexts,gate=study.run_discovery()
        frozen,dev=study.select_and_dev(contexts)
    finally:
        stop.set(); monitor.join(timeout=3); write(out/'gpu_samples.json',samples)
        write(out/'process_resources.json',dict(started_at_utc=started,finished_at_utc=stamp(),wall_seconds=time.perf_counter()-begin,
            process_cpu_seconds=time.process_time()-cpu,child_user_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
            child_system_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_load_wall_seconds=load_seconds,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_generation_timing_definition='Synchronized wall time around generate; not GPU-kernel-only seconds. Raw nvidia-smi utilization, memory, power samples retained.'))
    write(out/'completed.json',dict(completed_at_utc=stamp(),logical_outputs=len(study.candidates),frozen_programs=len(frozen),
        first_closed_loop=gate,final_opened=False,sft_grpo_started=False,stop_reason='Predeclared small study complete; no expansion to 240 outputs'))
    print(json.dumps(read(out/'completed.json'),indent=2),flush=True)


if __name__=='__main__': main()
