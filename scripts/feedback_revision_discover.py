"""Frozen common-SFT A/P/R/S discovery; no parameter updates or final access."""
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
from peft import PeftModel
from stage3_common import CODE,read,write,digest,stamp
from stage3_discover import Study,SYSTEM,TASK
from qlora_pilot import load_base
from training_study import monitor_gpu
from feedback_revision_cases import case_feedback
from tsp_value.programs import canonical_program,INITIAL


class RevisionStudy(Study):
    def __init__(self,root,out,prepare):
        self.root,self.out,self.prepare=root,out,prepare
        self.cfg=read(CODE/'configs/feedback_revision_v1.json')
        complete=read(prepare/'completed.json');assert complete['signal_gate_passed']
        assert complete['protocol_sha256']==digest(CODE/'configs/feedback_revision_v1.json')
        self.bankpath=prepare/'bank_observables.jsonl';assert digest(self.bankpath)==complete['bank_sha256']
        self.bank=[json.loads(x) for x in self.bankpath.read_text().splitlines()]
        self.true={r['bank_id']:r for r in (read(p) for p in (prepare/'exact_states').glob('*.json'))}
        self.pools={p:read(prepare/(p+'_instances.json')) for p in self.cfg['pools']}
        self.programs={};self.cache={};self.decisions={};self.candidates=[];self.calls=[]
        for name in ('programs','evaluations','generations','candidates','feedback'):(out/name).mkdir(exist_ok=True)

    def prompt_for(self,ctx,arm,seed,index):
        p,ev=ctx['parent'],ctx['parent_eval'];last=ctx['last']
        recent=[dict(expression=r.get('expression'),valid=r['valid'],gap_percent=r.get('mean_discovery_gap_percent'),
            error=(r.get('error') or '').splitlines()[-1:] ) for r in ctx['history'][-4:]]
        endpoint=dict(parent_code=p['source'],parent_mean_gap_percent=ev['mean_gap_percent'],
            gap_by_size=ev['by_nodes_mean_gap_percent'],original_instances=len(self.pools['discovery']),
            solver_seeds=self.cfg['endpoint_solver_seeds'],work_caps=self.cfg['delta_budget_by_nodes'],recent_attempts=recent)
        user=TASK+'\nAvoid arithmetic changes that leave every edge ranking unchanged.\nREAL ENDPOINT FEEDBACK:\n'+json.dumps(endpoint,separators=(',',':'))
        if arm!='A':
            previous={k:last[k] for k in ('expression','source','program_sha256')} if last and last['valid'] else p
            compared_parent=self.programs[last['parent_sha256']] if arm=='P' and last and last['valid'] else p
            display,audits=case_feedback(self.bank,self.true,compared_parent,previous,arm,seed,index)
            user+='\nADDITIONAL LOCAL CASES:\n'+json.dumps(display,separators=(',',':'))
            path=self.out/'feedback'/f'{arm}-seed{seed}-candidate{index:03d}.json'
            evidence=dict(arm=arm,seed=seed,index=index,parent_sha256=p['program_sha256'],display=display,audits=audits)
            if path.exists():assert read(path)==evidence
            else:write(path,evidence)
        return [dict(role='system',content=SYSTEM),dict(role='user',content=user)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    out=CODE/'results/feedback-revision-v1/discovery';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'STOPPED_BY_USER.json').exists(), 'User deferred this study; do not resume automatically'
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'completed.json').exists()
    start,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1)
    study=RevisionStudy(a.root,out,CODE/'results/feedback-revision-v1/fresh-prepare');cfg=study.cfg
    assert not cfg['parameter_training'] and not cfg['final_evaluation']
    source=[Path(__file__),CODE/'scripts/feedback_revision_cases.py',CODE/'scripts/feedback_revision_prepare.py',
        CODE/'scripts/stage3_discover.py',CODE/'scripts/stage3_common.py',CODE/'scripts/stage3_worker.py',
        CODE/'scripts/qlora_pilot.py',CODE/'src/tsp_value/programs.py',CODE/'src/tsp_value/solver.py',CODE/'src/tsp_value/data.py']
    prov=dict(protocol_sha256=digest(CODE/'configs/feedback_revision_v1.json'),bank_sha256=digest(study.bankpath),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in source},parameters_trained=False,final_opened=False)
    if (out/'protocol_frozen.json').exists():
        old=read(out/'protocol_frozen.json')
        for k,v in prov.items():assert old[k]==v
    else:
        write(out/'protocol_frozen.json',dict(at_utc=stamp(),protocol=cfg,**prov))
        for p in source+[CODE/'configs/feedback_revision_v1.json']:
            dst=out/'executed_source'/p.relative_to(CODE);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    manifest=read(a.root/'artifacts/qwen_download.json');assert manifest['config']['revision']==cfg['qwen_revision']
    common=read(CODE/'results/training/sft-v1/checkpoint_frozen.json');adapter=a.root/'checkpoints/sft-v1/epoch2'
    assert digest(adapter/'adapter_model.safetensors')==cfg['adapter_sha256']==common['adapter_sha256']
    study.tokenizer=AutoTokenizer.from_pretrained(manifest['snapshot_path'],local_files_only=True)
    torch.cuda.reset_peak_memory_stats();load=time.perf_counter()
    study.model=PeftModel.from_pretrained(load_base(manifest['snapshot_path'],False),adapter,is_trainable=False)
    study.model.eval().requires_grad_(False);study.model.config.use_cache=True
    assert not any(p.requires_grad for p in study.model.parameters())
    load_seconds=time.perf_counter()-load;stop,thread=monitor_gpu(out,stamp(),start)
    initial=canonical_program(INITIAL);study.save_program(initial)
    ev,_=study.endpoint(initial,'discovery',cfg['endpoint_solver_seeds']);assert ev['ok']
    contexts={(seed,arm):dict(parent=initial,parent_eval=ev,last=None,population=[initial],history=[])
              for seed in cfg['discovery_seeds'] for arm in cfg['arms']}
    try:
        for seed in cfg['discovery_seeds']:
            for index in range(cfg['candidates_per_arm_per_seed']):
                for arm in cfg['arms']:
                    assert time.perf_counter()-start<cfg['budget']['discovery_wall_seconds'],'Fixed wall-time budget exceeded'
                    ctx=contexts[seed,arm];parent=ctx['parent'];cid=f'{arm}-seed{seed}-candidate{index:03d}'
                    path=out/'candidates'/f'{cid}.json'
                    if path.exists():candidate=read(path)
                    else:
                        messages=study.prompt_for(ctx,arm,seed,index);rawpath=out/'generations'/f'{cid}.json'
                        raw=read(rawpath) if rawpath.exists() else study.generate(messages,seed*1000+index,cid)
                        assert raw['messages']==messages
                        candidate=dict(candidate_id=cid,arm=arm,discovery_seed=seed,index=index,parent_sha256=parent['program_sha256'],valid=False,at_utc=stamp())
                        try:
                            p=canonical_program(raw['response']);duplicate=p['program_sha256'] in study.programs;study.save_program(p)
                            candidate.update(**p,duplicate_program=duplicate)
                            result,reused=study.endpoint(p,'discovery',cfg['endpoint_solver_seeds'])
                            if not result['ok']:raise ValueError(result.get('error','Endpoint failed'))
                            candidate.update(valid=True,mean_discovery_gap_percent=result['mean_gap_percent'],gap_by_size=result['by_nodes_mean_gap_percent'],
                                evaluation_reused=reused,same_endpoint_routes_as_parent=all(x['best_tour']==y['best_tour'] for x,y in zip(result['results'],ctx['parent_eval']['results'])))
                        except Exception:candidate['error']=traceback.format_exc()
                        write(path,candidate)
                    if candidate['valid']:
                        p={k:candidate[k] for k in ('expression','source','program_sha256')};study.save_program(p)
                        if p['program_sha256'] not in {q['program_sha256'] for q in ctx['population']}:ctx['population'].append(p)
                        if candidate['mean_discovery_gap_percent']<ctx['parent_eval']['mean_gap_percent']-1e-12:
                            ctx['parent']=p;ctx['parent_eval']=study.endpoint(p,'discovery',cfg['endpoint_solver_seeds'])[0]
                    ctx['last']=candidate;ctx['history'].append(candidate);study.candidates.append(candidate)
                    write(out/'progress.json',dict(at_utc=stamp(),outputs=len(study.candidates),total=cfg['total_outputs'],candidate_id=cid,
                        wall_seconds=time.perf_counter()-start,parameter_training_started=False,final_opened=False))
                    print(cid,candidate['valid'],candidate.get('mean_discovery_gap_percent'),flush=True)
                if seed==cfg['discovery_seeds'][0] and index==1:
                    elapsed=time.perf_counter()-start-load_seconds
                    projected=load_seconds+elapsed*cfg['total_outputs']/8
                    write(out/'nested_timing_pilot.json',dict(outputs_included=8,projection_seconds=projected,wall_seconds=elapsed,not_outcome_selected=True))
                    assert projected<cfg['budget']['discovery_wall_seconds'],'Timing projection exceeds fixed budget; stop before expansion'
        frozen=[]
        for (seed,arm),ctx in contexts.items():
            ranked=sorted(ctx['population'],key=lambda p:study.endpoint(p,'discovery',cfg['endpoint_solver_seeds'])[0]['mean_gap_percent'])[:cfg['elite_count']]
            gaps=[study.endpoint(p,'program_select',cfg['program_select_solver_seeds'])[0]['mean_gap_percent'] for p in ranked]
            frozen.append(dict(arm=arm,discovery_seed=seed,selected_program=ranked[int(np.argmin(gaps))],
                elite_hashes=[p['program_sha256'] for p in ranked],program_select_gaps=gaps,outputs=len(ctx['history'])))
        write(out/'frozen_programs.json',dict(at_utc=stamp(),programs=frozen,all_frozen_before_dev=True,protocol_sha256=prov['protocol_sha256']))
        base_ev,_=study.endpoint(initial,'program_dev_eval',cfg['program_dev_eval_solver_seeds']);assert base_ev['ok']
        dev=[dict(label='gls',evaluation=base_ev)]
        for chosen in frozen:
            result,reused=study.endpoint(chosen['selected_program'],'program_dev_eval',cfg['program_dev_eval_solver_seeds']);assert result['ok']
            dev.append(dict(label=f"{chosen['arm']}-seed{chosen['discovery_seed']}",arm=chosen['arm'],discovery_seed=chosen['discovery_seed'],
                program_sha256=chosen['selected_program']['program_sha256'],evaluation=result,reused=reused))
        write(out/'dev_evaluation.json',dict(at_utc=stamp(),runs=dev,frozen_programs_sha256=digest(out/'frozen_programs.json'),final_opened=False))
        write(out/'completed.json',dict(at_utc=stamp(),outputs=len(study.candidates),valid=sum(r['valid'] for r in study.candidates),
            frozen_lineages=len(frozen),parameters_trained=False,final_opened=False))
    finally:
        stop.set();thread.join(timeout=4)
        write(out/'process_resources.json',dict(at_utc=stamp(),wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
            child_user_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
            child_system_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,model_load_seconds=load_seconds,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            parameter_training_started=False,final_opened=False))


if __name__=='__main__':main()
