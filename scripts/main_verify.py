"""Audit every main-study route, prompt, parent choice and independent frozen-program selection."""
import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import re
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from stage3_common import CODE,read,write,digest,stamp,worker
from training_study import TrainingStudy
from stage3_worker import restore_state
from tsp_value.programs import canonical_program,compile_rule,INITIAL
from tsp_value.solver import clone,select_edge
from tsp_value.data import distances,length,validate_tour
from tsp_value.value_model import EdgeValue,state_features,collate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    out=CODE/'results/main/discovery-v1';assert read(out/'completed.json')['status']=='completed'
    audit=CODE/'results/main/verification-v1';audit.mkdir(exist_ok=True);assert not (audit/'summary.json').exists()
    start,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1)
    cfg=read(CODE/'configs/main_control_v1.json');provenance=read(out/'protocol_frozen.json')
    assert provenance['protocol']==cfg
    for name,expected in provenance['source_sha256'].items():assert digest(CODE/name)==expected,name
    study=TrainingStudy(a.root,out,CODE/'results/main/prepare-v1')
    def endpoint(program,pool,seeds,method='gls'):
        key=pool+'-'+method+'-'+program['program_sha256']+'-'+'-'.join(map(str,seeds))
        assert (out/'evaluations'/f'{key}.json').exists(),key
        return study.endpoint(program,pool,seeds,method)[0]
    frozen=read(out/'frozen_programs.json');coords={r['instance_id']:r for rows in study.pools.values() for r in rows}
    matrices={key:distances(row['coords']) for key,row in coords.items()};routes=0;maximum=0.;failed_evaluations=0
    for path in sorted((out/'evaluations').glob('*.json')):
        ev=read(path);allowed={r['instance_id'] for r in study.pools[ev['pool']]}
        if ev['pool']=='program_dev_eval':assert ev['at_utc']>=frozen['at_utc']
        failed_evaluations+=not ev['ok']
        for r in ev.get('results',[]):
            assert r['instance_id'] in allowed and r['project_use']==ev['pool'] and r['seed'] in ev['seeds']
            item=coords[r['instance_id']];validate_tour(r['best_tour'],r['nodes']);cost=length(r['best_tour'],matrices[r['instance_id']])
            maximum=max(maximum,abs(cost-r['original_length']));assert abs(cost-r['original_length'])<1e-10
            assert abs(100*(cost/item['reference_length']-1)-r['gap_percent'])<1e-10
            if ev['method']=='gls':assert r['delta_evaluations']==cfg['delta_budget_by_nodes'][str(r['nodes'])]
            routes+=1
    bankstates={r['bank_id']:restore_state(r['state']) for r in study.bank};decisions=0
    for path in (out/'decisions').glob('*.json'):
        value=read(path);assert value['bank_sha256']==study.bankhash and value['ok']
        p=canonical_program((out/'programs'/f'{value["program_sha256"]}.py').read_text());rule=compile_rule(p['expression'])
        for row in value['results']:
            actual=select_edge(clone(bankstates[row['bank_id']]),rule);assert list(actual)==row['edge'];decisions+=1
    models=[]
    for seed in cfg['transformer_seeds']:
        path=a.root/f'checkpoints/stage2/training-v1/transformer-seed{seed}/best.pt';expected=read(CODE/f'results/stage2/training-v1/transformer-seed{seed}_summary.json')['checkpoint_sha256'];assert digest(path)==expected
        ck=torch.load(path,map_location='cpu');model=EdgeValue('transformer',**ck['config']['architecture']).eval();model.load_state_dict(ck['model']);models.append(model)
    prediction_error=0.
    for row in study.bank:
        batch=collate([state_features(row['state'],row['edges'])],'cpu')
        with torch.inference_mode():actual=np.stack([m(batch).numpy()[0] for m in models])
        prediction_error=max(prediction_error,float(np.max(np.abs(actual-np.asarray(row['model_scores_pp'])))))
    assert prediction_error<=1e-6
    tokenizer=AutoTokenizer.from_pretrained(read(a.root/'artifacts/qwen_download.json')['snapshot_path'],local_files_only=True)
    allrows=[json.loads(x) for x in (a.root/'data/prepared/stage1/project_instances.jsonl').read_text().splitlines()]
    protected={r['instance_id'] for r in allrows if r['project_use']!='discovery'}
    initial=canonical_program(INITIAL);study.programs[initial['program_sha256']]=initial
    baseline=endpoint(initial,'discovery',cfg['endpoint_solver_seeds']);contexts={};outputs=feedback_rows=valid=0
    for generator in cfg['generators']:
        for seed in cfg['discovery_seeds']:
            for arm in cfg['feedback_arms'][generator]:contexts[generator,seed,arm]=dict(parent=initial,parent_eval=baseline,last=None,population=[initial],history=[])
            for index in range(cfg['candidates_per_arm_per_seed']):
                for arm in cfg['feedback_arms'][generator]:
                    ctx=contexts[generator,seed,arm];cid=f'{generator}-{arm}-seed{seed}-candidate{index:03d}'
                    c=read(out/'candidates'/f'{cid}.json');raw=read(out/'generations'/f'{cid}.json');messages=study.prompt_for(ctx,arm)
                    assert raw['messages']==messages and raw['sampling_seed']==seed*1000+index
                    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                    assert raw['rendered_prompt']==prompt and tokenizer(prompt)['input_ids']==raw['input_token_ids']
                    assert raw['input_tokens']==len(raw['input_token_ids']) and raw['output_tokens']==len(raw['output_token_ids'])
                    assert not (protected & set(re.findall(r'joshi:[A-Za-z0-9_.-]+:\d+',prompt)))
                    assert c['parent_sha256']==ctx['parent']['program_sha256']
                    if 'program_sha256' in c:
                        p=canonical_program(raw['response']);assert p['program_sha256']==c['program_sha256']
                        assert c['duplicate_program']==(p['program_sha256'] in study.programs)
                        assert (out/'programs'/f'{p["program_sha256"]}.py').read_text()==p['source'];study.programs[p['program_sha256']]=p
                    if c['valid']:
                        valid+=1;ev=endpoint(p,'discovery',cfg['endpoint_solver_seeds']);assert ev['ok']
                        assert abs(ev['mean_gap_percent']-c['mean_discovery_gap_percent'])<1e-12
                        if arm=='B':
                            fb=study.feedback(ctx['parent'],p,'B');assert fb==c['value_feedback'];feedback_rows+=len(fb['states_detail'])
                        else:assert 'value_feedback' not in c
                        if p['program_sha256'] not in {x['program_sha256'] for x in ctx['population']}:ctx['population'].append(p)
                        if ev['mean_gap_percent']<ctx['parent_eval']['mean_gap_percent']-1e-12:ctx['parent'],ctx['parent_eval']=p,ev
                    else:assert c.get('error')
                    ctx['last']=c;ctx['history'].append(c);outputs+=1
    assert outputs==cfg['total_outputs']==1080 and len(frozen['programs'])==len(contexts)==45
    selected={(r['generator'],r['discovery_seed'],r['feedback_arm']):r for r in frozen['programs']}
    for key,ctx in contexts.items():
        ranked=sorted(ctx['population'],key=lambda p:endpoint(p,'discovery',cfg['endpoint_solver_seeds'])['mean_gap_percent'])[:cfg['elite_count']]
        gaps=[endpoint(p,'program_select',cfg['program_select_solver_seeds'])['mean_gap_percent'] for p in ranked]
        expected=ranked[int(np.argmin(gaps))];assert expected==selected[key]['selected_program']
        assert [p['program_sha256'] for p in ranked]==selected[key]['elites']
    baseline_methods=[('gls',initial),('two_opt',initial)];replays=[]
    for method,p in baseline_methods:
        stored=endpoint(p,'program_dev_eval',cfg['program_dev_eval_solver_seeds'],method)
        for n in (50,100):
            item=next(r for r in study.pools['program_dev_eval'] if r['nodes']==n)
            request=dict(mode='endpoint',method=method,expression=p['expression'],items=[item],seeds=[0,1,2],budget_by_nodes=cfg['delta_budget_by_nodes'])
            actual=worker(request,cfg);assert actual['ok']
            for r in actual['results']:
                prior=next(x for x in stored['results'] if x['instance_id']==r['instance_id'] and x['seed']==r['seed'])
                assert r['best_tour']==prior['best_tour'] and r['original_length']==prior['original_length'] and r['delta_evaluations']==prior['delta_evaluations']
                replays.append(dict(method=method,instance_id=r['instance_id'],seed=r['seed'],exact=True))
    result=dict(status='passed',at_utc=stamp(),outputs=outputs,valid_outputs=valid,contexts=45,routes_recomputed=routes,max_route_cost_error=maximum,
        failed_evaluations_retained=failed_evaluations,decisions_recomputed=decisions,feedback_rows_recomputed=feedback_rows,bank_prediction_max_error=prediction_error,
        exact_prompt_token_and_parent_reconstruction=True,all_real_quality_selections_correct=True,all_programs_frozen_before_own_dev=True,
        independent_replays=replays,final_opened=False,wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,
        verification_source_sha256=digest(Path(__file__)))
    write(audit/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='independent_replays'},indent=2))


if __name__=='__main__':main()
