"""Independent outcome, source, prompt and feedback audit after core A/B/C freezing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import resource
import time
import numpy as np
from stage3_common import CODE,read,write,digest,stamp,evaluate
from core_mechanism import select_slice
from tsp_value.data import distances,length,validate_tour
from tsp_value.programs import canonical_program,INITIAL


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start,cpu=time.perf_counter(),time.process_time();out=a.output;out.mkdir(parents=True,exist_ok=False)
    study=a.base/'discovery-v1';prepare=a.base/'prepare-v1';cfg=read(prepare/'protocol_frozen.json')['protocol']
    completed=read(study/'completed.json');assert completed['logical_outputs']==len(cfg['arms'])*len(cfg['discovery_seeds'])*cfg['candidates_per_arm_per_seed']
    assert not completed['final_opened'] and not completed['parameter_training_started']
    provenance=read(study/'provenance.json');assert provenance['protocol_sha256']==digest(CODE/'configs/core_abc_v1.json')
    for rel,expected in provenance['source_sha256'].items():
        snapshot=a.base/'r1_source'/rel
        assert digest(snapshot if snapshot.exists() else CODE/rel)==expected,rel
    bank=[json.loads(x) for x in (prepare/'bank_observables.jsonl').read_text().splitlines()];bankbyid={r['bank_id']:r for r in bank}
    exact={r['bank_id']:r for r in [read(p) for p in (prepare/'exact_states').glob('*.json')]}
    assert digest(prepare/'bank_observables.jsonl')==provenance['bank_sha256']
    candidates=[read(p) for p in (study/'candidates').glob('*.json')];calls=[read(p) for p in (study/'generations').glob('*.json')]
    assert {r['candidate_id'] for r in candidates}=={r['candidate_id'] for r in calls}
    byid={r['candidate_id']:r for r in candidates};counts=Counter((r['discovery_seed'],r['arm']) for r in candidates)
    for seed in cfg['discovery_seeds']:
        for arm in cfg['arms']:assert counts[seed,arm]==cfg['candidates_per_arm_per_seed']
    rows=[json.loads(x) for x in (a.root/'data/prepared/stage1/project_instances.jsonl').read_text().splitlines()]
    byinstance={r['instance_id']:r for r in rows};protected={r['instance_id'] for r in rows if r['project_use']!='discovery'}
    for call in calls:
        candidate=byid[call['candidate_id']];assert call['sampling_seed']==candidate['discovery_seed']*1000+candidate['index']
        assert call['input_tokens']==len(call['input_token_ids']) and call['output_tokens']==len(call['output_token_ids'])
        prompt=call['rendered_prompt']
        prompt_ids=set(re.findall(r'joshi:tsp(?:50|100)_train_concorde\.txt:\d+',prompt))
        assert not prompt_ids&protected
        assert ('ADDITIONAL STATE VALUE FEEDBACK:' in prompt)==(candidate['arm']!='A')
        if candidate['arm']=='B':assert 'measured single-action branch rollouts' not in prompt
        if candidate['arm']=='C':assert 'Transformer ensemble predictions' not in prompt
        if candidate['valid']:assert canonical_program(candidate['source'])['program_sha256']==candidate['program_sha256']
    feedback_checked=0
    for c in candidates:
        if not c['valid'] or c['arm']=='A':continue
        for row in c['value_feedback']['states_detail']:
            state=bankbyid[row['bank_id']];i,j=state['edges'].index(row['parent_edge']),state['edges'].index(row['child_edge'])
            scores=np.asarray(exact[row['bank_id']]['true_scores_pp'])[None,:] if c['arm']=='C' else np.asarray(state['model_scores_pp'])
            expected=float((scores[:,j]-scores[:,i]).mean());assert abs(expected-row['child_minus_parent_pp'])<1e-10
            assert abs(float(scores[:,i].mean())-row['parent_value_pp'])<1e-10
            assert abs(float(scores[:,j].mean())-row['child_value_pp'])<1e-10;feedback_checked+=1
    frozen=read(study/'frozen_programs.json');dev=read(study/'dev_evaluation.json')
    assert frozen['protocol_sha256']==provenance['protocol_sha256'];assert len(frozen['programs'])==12
    assert dev['frozen_programs_sha256']==digest(study/'frozen_programs.json')
    for f in frozen['programs']:
        assert f['selected_program']['program_sha256']==f['elites'][int(np.argmin(f['select_mean_gaps_percent']))]
    expecteddev={r['instance_id'] for r in select_slice(rows,'program_dev_eval',cfg['program_dev_eval_offset_per_size'],cfg['program_dev_eval_instances_per_size'])}
    matrices={};checked=0;maximum=0.
    for p in (study/'evaluations').glob('*.json'):
        ev=read(p);assert ev['ok']
        for r in ev['results']:
            item=byinstance[r['instance_id']];assert item['project_use']==ev['pool']
            if ev['pool']=='program_dev_eval':assert r['instance_id'] in expecteddev
            if r['instance_id'] not in matrices:matrices[r['instance_id']]=distances(item['coords'])
            validate_tour(r['best_tour'],item['nodes']);actual=length(np.asarray(r['best_tour']),matrices[r['instance_id']])
            maximum=max(maximum,abs(actual-r['original_length']));assert abs(actual-r['original_length'])<1e-10
            assert abs(r['gap_percent']-100*(actual/item['reference_length']-1))<1e-10
            if ev['method']=='gls':assert r['delta_evaluations']==cfg['delta_budget_by_nodes'][str(item['nodes'])]
            checked+=1
    replay_items=select_slice(rows,'program_dev_eval',cfg['program_dev_eval_offset_per_size'],1)
    programs={f['selected_program']['program_sha256']:f['selected_program'] for f in frozen['programs']};base=canonical_program(INITIAL);programs[base['program_sha256']]=base
    replay=[]
    for key,program in programs.items():
        rr=evaluate(program['expression'],replay_items,cfg['program_dev_eval_solver_seeds'],cfg);write(out/f'replay-{key}.json',rr);assert rr['ok']
        original=next(r['evaluation'] for r in dev['runs'] if r.get('program_sha256')==key or (key==base['program_sha256'] and r['label']=='gls'))
        old={(r['instance_id'],r['seed']):r for r in original['results']}
        for row in rr['results']:
            for field in ('best_tour','original_length','gap_percent','delta_evaluations','penalty_events'):assert row[field]==old[row['instance_id'],row['seed']][field]
        replay.append(dict(program_sha256=key,routes=len(rr['results']),worker_cpu_seconds=rr['cpu_seconds']))
    summary=dict(status='passed',at_utc=stamp(),logical_outputs=len(candidates),valid_outputs=sum(r['valid'] for r in candidates),
        endpoint_routes_recomputed=checked,max_route_cost_error=maximum,same_state_feedback_rows_verified=feedback_checked,
        paired_generation_seeds_verified=True,protected_pool_prompt_leakage=False,all_12_selections_frozen_before_dev=True,
        protocol_and_executed_source_hashes_verified=True,new_dev_slice_verified=True,independent_frozen_program_replays=replay,
        final_opened=False,wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
    write(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
