"""Independent endpoint, population lineage and selection audit for the supplementary control."""
import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import re
import time
import numpy as np
from stage3_common import CODE,read,write,digest,stamp,worker
from training_study import TrainingStudy
from tsp_value.programs import canonical_program,INITIAL
from tsp_value.data import distances,length,validate_tour


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    out=CODE/'results/eoh-control-v1';assert read(out/'completed.json')['status']=='completed'
    assert not (out/'verification.json').exists();start,cpu=time.perf_counter(),time.process_time()
    provenance=read(out/'protocol_frozen.json');cfg=provenance['protocol']['eoh_style']
    for name,expected in provenance['source_sha256'].items():assert digest(CODE/name)==expected,name
    assert digest(CODE/'configs/solver_controls_v1.json')==provenance['protocol_sha256']
    study=TrainingStudy(a.root,out,CODE/'results/main/prepare-v1');maincfg=study.cfg
    rows={r['instance_id']:r for rr in study.pools.values() for r in rr};ds={k:distances(r['coords']) for k,r in rows.items()}
    frozen=read(out/'frozen_programs.json');routes=0
    for path in (out/'evaluations').glob('*.json'):
        ev=read(path);allowed={r['instance_id'] for r in study.pools[ev['pool']]}
        if ev['pool']=='program_dev_eval':assert ev['at_utc']>=frozen['at_utc']
        for r in ev.get('results',[]):
            assert r['instance_id'] in allowed and r['project_use']==ev['pool'];validate_tour(r['best_tour'],r['nodes'])
            actual=length(r['best_tour'],ds[r['instance_id']]);assert abs(actual-r['original_length'])<1e-10
            assert abs(100*(actual/rows[r['instance_id']]['reference_length']-1)-r['gap_percent'])<1e-10
            assert r['delta_evaluations']==maincfg['delta_budget_by_nodes'][str(r['nodes'])];routes+=1
    def endpoint(p,pool,seeds):
        path=out/'evaluations'/f'{pool}-gls-{p["program_sha256"]}-{"-".join(map(str,seeds))}.json'
        assert path.exists();return study.endpoint(p,pool,seeds)[0]
    initial=canonical_program(INITIAL);populations={};outputs=valid=0
    protected={r['instance_id'] for r in [json.loads(x) for x in (a.root/'data/prepared/stage1/project_instances.jsonl').read_text().splitlines()] if r['project_use']!='discovery'}
    for seed in cfg['discovery_seeds']:
        population=[initial];rng=np.random.default_rng(seed+7331)
        def score(p):return endpoint(p,'discovery',maincfg['endpoint_solver_seeds'])['mean_gap_percent']
        for i in range(cfg['outputs_per_seed']):
            population=sorted(population,key=score);operator=cfg['operators_cycle'][i%5]
            weights=1/(np.arange(len(population))+1);weights/=weights.sum();count=min(2 if operator in ('e1','e2') else 1,len(population))
            indices=rng.choice(len(population),count,replace=False,p=weights);expected_parents=[population[j]['program_sha256'] for j in indices]
            cid=f'eoh-seed{seed}-candidate{i:03d}';c=read(out/'candidates'/f'{cid}.json');raw=read(out/'generations'/f'{cid}.json')
            assert c['parent_sha256']==expected_parents and c['operator']==operator and raw['sampling_seed']==seed*1000+i
            assert raw['input_tokens']==len(raw['input_token_ids']) and raw['output_tokens']==len(raw['output_token_ids'])
            prompt=raw['rendered_prompt'];assert not protected & set(re.findall(r'joshi:[A-Za-z0-9_.-]+:\d+',prompt))
            assert 'ADDITIONAL STATE VALUE FEEDBACK' not in prompt and 'value_feedback' not in c
            if c['valid']:
                p=canonical_program(raw['response']);assert p['program_sha256']==c['program_sha256']
                ev=endpoint(p,'discovery',maincfg['endpoint_solver_seeds']);assert ev['ok'] and abs(ev['mean_gap_percent']-c['mean_discovery_gap_percent'])<1e-12
                if p['program_sha256'] not in {x['program_sha256'] for x in population}:population.append(p)
                population=sorted(population,key=score)[:cfg['population']];valid+=1
            else:assert c.get('error')
            outputs+=1
        populations[seed]=population
    assert outputs==72 and valid==read(out/'completed.json')['valid'];replays=[]
    for r in frozen['programs']:
        population=populations[r['discovery_seed']];gaps=[]
        for p in population:
            ev=endpoint(p,'program_select',maincfg['program_select_solver_seeds']);gaps.append(ev['mean_gap_percent'] if ev['ok'] else 1e100)
        assert r['program']==population[int(np.argmin(gaps))]
        assert r['elites']==[p['program_sha256'] for p in population]
        stored=endpoint(r['program'],'program_dev_eval',maincfg['program_dev_eval_solver_seeds'])
        for n in (50,100):
            item=next(x for x in study.pools['program_dev_eval'] if x['nodes']==n)
            actual=worker(dict(mode='endpoint',method='gls',expression=r['program']['expression'],items=[item],seeds=[0],budget_by_nodes=maincfg['delta_budget_by_nodes']),maincfg)
            assert actual['ok'];prior=next(x for x in stored['results'] if x['instance_id']==item['instance_id'] and x['seed']==0)
            assert actual['results'][0]['best_tour']==prior['best_tour'];replays.append(dict(seed=r['discovery_seed'],instance_id=item['instance_id'],exact=True))
    result=dict(status='passed',at_utc=stamp(),outputs=outputs,valid=valid,routes_recomputed=routes,max_route_cost_error=0.,
        parent_population_rng_and_real_quality_selection_verified=True,all_three_programs_frozen_before_own_dev=True,
        no_state_or_dev_feedback_in_discovery=True,independent_replays=replays,
        wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,final_opened=False,
        verification_source_sha256=digest(Path(__file__)))
    write(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
