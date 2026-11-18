"""Independently reprice every completed five-seed control and audit timing availability."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import time
import numpy as np
from stage3_common import CODE,read,write,digest,stamp
from tsp_value.data import distances,length,validate_tour
from tsp_value.programs import canonical_program,compile_rule,INITIAL
from tsp_value.solver import create_state,run


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    start,cpu=time.perf_counter(),time.process_time();out=CODE/'results/solver-controls/dev-v1'
    complete=read(out/'completed.json');assert complete['status']=='completed'
    protocol=read(out/'protocol_frozen.json');cfg=protocol['protocol'];selection=read(out/'selection.json')
    for name,h in protocol['source_sha256'].items():assert digest(CODE/name)==h,name
    data=a.root/'data/prepared/stage1/project_instances.jsonl';assert digest(data)==protocol['data_sha256']
    import json
    rows={r['instance_id']:r for r in map(json.loads,data.read_text().splitlines())}
    seeds=cfg['solver_seeds'];assert seeds==[0,1,2,3,4]
    assert len(selection['exact'])==64 and len(selection['timing'])==4
    assert all(rows[k]['project_use']=='program_dev_eval' for k in selection['exact'])
    errors=[];routes=0;timed=0;maximum=0.;overruns=[];profiles=[];coverage={}
    for kind in ('exact','timing','external'):
        for file in sorted((out/kind).glob('*.json')):
            result=read(file);assert result['ok'];rr=result['results'];assert len(rr)==(4 if kind=='timing' else 64)*5
            expected={(k,s) for k in selection['timing' if kind=='timing' else 'exact'] for s in seeds}
            assert {(r['instance_id'],r['seed']) for r in rr}==expected
            coverage[f'{kind}/{file.name}']=len(rr)
            for r in rr:
                item=rows[r['instance_id']];d=distances(item['coords'])
                records=r['checkpoints'] if kind=='timing' else [r]
                if kind=='timing':
                    assert r['error'] is None
                    assert [q['deadline_seconds'] for q in records]==cfg['deadline_seconds']
                    assert records[-1]['original_length']<=records[0]['original_length']+1e-10
                    timed+=1
                for q in records:
                    t=np.asarray(q['best_tour']);validate_tour(t,item['nodes']);actual=length(t,d)
                    error=abs(actual-q['original_length']);maximum=max(maximum,error);assert error<1e-10
                    assert abs(q['gap_percent']-100*(actual/item['reference_length']-1))<1e-10
                    if kind=='timing':
                        assert q['available_by_deadline'] and q['available_seconds']<=q['deadline_seconds']
                        assert q['delta_evaluations']<=r['actual_delta_evaluations']
                        overruns.append(q['atomic_overshoot_seconds'])
                    elif kind=='exact':
                        cap=cfg['delta_budget_by_nodes'][str(item['nodes'])]
                        if file.stem.startswith('gls-') or file.stem in ('native_gls','neural-direct'):assert q['delta_evaluations']==cap
                        else:assert q['delta_evaluations']<=cap
                    elif file.stem=='lkh1000':
                        scale=q['integer_input_scale'];integer=np.floor(d*scale+.5).astype(np.int64)
                        assert int(integer[t,np.roll(t,-1)].sum())==q['integer_input_route_cost']
                        assert abs(q['integer_input_route_cost']/scale-actual)<=q['input_quantization_route_error_bound']+1e-9
                    else:assert q['model_input_reference_tour'] is False
                    routes+=1
                if 'profile' in r:
                    p=r['profile'];assert 0<=p['changed_actions']<=p['calls'];assert p['total_wall_seconds']>=p['inference_wall_seconds'];profiles.append(p)
    assert len(list((out/'exact').glob('*.json')))==len(protocol['program_hashes'])+4
    assert len(list((out/'timing').glob('*.json')))==len(protocol['program_hashes'])+4
    assert len(list((out/'external').glob('*.json')))==3
    initial=canonical_program(INITIAL)['program_sha256'];native=read(out/'exact/native_gls.json')['results']
    generated=read(out/'exact'/f'gls-{initial}.json')['results'];assert all(x['best_tour']==y['best_tour'] for x,y in zip(native,generated))
    # Re-run three methods on both sizes, including a non-initial generated rule.
    from solver_dev_benchmark import collect_programs
    programs,_=collect_programs(read(CODE/'configs/solver_controls_v1.json'),cfg)
    chosen=next(k for k in programs if k!=initial);replays=[]
    for n in (50,100):
        item=next(rows[k] for k in selection['exact'] if rows[k]['nodes']==n)
        for label,method,rule in [('native_gls','gls',None),('two_opt','two_opt',None),(f'gls-{chosen}','gls',compile_rule(programs[chosen]['expression']))]:
            state=create_state(item['coords'],0,method);run(state,cfg['delta_budget_by_nodes'][str(n)],rule)
            r=next(r for r in read(out/'exact'/f'{label}.json')['results'] if r['instance_id']==item['instance_id'] and r['seed']==0)
            assert state['best_tour'].tolist()==r['best_tour'] and state['delta_evaluations']==r['delta_evaluations']
            replays.append(dict(method=label,nodes=n,exact=True))
    write(out/'verification.json',dict(status='passed',at_utc=stamp(),routes_recomputed=routes,max_cost_error=maximum,
        coverage=coverage,original_instances=64,solver_seeds=seeds,timing_pilot_original_instances=4,timing_trajectories=timed,
        maximum_atomic_overshoot_seconds=max(overruns),hard_process_stop_claim=False,independent_replays=replays,
        native_and_compiled_standard_exactly_equal=True,all_programs_retained=True,final_opened=False,
        wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,source_sha256=digest(Path(__file__))))
    print('dev verification passed',routes,'routes',flush=True)


if __name__=='__main__':main()
