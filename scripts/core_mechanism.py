"""Fresh train-only multi-horizon mechanism diagnosis; no outcome-based selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from stage3_common import CODE,digest,read,write,stamp
from stage2_labels import states_for
from tsp_value.solver import clone,select_edge,penalize,run


def cycle_hash(t):
    a=np.asarray(t,dtype=np.int64)
    x=np.roll(a,-int(np.flatnonzero(a==0)[0])); b=a[::-1]
    y=np.roll(b,-int(np.flatnonzero(b==0)[0])); z=min(tuple(x),tuple(y))
    return hashlib.sha256(np.asarray(z,dtype='<i8').tobytes()).hexdigest()


def select_slice(rows,pool,offset,count):
    result=[]
    for n in (50,100):
        rr=[r for r in rows if r['project_use']==pool and r['nodes']==n][offset:offset+count]
        assert len(rr)==count;result.extend(rr)
    return result


def task(item,cfg):
    start,cpu=time.perf_counter(),time.process_time();states=[];replays=[]
    for event,s in states_for(item,cfg['penalty_events']):
        reference=select_edge(clone(s))
        options=sorted({tuple(sorted((int(u),int(v)))) for u,v in zip(s['tour'],np.roll(s['tour'],-1))}-{reference})
        rng=np.random.default_rng(int(hashlib.sha256((item['instance_id']+':'+str(event)).encode()).hexdigest()[:8],16))
        candidates=[options[j] for j in rng.choice(len(options),cfg['candidates'],replace=False)]
        branches=[]
        for index,edge in enumerate([reference]+candidates):
            branch=clone(s);branch['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state
            penalize(branch,edge);before=branch['delta_evaluations'];initial_moves=branch['moves'];records=[];calls=ties=0
            def observed_rule(d,p,state):
                nonlocal calls,ties
                scores=d/(1+p);calls+=1;ties+=int(np.sum(scores==scores.max())>1)
                return scores
            prior=0
            for sweeps in cfg['horizon_sweeps']:
                horizon=int(sweeps*len(s['pair_i']))
                run(branch,horizon-prior,rule=observed_rule);prior=horizon
                assert branch['delta_evaluations']-before==horizon
                records.append(dict(sweeps=sweeps,horizon=horizon,best_length=branch['best_length'],
                    current_length=branch['current_length'],best_cycle_sha256=cycle_hash(branch['best_tour']),
                    current_cycle_sha256=cycle_hash(branch['tour']),moves_after_intervention=branch['moves']-initial_moves,
                    penalty_events=branch['penalty_events'],future_priority_calls=calls,exact_max_tie_calls=ties))
            # One separate direct-budget check per node-size item at the first state.
            if event==cfg['penalty_events'][0] and index==0:
                direct=clone(s);direct['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state
                penalize(direct,edge);t=time.perf_counter();run(direct,prior)
                for k in ('tour','best_tour','penalties'):assert np.array_equal(branch[k],direct[k]),k
                for k in ('best_length','current_length','cursor','moves','penalty_events'):assert branch[k]==direct[k],k
                replays.append(dict(instance_id=item['instance_id'],state_index=event,edge=list(edge),horizon=prior,
                    direct_equals_multi_horizon=True,wall_seconds=time.perf_counter()-t))
            branches.append(dict(edge=list(edge),is_reference=index==0,checkpoints=records))
        comparisons=[]
        for j in range(1,len(branches)):
            for ref,candidate in zip(branches[0]['checkpoints'],branches[j]['checkpoints']):
                benefit=100*(ref['best_length']-candidate['best_length'])/s['best_length']
                comparisons.append(dict(candidate_index=j-1,edge=branches[j]['edge'],sweeps=ref['sweeps'],
                    benefit_pp=benefit,current_relative_difference_pp=100*(ref['current_length']-candidate['current_length'])/s['best_length'],
                    same_best_cycle=ref['best_cycle_sha256']==candidate['best_cycle_sha256'],
                    same_current_cycle=ref['current_cycle_sha256']==candidate['current_cycle_sha256'],
                    candidate_moves_after_intervention=candidate['moves_after_intervention']))
        states.append(dict(instance_id=item['instance_id'],nodes=item['nodes'],project_use=item['project_use'],state_index=event,
            normalization_best_length=s['best_length'],reference_edge=list(reference),branches=branches,comparisons=comparisons))
    return dict(instance_id=item['instance_id'],states=states,replays=replays,wall_seconds=time.perf_counter()-start,
        worker_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();cfg=read(CODE/'configs/core_mechanism_v1.json');out=a.output;out.mkdir(parents=True,exist_ok=False)
    start,cpu=time.perf_counter(),time.process_time()
    data=a.root/'data/prepared/stage1/project_instances.jsonl'
    assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    items=select_slice(rows,'value_train',cfg['per_size_offset'],cfg['instances_per_size'])
    write(out/'protocol_frozen.json',dict(frozen_at_utc=stamp(),config=cfg,config_sha256=digest(CODE/'configs/core_mechanism_v1.json'),
        normalized_data_sha256=digest(data),source_sha256={str(p.relative_to(CODE)):digest(p) for p in [Path(__file__),CODE/'src/tsp_value/solver.py',CODE/'scripts/stage2_labels.py']}))
    write(out/'selection.json',[{k:r[k] for k in ('instance_id','nodes','project_use')} for r in items])
    all_states=[];timing=[];replays=[]
    with gzip.open(out/'states.jsonl.gz','xt') as log,ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs={pool.submit(task,item,cfg):item['instance_id'] for item in items}
        for f in as_completed(jobs):
            result=f.result();ss=result.pop('states');replays.extend(result.pop('replays'));timing.append(result)
            for state in ss:log.write(json.dumps(state)+'\n');all_states.append(state)
            log.flush();print('completed '+str(len(timing))+'/'+str(len(items)),flush=True)
    groups=[]
    for n in (50,100):
        g=[s for s in all_states if s['nodes']==n]
        for h in cfg['horizon_sweeps']:
            rr=[dict(instance_id=s['instance_id'],state_index=s['state_index'],**r) for s in g for r in s['comparisons'] if r['sweeps']==h]
            zero=[r for r in rr if abs(r['benefit_pp'])<=1e-7]
            groups.append(dict(nodes=n,sweeps=h,states=len(g),comparisons=len(rr),zero_fraction=len(zero)/len(rr),
                informative_state_fraction=float(np.mean([any(abs(r['benefit_pp'])>1e-7 for r in s['comparisons'] if r['sweeps']==h) for s in g])),
                true_ties_with_different_current_cycle=sum(not r['same_current_cycle'] for r in zero),
                true_ties_with_different_current_length=sum(abs(r['current_relative_difference_pp'])>1e-7 for r in zero),
                mean_abs_benefit_pp=float(np.mean([abs(r['benefit_pp']) for r in rr]))))
    temporal=[]
    for n in (50,100):
        sequences=[sorted([r for r in s['comparisons'] if r['candidate_index']==j],key=lambda r:r['sweeps'])
            for s in all_states if s['nodes']==n for j in range(cfg['candidates'])]
        temporal.append(dict(nodes=n,action_sequences=len(sequences),
            any_nonzero_sequences=sum(any(abs(r['benefit_pp'])>1e-7 for r in seq) for seq in sequences),
            early_nonzero_but_final_zero=sum(any(abs(r['benefit_pp'])>1e-7 for r in seq[:-1]) and abs(seq[-1]['benefit_pp'])<=1e-7 for seq in sequences),
            best_quality_always_tied_but_current_ever_different=sum(all(abs(r['benefit_pp'])<=1e-7 for r in seq) and any(not r['same_current_cycle'] for r in seq) for seq in sequences)))
    final=[b['checkpoints'][-1] for s in all_states for b in s['branches']]
    summary=dict(config=cfg,groups=groups,temporal=temporal,states=len(all_states),original_instances=len(items),
        unique_continued_branches=sum(len(s['branches']) for s in all_states),horizons_per_branch=len(cfg['horizon_sweeps']),
        additional_direct_replay_branches=len(replays),replays=replays,
        future_priority_calls=sum(r['future_priority_calls'] for r in final),exact_max_tie_calls=sum(r['exact_max_tie_calls'] for r in final),
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in timing),timing=timing,
        peak_worker_rss_kib=max(r['peak_rss_kib'] for r in timing),final_opened=False,value_val_read=False,gpu_used=False)
    write(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('timing','replays')},indent=2))


if __name__=='__main__':main()
