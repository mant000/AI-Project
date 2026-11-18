"""Freeze new pool slices and an all-edge discovery bank before exact feedback lookup."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from stage3_common import CODE,read,write,digest,stamp,append
from core_mechanism import select_slice,cycle_hash
from stage2_labels import states_for
from stage3_worker import restore_state
from tsp_value.solver import clone,select_edge,penalize,run
from tsp_value.value_model import EdgeValue,state_features,collate
from tsp_value.data import coordinate_hash


def measured_state(row,cfg):
    t,c=time.perf_counter(),time.process_time();s=restore_state(row['state']);records=[]
    for edge in row['edges']:
        start=time.perf_counter();branch=clone(s)
        branch['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state
        penalize(branch,edge);before=branch['delta_evaluations'];calls=ties=0
        def observed(d,p,state):
            nonlocal calls,ties
            score=d/(1+p);calls+=1;ties+=int(np.sum(score==score.max())>1);return score
        h=cfg['horizon_by_nodes'][str(row['nodes'])];run(branch,h,rule=observed)
        assert branch['delta_evaluations']-before==h
        records.append(dict(edge=edge,best_length=branch['best_length'],current_length=branch['current_length'],
            best_tour=branch['best_tour'].tolist(),current_cycle_sha256=cycle_hash(branch['tour']),
            best_cycle_sha256=cycle_hash(branch['best_tour']),delta_evaluations=h,seed=cfg['continuation_seed'],
            future_priority_calls=calls,exact_max_tie_calls=ties,wall_seconds=time.perf_counter()-start))
    reference=records[-1]['best_length'];values=[100*(reference-r['best_length'])/s['best_length'] for r in records]
    return dict(bank_id=row['bank_id'],instance_id=row['instance_id'],nodes=row['nodes'],
        true_scores_pp=values,records=records,branches=len(records),wall_seconds=time.perf_counter()-t,
        worker_cpu_seconds=time.process_time()-c,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completed.json').exists(),'Do not rerun a completed bank'
    cfg=read(CODE/'configs/core_abc_v1.json');start,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1)
    assert not cfg['final_evaluation'] and not cfg['parameter_training']
    source=[Path(__file__),CODE/'scripts/core_mechanism.py',CODE/'scripts/stage3_worker.py',CODE/'src/tsp_value/value_model.py',CODE/'src/tsp_value/solver.py']
    frozen=dict(frozen_at_utc=stamp(),protocol=cfg,protocol_sha256=digest(CODE/'configs/core_abc_v1.json'),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in source})
    if (out/'protocol_frozen.json').exists():assert read(out/'protocol_frozen.json')['protocol']==cfg
    else:write(out/'protocol_frozen.json',frozen)
    data=a.root/'data/prepared/stage1/project_instances.jsonl'
    assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    pools={pool:select_slice(rows,pool,cfg[pool+'_offset_per_size'],cfg[pool+'_instances_per_size']) for pool in ('discovery','program_select','program_dev_eval')}
    old=read(CODE/'results/stage3/prepare-v1/pool_isolation.json')['selection']
    old_ids={r['instance_id'] for rr in old.values() for r in rr}
    sets={pool:{coordinate_hash(r['coords']) for r in rr} for pool,rr in pools.items()}
    for pool,rr in pools.items():
        assert not {r['instance_id'] for r in rr}&old_ids
        for other in pools:
            if other!=pool:assert not sets[pool]&sets[other]
    write(out/'pool_isolation.json',dict(normalized_data_sha256=digest(data),previous_stage3_selection_sha256=digest(CODE/'results/stage3/prepare-v1/pool_isolation.json'),
        previous_stage3_original_id_overlap=False,all_new_pool_coordinate_hashes_disjoint=True,
        selection={pool:[{k:r[k] for k in ('instance_id','nodes','project_use')} for r in rr] for pool,rr in pools.items()},
        unused_dev_after_this_experiment='remaining offsets 32..63 per size in fixed program_dev_eval pool; not opened',final_opened=False))
    bankpath=out/'bank_observables.jsonl'
    if bankpath.exists():bank=[json.loads(x) for x in bankpath.read_text().splitlines()]
    else:
        models=[];checkpoints=[]
        for seed in cfg['transformer_seeds']:
            p=a.root/f'checkpoints/stage2/training-v1/transformer-seed{seed}/best.pt'
            expected=read(CODE/f'results/stage2/training-v1/transformer-seed{seed}_summary.json')['checkpoint_sha256'];assert digest(p)==expected
            ck=torch.load(p,map_location='cpu');m=EdgeValue('transformer',**ck['config']['architecture']).cuda().eval();m.load_state_dict(ck['model']);models.append(m)
            checkpoints.append(dict(seed=seed,sha256=expected))
        bank=[]
        bank_items=[]
        for n in (50,100):bank_items.extend([r for r in pools['discovery'] if r['nodes']==n][:cfg['bank_instances_per_size']])
        for item in bank_items:
            for event,s in states_for(item,cfg['bank_penalty_events']):
                ref=select_edge(clone(s));edges=sorted({tuple(sorted((int(u),int(v)))) for u,v in zip(s['tour'],np.roll(s['tour'],-1))}-{ref})+[ref]
                f=state_features(s,edges)
                with torch.inference_mode():scores=np.stack([m(collate([f],'cuda')).cpu().numpy()[0] for m in models])
                snapshot={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in s.items() if k not in ('distance','pair_i','pair_j')}
                bank.append(dict(bank_id=item['instance_id']+':event'+str(event),instance_id=item['instance_id'],nodes=item['nodes'],project_use='discovery',
                    state_index=event,state=snapshot,reference_edge=list(ref),edges=[list(e) for e in edges],model_scores_pp=scores.tolist()))
        bankpath.write_text(''.join(json.dumps(r)+'\n' for r in bank))
        write(out/'bank_predictions_frozen.json',dict(frozen_at_utc=stamp(),bank_sha256=digest(bankpath),states=len(bank),checkpoints=checkpoints,
            future_values_accessed=False,all_current_undirected_edges_scored=True))
        del models;torch.cuda.empty_cache()
    assert digest(bankpath)==read(out/'bank_predictions_frozen.json')['bank_sha256']
    exact=out/'exact_states';exact.mkdir(exist_ok=True);finished=[]
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs={}
        for i,row in enumerate(bank):
            p=exact/f'state{i:03d}.json'
            if p.exists():r=read(p);assert r['bank_id']==row['bank_id'];finished.append(r)
            else:jobs[pool.submit(measured_state,row,cfg)]=p
        for f in as_completed(jobs):
            result=f.result();write(jobs[f],result);finished.append(result)
            print('measured '+str(len(finished))+'/'+str(len(bank))+' states',flush=True)
    by_id={r['bank_id']:r for r in finished};comparisons=[];summary_by_size=[]
    for row in bank:
        exactrow=by_id[row['bank_id']];true=np.asarray(exactrow['true_scores_pp']);pred=np.asarray(row['model_scores_pp']).mean(0);std=np.asarray(row['model_scores_pp']).std(0)
        for j,edge in enumerate(row['edges'][:-1]):comparisons.append(dict(bank_id=row['bank_id'],nodes=row['nodes'],edge=edge,
            measured_pp=float(true[j]),predicted_pp=float(pred[j]),optimization_std_pp=float(std[j]),
            provisional_prediction_exceeds_threshold=abs(pred[j])>max(.001,std[j])))
    for n in (50,100):
        rr=[r for r in comparisons if r['nodes']==n];zero=[r for r in rr if abs(r['measured_pp'])<=1e-7]
        summary_by_size.append(dict(nodes=n,nonreference_actions=len(rr),zero_fraction=len(zero)/len(rr),
            false_threshold_preferences_on_zero=sum(r['provisional_prediction_exceeds_threshold'] for r in zero),
            mean_abs_error_pp=float(np.mean([abs(r['measured_pp']-r['predicted_pp']) for r in rr]))))
    write(out/'teacher_diagnostic.json',dict(groups=summary_by_size,comparisons=comparisons,
        diagnostic_only_no_model_or_threshold_change=True,final_opened=False))
    resources=dict(started_at_utc=frozen['frozen_at_utc'],finished_at_utc=stamp(),wall_seconds=time.perf_counter()-start,
        process_cpu_seconds=time.process_time()-cpu,worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in finished),
        unique_measured_branches=sum(r['branches'] for r in finished),states=len(bank),
        future_priority_calls=sum(r['future_priority_calls'] for state in finished for r in state['records']),
        exact_max_tie_calls=sum(r['exact_max_tie_calls'] for state in finished for r in state['records']),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,final_opened=False)
    write(out/'resources.json',resources);write(out/'completed.json',dict(at_utc=stamp(),bank_sha256=digest(bankpath),protocol_sha256=frozen['protocol_sha256'],**resources))
    print(json.dumps(dict(resources=resources,diagnostic=summary_by_size),indent=2))


if __name__=='__main__':main()
