"""Freeze protocol, verify live encoding, build discovery-only bank and real probe."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import torch
from stage3_common import CODE,read,write,digest,stamp,choose,worker
from stage2_labels import states_for
from tsp_value.value_model import EdgeValue,state_features,features,collate
from tsp_value.programs import canonical_program,INITIAL


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    out=a.output; out.mkdir(parents=True,exist_ok=False); cfg=read(CODE/'configs/stage3_protocol.json')
    start,cpu=time.perf_counter(),time.process_time(); torch.set_num_threads(1)
    assert cfg['final_evaluation'] is False and cfg['sft_grpo'] is False
    # This record is persisted before any discovery outcomes or probe labels exist.
    provenance=dict(frozen_at_utc=stamp(),protocol=cfg,protocol_sha256=digest(CODE/'configs/stage3_protocol.json'),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in sorted(CODE.glob('scripts/stage3_*.py'))},
        solver_sha256=digest(CODE/'src/tsp_value/solver.py'),encoder_sha256=digest(CODE/'src/tsp_value/value_model.py'),
        program_language_sha256=digest(CODE/'src/tsp_value/programs.py'))
    write(out/'protocol_frozen.json',provenance)
    live=dict(at_utc=stamp(),processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,args','--sort=-pcpu'],text=True),
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
        disk=subprocess.check_output(['df','-h',str(a.root),'/autodl-fs/data'],text=True))
    write(out/'remote_preflight.json',live)
    manifest=read(a.root/'artifacts/qwen_download.json')
    assert manifest['status']=='completed_verified' and manifest['author_weight_hashes_verified']
    assert manifest['config']['revision']==cfg['qwen_revision']
    checks=[]
    for file in manifest['files']:
        p=Path(manifest['snapshot_path'])/file['name']
        check=dict(file=file['name'],actual_bytes=p.stat().st_size,sha256=digest(p))
        assert check['actual_bytes']==file['bytes'] and check['sha256']==file['sha256']
        checks.append(check)
    write(out/'qwen_weight_verification.json',dict(snapshot=manifest['snapshot_path'],files=checks,checked_at_utc=stamp()))
    data=a.root/'data/prepared/stage1/project_instances.jsonl'
    assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    rows=[json.loads(line) for line in data.read_text().splitlines()]
    pools={pool:choose(rows,pool,cfg[key]) for pool,key in [('discovery','discovery_instances_per_size'),
        ('program_select','program_select_instances_per_size'),('program_dev_eval','program_dev_eval_instances_per_size')]}
    identities={pool:{r['instance_id'] for r in rr} for pool,rr in pools.items()}
    from tsp_value.data import coordinate_hash
    hashes={pool:{coordinate_hash(r['coords']) for r in rr} for pool,rr in pools.items()}
    f32={pool:{coordinate_hash(r['coords'],True) for r in rr} for pool,rr in pools.items()}
    for left in pools:
        for right in pools:
            if left!=right:
                assert not identities[left]&identities[right] and not hashes[left]&hashes[right] and not f32[left]&f32[right]
    write(out/'pool_isolation.json',dict(data_sha256=digest(data),manifest_sha256=digest(CODE/'splits/manifest.csv'),
        selection={pool:[{k:r[k] for k in ('instance_id','nodes','project_use')} for r in rr] for pool,rr in pools.items()},
        all_selected_original_ids_and_exact_and_float32_coordinate_hashes_disjoint=True,
        final_content_read=False,value_val_content_read=False))
    original=subprocess.check_output(['git','show','a284c877caed5bcec0e39b74d520f5114175d8eb:src/tsp_value/value_model.py'],cwd=CODE,text=True)
    original_namespace={}; exec(compile(original,'<frozen-stage2-encoder>','exec'),original_namespace)
    with gzip.open(CODE/'results/stage2/train/states.jsonl.gz','rt') as f: train=sorted([json.loads(x) for x in f],key=lambda r:(r['nodes'],r['instance_id'],r['state_index']))
    maximum=0.; checked=0
    for r in train:
        assert r['project_use']=='value_train'
        old=original_namespace['features'](r); new=features(r)
        live=state_features(r['state'],r['candidates']+[r['reference_edge']])
        assert 'y' not in live
        for k in old:
            assert np.array_equal(old[k],new[k]),k
            if k!='y': assert np.array_equal(old[k],live[k]),k
            maximum=max(maximum,float(np.max(np.abs(old[k]-new[k]))))
        checked+=1
    security=[]
    for bad in ["__import__('os').system('id')",'d.__class__','d[0]','[d for x in p]','d ** (10 ** 10)',
                'open(1)','(lambda: d)()', 'def priority(d,p,mean_d,max_d,lam,progress,stagnation,best_ratio):\n import os\n return d']:
        try: canonical_program(bad)
        except Exception: security.append(dict(expression=bad,rejected=True))
        else: raise AssertionError('Unsafe syntax accepted: '+bad)
    # Also exercise valid, scalar, nonfinite and overflow handling in the isolated worker.
    cases=[(INITIAL,True),('1',True),('d / 0',False),('exp(100 * p + 100)',False)]
    isolated=[]
    for expr,expected in cases:
        result=worker(dict(mode='endpoint',expression=expr,items=pools['discovery'][:1],seeds=[0],
                           budget_by_nodes=cfg['delta_budget_by_nodes']),cfg)
        # exp(100) is finite before penalties, but overflows during the full run.
        assert result['ok']==expected,(expr,result)
        isolated.append(dict(expression=expr,expected_ok=expected,**result))
    write(out/'security_checks.json',dict(ast_rejections=security,isolated_runtime=isolated))
    models=[]; ckchecks=[]
    for seed in cfg['transformer_seeds']:
        p=a.root/f'checkpoints/stage2/training-v1/transformer-seed{seed}/best.pt'
        expected=read(CODE/f'results/stage2/training-v1/transformer-seed{seed}_summary.json')['checkpoint_sha256']
        assert digest(p)==expected
        ck=torch.load(p,map_location='cpu'); m=EdgeValue('transformer',**ck['config']['architecture']).cuda().eval()
        m.load_state_dict(ck['model']); models.append(m); ckchecks.append(dict(seed=seed,path=str(p),sha256=expected,selected_epoch=ck['epoch']))
    prediction_error=0.
    with torch.inference_mode():
        for r in train[::64]:
            old=original_namespace['features'](r); new=state_features(r['state'],r['candidates']+[r['reference_edge']])
            for model in models:
                x=model(collate([old],'cuda')); y=model(collate([new],'cuda'))
                prediction_error=max(prediction_error,float((x-y).abs().max()))
    assert prediction_error==0
    write(out/'feature_equivalence.json',dict(reference_commit='a284c877caed5bcec0e39b74d520f5114175d8eb',
        original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),training_states_checked=checked,
        tensor_max_abs_error=maximum,prediction_max_abs_error_pp=prediction_error,
        training_equivalence_only=True,live_encoding_has_no_label_key=True,checkpoints=ckchecks))
    bank=[]
    for item in choose(rows,'discovery',cfg['bank_instances_per_size']):
        for event,s in states_for(item,cfg['bank_state_penalty_events']):
            from tsp_value.solver import clone,select_edge
            ref=select_edge(clone(s))
            edges=sorted({tuple(sorted((int(x),int(y)))) for x,y in zip(s['tour'],np.roll(s['tour'],-1))}-{ref})+[ref]
            f=state_features(s,edges)
            with torch.inference_mode(): pred=np.stack([m(collate([f],'cuda')).cpu().numpy()[0] for m in models])
            snapshot={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in s.items() if k not in ('distance','pair_i','pair_j')}
            bank.append(dict(bank_id=item['instance_id']+':event'+str(event),instance_id=item['instance_id'],nodes=item['nodes'],
                project_use='discovery',state_index=event,state=snapshot,reference_edge=list(ref),edges=[list(e) for e in edges],
                model_scores_pp=pred.tolist(),mean_scores_pp=pred.mean(0).tolist(),model_std_pp=pred.std(0).tolist()))
    # Bank, all-edge predictions and immutable hashes are persisted BEFORE the real probe.
    bankpath=out/'bank.jsonl'; bankpath.write_text(''.join(json.dumps(r)+'\n' for r in bank))
    write(out/'bank_frozen.json',dict(at_utc=stamp(),sha256=digest(bankpath),states=len(bank),instances=len({r['instance_id'] for r in bank}),
        all_current_undirected_edges_scored=True,all_predictions_computed_before_probe_labels=True,
        encoder_used=provenance['encoder_sha256'],checkpoints=ckchecks))
    probe_ids={r['instance_id'] for r in choose(rows,'discovery',cfg['probe_instances_per_size'])}
    probes=[]
    for row in bank:
        if row['instance_id'] not in probe_ids: continue
        edge_list=row['edges'][:-1]
        rng=np.random.default_rng(int(hashlib.sha256((row['instance_id']+':'+str(row['state_index'])).encode()).hexdigest()[:8],16))
        edges=[edge_list[i] for i in rng.choice(len(edge_list),cfg['probe_candidates_uniform'],replace=False)]
        top=row['edges'][int(np.argmax(row['mean_scores_pp']))]
        if top!=row['reference_edge'] and top not in edges: edges.append(top)
        probes.append(dict(**row,probe_edges=edges))
    result=worker(dict(mode='probe',states=probes,seeds=cfg['continuation_seeds'],horizon_by_nodes=cfg['horizon_by_nodes']),cfg)
    write(out/'probe_raw.json',result); assert result['ok'],result
    comparisons=[]
    for row,rr in zip(probes,result['results']):
        actual=np.asarray([x['relative_benefit'] for x in rr['records']]).reshape(2,-1)*100
        for j,edge in enumerate(row['probe_edges']):
            idx=row['edges'].index(edge); pred=row['mean_scores_pp'][idx]; std=row['model_std_pp'][idx]; true=float(actual[:,j].mean())
            comparisons.append(dict(bank_id=row['bank_id'],edge=edge,predicted_pp=pred,ensemble_std_pp=std,true_pp=true,
                seeds_identical=bool(actual[0,j]==actual[1,j]),confident=abs(pred)>max(.001,std)))
    relevant=[r for r in comparisons if abs(r['true_pp'])>1e-7 and r['confident']]
    opposite=sum(r['predicted_pp']*r['true_pp']<0 for r in relevant)
    severe=len(relevant)>=4 and opposite/len(relevant)>.5
    summary=dict(states=len(probes),original_instances=len(probe_ids),branches=sum(r['branches_executed'] for r in result['results']),
        candidate_comparisons=len(comparisons),zero_fraction=float(np.mean([abs(r['true_pp'])<=1e-7 for r in comparisons])),
        confident_nonzero_pairs=len(relevant),opposite_signs=opposite,severe_mismatch_gate=severe,
        identical_continuation_seed_fraction=float(np.mean([r['seeds_identical'] for r in comparisons])),
        mean_absolute_error_pp=float(np.mean([abs(r['predicted_pp']-r['true_pp']) for r in comparisons])),
        comparisons=comparisons,warning='Sparse, discovery-only diagnostic; two continuation seeds may be redundant. No true probe outcome enters either LLM prompt.')
    write(out/'probe_summary.json',summary)
    torch.cuda.synchronize()
    write(out/'prepare_resources.json',dict(started_at_utc=provenance['frozen_at_utc'],finished_at_utc=stamp(),
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        child_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_used=True,scope='Encoding equivalence, all-edge bank inference, real probe; no 7B generation or training'))
    print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2),flush=True)


if __name__=='__main__': main()
