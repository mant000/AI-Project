"""Dense-current-edge pilot; reuse existing stage2 action costs and run only missing edges."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import json
from pathlib import Path
import resource
import time
import traceback
import numpy as np
from stage3_common import CODE,read,write,digest,stamp
from core_mechanism import select_slice
from stage2_labels import states_for
from stage3_worker import restore_state
from tsp_value.solver import clone,select_edge,penalize,run


def label_state(row,cfg):
    t,c=time.perf_counter(),time.process_time();s=restore_state(row['state']);records=[];reused=0
    for edge in row['edges']:
        key=','.join(map(str,edge))
        if key in row['cached']:
            length=row['cached'][key];record=dict(edge=edge,best_length=length,reused=True,best_tour=None);reused+=1
        else:
            branch=clone(s);branch['rng_state']=np.random.default_rng(cfg['continuation_seed']).bit_generator.state
            penalize(branch,edge);before=branch['delta_evaluations'];h=cfg['horizon_by_nodes'][str(row['nodes'])];start=time.perf_counter();run(branch,h)
            assert branch['delta_evaluations']-before==h
            record=dict(edge=edge,best_length=branch['best_length'],best_tour=branch['best_tour'].tolist(),reused=False,
                delta_evaluations=h,wall_seconds=time.perf_counter()-start)
        records.append(record)
    reference=records[-1]['best_length'];y=[float(100*(reference-r['best_length'])/s['best_length']) for r in records]
    return dict(**row,records=records,true_scores_pp=y,new_branches=len(records)-reused,reused_branches=reused,
        worker_cpu_seconds=time.process_time()-c,wall_seconds=time.perf_counter()-t,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def bank_row(item,event,s,reference,cached):
    edges=sorted({tuple(sorted((int(u),int(v)))) for u,v in zip(s['tour'],np.roll(s['tour'],-1))}-{tuple(reference)})+[tuple(reference)]
    snapshot={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in s.items() if k not in ('distance','pair_i','pair_j')}
    return dict(bank_id=item['instance_id']+':event'+str(event),instance_id=item['instance_id'],nodes=item['nodes'],
        project_use=item['project_use'],state_index=event,state=snapshot,edges=[list(e) for e in edges],cached=cached)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'completed.json').exists();cfg=read(CODE/'configs/core_teacher_v2.json')
    start,cpu=time.perf_counter(),time.process_time();frozen=dict(at_utc=stamp(),protocol=cfg,protocol_sha256=digest(CODE/'configs/core_teacher_v2.json'),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in [Path(__file__),CODE/'src/tsp_value/solver.py',CODE/'scripts/stage2_labels.py']})
    if (out/'protocol_frozen.json').exists():assert read(out/'protocol_frozen.json')['protocol']==cfg
    else:write(out/'protocol_frozen.json',frozen)
    data=a.root/'data/prepared/stage1/project_instances.jsonl';assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    items=[json.loads(x) for x in data.read_text().splitlines()];byid={r['instance_id']:r for r in items}
    rows_path=out/'bank_observables.jsonl'
    if rows_path.exists():bank=[json.loads(x) for x in rows_path.read_text().splitlines()]
    else:
        path=CODE/'results/stage2/train/states.jsonl.gz'
        with gzip.open(path,'rt') as f:previous=[json.loads(x) for x in f]
        wanted=set()
        for n in (50,100):wanted.update(sorted({r['instance_id'] for r in previous if r['nodes']==n},key=lambda iid:int(iid.rsplit(':',1)[1]))[:cfg['train']['instances_per_size']])
        bank=[]
        for r in previous:
            if r['instance_id'] not in wanted:continue
            assert r['project_use']=='value_train' and r['horizon']==cfg['horizon_by_nodes'][str(r['nodes'])]
            s=restore_state(r['state']);cached={','.join(map(str,sorted(b['edge']))):b['best_length'] for b in r['branch_measurements'] if b['seed']==cfg['continuation_seed']}
            assert len(cached)==9
            row=bank_row(byid[r['instance_id']],r['state_index'],s,r['reference_edge'],cached);row['role']='train';bank.append(row)
        for role in ('validation','heldout'):
            settings=cfg[role]
            for item in select_slice(items,settings['pool'],settings['offset_per_size'],settings['instances_per_size']):
                for event,s in states_for(item,cfg['state_events']):
                    ref=select_edge(clone(s));row=bank_row(item,event,s,ref,{});row['role']=role;bank.append(row)
        bank.sort(key=lambda r:(r['role'],r['nodes'],int(r['instance_id'].rsplit(':',1)[1]),r['state_index']))
        rows_path.write_text(''.join(json.dumps(r)+'\n' for r in bank))
        sets={role:{r['instance_id'] for r in bank if r['role']==role} for role in ('train','validation','heldout')}
        assert not sets['train']&sets['validation'] and not sets['validation']&sets['heldout']
        write(out/'selection.json',dict(roles={role:sorted(ids) for role,ids in sets.items()},state_counts={role:sum(r['role']==role for r in bank) for role in sets},
            old_train_labels_sha256=digest(path),all_current_edges_scored=True,old_compatible_seed71_costs_reused=True,
            validation_and_heldout_are_fresh_value_val_slices=True,final_opened=False))
    exact=out/'states';exact.mkdir(exist_ok=True);finished=[]
    try:
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            jobs={}
            for i,row in enumerate(bank):
                p=exact/f'state{i:04d}.json'
                if p.exists():result=read(p);assert result['bank_id']==row['bank_id'] and result['role']==row['role'];finished.append(result)
                else:jobs[pool.submit(label_state,row,cfg)]=p
            for f in as_completed(jobs):
                result=f.result();write(jobs[f],result);finished.append(result)
                if len(finished)%16==0:print('completed '+str(len(finished))+'/'+str(len(bank))+' dense states',flush=True)
        roles={}
        for role in ('train','validation','heldout'):
            group=[r for r in finished if r['role']==role]
            roles[role]=dict(original_instances=len({r['instance_id'] for r in group}),states=len(group),
                action_values=sum(len(r['edges']) for r in group),new_branches=sum(r['new_branches'] for r in group),
                reused_branches=sum(r['reused_branches'] for r in group),worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in group))
        summary=dict(at_utc=stamp(),protocol_sha256=frozen['protocol_sha256'],roles=roles,
            new_branches=sum(r['new_branches'] for r in finished),reused_branches=sum(r['reused_branches'] for r in finished),
            worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in finished),wall_seconds=time.perf_counter()-start,
            process_cpu_seconds=time.process_time()-cpu,peak_worker_rss_kib=max(r['peak_rss_kib'] for r in finished),
            gpu_used=False,final_opened=False,heldout_model_metrics_computed=False)
        write(out/'completed.json',summary);print(json.dumps(summary,indent=2))
    except Exception:
        write(out/'failed_attempt.json',dict(at_utc=stamp(),error=traceback.format_exc(),wall_seconds=time.perf_counter()-start,
            process_cpu_seconds=time.process_time()-cpu,saved_states=len(finished)))
        raise


if __name__=='__main__':main()
