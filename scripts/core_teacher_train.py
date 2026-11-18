"""Same-data dense-action MLP/GNN/Transformer pilot, followed by untouched teacher holdout."""
import argparse
from collections import defaultdict
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from stage3_common import CODE,read,write,digest,stamp
from tsp_value.value_model import state_features,collate,objective
from tsp_value.value_gnn import create_value_model
from stage2_train import save_checkpoint,restore,check_resume,check_symmetry


def load_role(data,role):
    import json
    observables=[json.loads(x) for x in (data/'bank_observables.jsonl').read_text().splitlines()]
    rows=[read(data/'states'/f'state{i:04d}.json') for i,r in enumerate(observables) if r['role']==role]
    rows=sorted(rows,key=lambda r:(r['nodes'],r['instance_id'],r['state_index']))
    fs=[]
    for row in rows:
        f=state_features(row['state'],row['edges']);f['y']=np.asarray(row['true_scores_pp'],dtype=np.float32);fs.append(f)
    return rows,fs


def evaluate(rows,fs,predictions):
    raw=[]
    for row,f,p in zip(rows,fs,predictions):
        y=f['y'];p=np.asarray(p);ties=np.flatnonzero(p>=p.max()-1e-9);choice=len(p)-1 if len(p)-1 in ties else int(ties[0])
        valid=np.abs(y[:,None]-y[None,:])>1e-7;pd=p[:,None]-p[None,:];yd=y[:,None]-y[None,:]
        acc=float(np.mean(np.where(np.abs(pd[valid])<=1e-9,.5,(pd[valid]*yd[valid]>0).astype(float)))) if valid.any() else None
        rho=float(spearmanr(p,y).statistic) if np.ptp(y)>1e-7 and np.ptp(p)>1e-9 else None
        raw.append(dict(instance_id=row['instance_id'],nodes=row['nodes'],state_index=row['state_index'],
            informative=bool(np.ptp(y)>1e-7),chosen_index=choice,selected_benefit_pp=float(y[choice]),
            oracle_benefit_pp=float(y.max()),top1_regret_pp=float(y.max()-y[choice]),spearman=rho,non_tied_pair_accuracy=acc,
            predictions_pp=p.tolist(),labels_pp=y.tolist()))
    def aggregate(rr):
        return dict(states=len(rr),original_instances=len({r['instance_id'] for r in rr}),informative_states=sum(r['informative'] for r in rr),
            top1_regret_pp=float(np.mean([r['top1_regret_pp'] for r in rr])),selected_benefit_pp=float(np.mean([r['selected_benefit_pp'] for r in rr])),
            oracle_benefit_pp=float(np.mean([r['oracle_benefit_pp'] for r in rr])),
            spearman=float(np.mean([r['spearman'] for r in rr if r['spearman'] is not None])) if any(r['spearman'] is not None for r in rr) else None,
            non_tied_pair_accuracy=float(np.mean([r['non_tied_pair_accuracy'] for r in rr if r['non_tied_pair_accuracy'] is not None])) if any(r['non_tied_pair_accuracy'] is not None for r in rr) else None)
    result=aggregate(raw);result['by_nodes']={str(n):aggregate([r for r in raw if r['nodes']==n]) for n in (50,100)}
    return result,raw


def predict(model,fs):
    model.eval();result=[]
    with torch.inference_mode():
        for f in fs:result.append(model(collate([f],'cuda')).cpu().numpy()[0])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--checkpoints',type=Path,required=True);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists();start,cpu=time.perf_counter(),time.process_time();cfg=read(CODE/'configs/core_teacher_v2.json')
    assert (a.data/'completed.json').exists() and read(a.data.parent/'data_verification.json')['status']=='passed'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    provenance=dict(started_at_utc=stamp(),protocol_sha256=digest(CODE/'configs/core_teacher_v2.json'),data_completed_sha256=digest(a.data/'completed.json'),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in [Path(__file__),CODE/'src/tsp_value/value_gnn.py',CODE/'src/tsp_value/value_model.py',CODE/'scripts/stage2_train.py']})
    if (out/'provenance.json').exists():
        previous=read(out/'provenance.json');assert previous['source_sha256']==provenance['source_sha256'] and previous['protocol_sha256']==provenance['protocol_sha256']
    else:write(out/'provenance.json',provenance)
    train_rows,train_fs=load_role(a.data,'train');val_rows,val_fs=load_role(a.data,'validation')
    assert not {r['instance_id'] for r in train_rows}&{r['instance_id'] for r in val_rows}
    checkpoints=[];training={}
    for kind in cfg['models']:
        for seed in cfg['seeds']:
            name=f'{kind}-seed{seed}';random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            if (out/f'{name}.json').exists():
                info=read(out/f'{name}.json');info.pop('curves',None);assert digest(Path(info['checkpoint']))==info['sha256']
                training[name]=info;checkpoints.append(info);print(name+' reused completed training',flush=True);continue
            model=create_value_model(kind,cfg['architecture']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['learning_rate'],weight_decay=cfg['optimizer']['weight_decay'])
            ckdir=a.checkpoints/name;best=float('inf');best_epoch=bad=step=0;curves=[];resume=None;torch.cuda.reset_peak_memory_stats();model_start=time.perf_counter()
            for epoch in range(1,cfg['epochs']['max']+1):
                batches=[]
                for n in (50,100):
                    order=rng.permutation([i for i,r in enumerate(train_rows) if r['nodes']==n])
                    batches.extend([order[i:i+cfg['batch_size']] for i in range(0,len(order),cfg['batch_size'])])
                rng.shuffle(batches);losses=[];model.train()
                for indices in batches:
                    batch=collate([train_fs[i] for i in indices],'cuda');opt.zero_grad(set_to_none=True)
                    loss,_,_=objective(model(batch),batch['y']);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1;losses.append(float(loss.detach()))
                metric,_=evaluate(val_rows,val_fs,predict(model,val_fs));curve=dict(epoch=epoch,step=step,train_loss=float(np.mean(losses)),validation=metric);curves.append(curve)
                if epoch==1:resume=check_resume(model,opt,collate([f for r,f in zip(train_rows,train_fs) if r['nodes']==50][:8],'cuda'),ckdir/'resume_check.pt',cfg,rng,'cuda',epoch,step)
                if metric['top1_regret_pp']<best-1e-9:
                    best=metric['top1_regret_pp'];best_epoch=epoch;bad=0;save_checkpoint(ckdir/'best.pt',model,opt,epoch,step,cfg,rng)
                else:bad+=1
                save_checkpoint(ckdir/'last.pt',model,opt,epoch,step,cfg,rng)
                print(f'{name} epoch{epoch} val_regret={metric["top1_regret_pp"]:.6f}',flush=True)
                if epoch>=cfg['epochs']['min'] and bad>=cfg['epochs']['patience']:break
            restore(ckdir/'best.pt',model,opt,rng);symmetry={str(n):check_symmetry(model,[f for r,f in zip(train_rows,train_fs) if r['nodes']==n],'cuda') for n in (50,100)}
            info=dict(kind=kind,seed=seed,epochs=len(curves),best_epoch=best_epoch,parameter_count=sum(p.numel() for p in model.parameters()),resume=resume,symmetry=symmetry,
                checkpoint=str(ckdir/'best.pt'),sha256=digest(ckdir/'best.pt'),wall_seconds=time.perf_counter()-model_start,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write(out/f'{name}.json',dict(**info,curves=curves));training[name]=info;checkpoints.append(info);del model,opt;torch.cuda.empty_cache()
    # All checkpoint choices are now final. Heldout targets have not been used in any choice.
    write(out/'all_checkpoints_frozen.json',dict(at_utc=stamp(),checkpoints=checkpoints,heldout_metrics_accessed=False))
    held_rows,held_fs=load_role(a.data,'heldout');results={};predictions=defaultdict(list)
    methods=[('dense-'+kind,kind,seed,a.checkpoints/f'{kind}-seed{seed}/best.pt') for kind in cfg['models'] for seed in cfg['seeds']]
    methods.extend(('stage2-'+kind,kind,seed,a.root/f'checkpoints/stage2/training-v1/{kind}-seed{seed}/best.pt') for kind in ('mlp','transformer') for seed in cfg['seeds'])
    for label,kind,seed,path in methods:
        ck=torch.load(path,map_location='cpu');model=create_value_model(kind,ck['config']['architecture']).cuda().eval();model.load_state_dict(ck['model']);pred=predict(model,held_fs)
        metric,raw=evaluate(held_rows,held_fs,pred);write(out/f'heldout-{label}-seed{seed}.json',dict(metrics=metric,predictions=raw));results[f'{label}-seed{seed}']=metric;predictions[label].append(pred);del model;torch.cuda.empty_cache()
    ensemble={}
    for label,runs in predictions.items():
        pred=[np.mean(np.stack([run[i] for run in runs]),axis=0) for i in range(len(held_fs))];metric,raw=evaluate(held_rows,held_fs,pred)
        ensemble[label]=metric;write(out/f'heldout-{label}-ensemble.json',dict(metrics=metric,predictions=raw))
    metric,_=evaluate(held_rows,held_fs,[np.r_[np.zeros(len(f['edges'])-1),1] for f in held_fs]);ensemble['reference']=metric
    summary=dict(completed_at_utc=stamp(),training=training,heldout_methods=results,heldout_ensembles=ensemble,
        original_instances=dict(train=len({r['instance_id'] for r in train_rows}),validation=len({r['instance_id'] for r in val_rows}),heldout=len({r['instance_id'] for r in held_rows})),
        all_checkpoints_frozen_sha256=digest(out/'all_checkpoints_frozen.json'),wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,heldout_used_for_training_or_checkpoint_selection=False,final_opened=False,
        limitation='Small dense-action pilot with fewer original train instances than stage2; changed data coverage is not an isolated causal ablation. No post-heldout tuning or favorable seed selection.')
    write(out/'summary.json',summary);print(__import__('json').dumps(ensemble,indent=2),flush=True)


if __name__=='__main__':main()
