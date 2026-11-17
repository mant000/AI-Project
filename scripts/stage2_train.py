"""Train real models; retain per-epoch validation and bitwise resume evidence."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import platform
import random
import resource
import sys
import time
import numpy as np
from scipy.stats import spearmanr
import torch

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from tsp_value.value_model import EdgeValue,features,collate,objective
from tsp_value.data import coordinate_hash


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_dataset(path):
    with gzip.open(path,'rt') as f: rows=[json.loads(line) for line in f]
    rows.sort(key=lambda x:(x['instance_id'],x['state_index']))
    return rows,[features(r) for r in rows]


def metrics(rows,ys,predictions):
    result=[]
    for r,y,p in zip(rows,ys,predictions):
        # The reference is a valid ninth action. Ties prefer reference, then index.
        choices=np.flatnonzero(p>=np.max(p)-1e-9)
        choice=len(p)-1 if len(p)-1 in choices else int(choices[0])
        valid=np.abs(y[:,None]-y[None,:])>1e-7
        pd=p[:,None]-p[None,:]; yd=y[:,None]-y[None,:]
        pair_acc=float(np.mean(np.where(np.abs(pd[valid])<=1e-9,.5,(pd[valid]*yd[valid]>0).astype(float)))) if valid.any() else None
        corr=float(spearmanr(p,y).statistic) if np.ptp(y)>1e-7 and np.ptp(p)>1e-9 else None
        result.append(dict(instance_id=r['instance_id'],nodes=r['nodes'],state_index=r['state_index'],
            informative=bool(np.ptp(y)>1e-7),spearman=corr,non_tied_pair_accuracy=pair_acc,
            chosen_index=choice,selected_benefit_pp=float(y[choice]),oracle_benefit_pp=float(y.max()),
            top1_regret_pp=float(y.max()-y[choice]),top1_optimal=bool(y[choice]>=y.max()-1e-7),
            mse=float(np.mean((p-y)**2)),predictions_pp=np.asarray(p).tolist(),labels_pp=y.tolist()))
    def aggregate(g):
        def mean(key):
            a=[x[key] for x in g if x[key] is not None]; return float(np.mean(a)) if a else None
        info=[x for x in g if x['informative']]
        return dict(states=len(g),instances=len({x['instance_id'] for x in g}),informative_states=len(info),
            spearman=mean('spearman'),non_tied_pair_accuracy=mean('non_tied_pair_accuracy'),
            top1_regret_pp=mean('top1_regret_pp'),selected_benefit_pp=mean('selected_benefit_pp'),
            oracle_benefit_pp=mean('oracle_benefit_pp'),top1_optimal_fraction=mean('top1_optimal'),mse=mean('mse'),
            informative_top1_regret_pp=float(np.mean([x['top1_regret_pp'] for x in info])) if info else None)
    summary=aggregate(result); summary['by_nodes']={str(n):aggregate([x for x in result if x['nodes']==n]) for n in (50,100)}
    # Cluster bootstrap original instances; states and duplicated seeds are never iid units.
    per_instance=defaultdict(list)
    for x in result: per_instance[x['instance_id']].append(x['selected_benefit_pp'])
    means=np.asarray([np.mean(v) for v in per_instance.values()]); rng=np.random.default_rng(700)
    boot=np.mean(means[rng.integers(len(means),size=(2000,len(means)))],axis=1)
    summary['selected_benefit_pp_instance_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
    return summary,result


def inference(model,fs,device,batch_size):
    model.eval(); results=[]
    with torch.no_grad():
        for i in range(0,len(fs),batch_size):
            results.extend(model(collate(fs[i:i+batch_size],device)).cpu().numpy())
    return np.asarray(results)


def save_checkpoint(path,model,opt,epoch,step,cfg,shuffle_rng):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,
        config=cfg,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),shuffle_rng=shuffle_rng.bit_generator.state),path)


def restore(path,model,opt,shuffle_rng):
    ck=torch.load(path,map_location='cpu'); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
    random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'])
    torch.cuda.set_rng_state_all(ck['cuda_rng']); shuffle_rng.bit_generator.state=ck['shuffle_rng']
    return ck


def check_resume(model,opt,batch,path,cfg,shuffle_rng,device,epoch,step_count):
    # A real AdamW/dropout step from the same checkpoint must match uninterrupted execution.
    assert len(opt.state)>0, 'Exercise populated Adam moments, not just untrained weights'
    save_checkpoint(path,model,opt,epoch,step_count,cfg,shuffle_rng)
    def step(m,o):
        m.train(); o.zero_grad(set_to_none=True); loss,_,_=objective(m(batch),batch['y']); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.); o.step(); return float(loss.detach())
    first=step(model,opt)
    expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    restore(path,model,opt,shuffle_rng)
    second=step(model,opt)
    maximum=max(float((expected[k]-v.detach().cpu()).abs().max()) for k,v in model.state_dict().items())
    # Restore initial state so this check does not add a hidden training step.
    restore(path,model,opt,shuffle_rng)
    assert maximum==0 and first==second
    return dict(uninterrupted_loss=first,resumed_loss=second,max_parameter_abs_error=maximum,
        optimizer_and_rng_restored=True,optimizer_state_entries=len(opt.state),after_training_epoch=epoch,
        checkpoint_sha256=digest(path))


def check_symmetry(model,fs,device):
    model.eval(); f=fs[0]; n=len(f['node']); perm=np.random.default_rng(33).permutation(n); inv=np.argsort(perm)
    other={**f,'node':f['node'][perm], 'pair':f['pair'][perm][:,perm], 'edges':inv[f['edges']]}
    swap={**f,'edges':f['edges'][:,::-1].copy()}
    a=inference(model,[f],device,1); b=inference(model,[other],device,1); c=inference(model,[swap],device,1)
    err=float(np.max(np.abs(a-b))); sym=float(np.max(np.abs(a-c)))
    assert err<1e-4 and sym<1e-6
    return dict(permutation_max_abs_error_pp=err,undirected_edge_max_abs_error_pp=sym,tolerance_pp=1e-4)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',type=Path,required=True); ap.add_argument('--val',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,required=True)
    args=ap.parse_args(); cfg=json.loads(args.config.read_text()); out=args.output; out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); cpu=time.process_time(); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True)
    device=torch.device('cuda'); assert torch.cuda.is_available()
    train_rows,train_fs=load_dataset(args.train); val_rows,val_fs=load_dataset(args.val)
    ids=[{r['instance_id'] for r in rows} for rows in (train_rows,val_rows)]
    coords=[{coordinate_hash(r['state']['coords']) for r in rows} for rows in (train_rows,val_rows)]
    assert not ids[0]&ids[1] and not coords[0]&coords[1]
    assert all(r['project_use']=='value_train' for r in train_rows)
    assert all(r['project_use']=='value_val' for r in val_rows)
    assert all(len([r for r in rows if r['instance_id']==i])==4 for rows,ss in ((train_rows,ids[0]),(val_rows,ids[1])) for i in ss)
    audit=dict(train_instances=len(ids[0]),val_instances=len(ids[1]),train_states=len(train_rows),val_states=len(val_rows),
        instance_id_overlap=sorted(ids[0]&ids[1]),coordinate_overlap=sorted(coords[0]&coords[1]),
        split_unit='original instance; all four states and both common seeds inherit pool',
        train_dataset_sha256=digest(args.train),val_dataset_sha256=digest(args.val))
    (out/'isolation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    provenance=dict(config=cfg,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),config_sha256=digest(args.config),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in (Path(__file__),CODE/'src/tsp_value/value_model.py')})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    ys=np.stack([f['y'] for f in val_fs]); summary={}
    for name,pred in [('gls',np.stack([f['utility'] for f in val_fs])),('reference',np.tile(np.r_[np.zeros(8),1],(len(val_fs),1))),
                      ('random_expected',None)]:
        if pred is None:
            # Expected uniformly sampled candidate including reference; exact, no Monte Carlo noise.
            summary[name]=dict(selected_benefit_pp=float(ys.mean()),top1_regret_pp=float(np.mean(ys.max(1)-ys.mean(1))))
        else:
            metric,raw=metrics(val_rows,ys,pred); summary[name]=metric
            (out/f'{name}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
    for kind in cfg['models']:
        for seed in cfg['training_seeds']:
            name=f'{kind}-seed{seed}'; ckdir=args.checkpoints/name
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            shuffle_rng=np.random.default_rng(seed)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model=EdgeValue(kind,**cfg['architecture']).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            resume=None
            model_start=time.perf_counter(); best=float('inf'); best_epoch=0; bad=0; step=0; curves=[]
            with (out/f'{name}_epochs.jsonl').open('x') as log:
                for epoch in range(1,cfg['max_epochs']+1):
                    torch.cuda.synchronize(); epoch_start=time.perf_counter(); model.train(); sums=np.zeros(3); steps=[]
                    order=shuffle_rng.permutation(len(train_fs))
                    for i in range(0,len(order),cfg['batch_size']):
                        batch=collate([train_fs[j] for j in order[i:i+cfg['batch_size']]],device)
                        t=time.perf_counter(); opt.zero_grad(set_to_none=True)
                        loss,mse,rank=objective(model(batch),batch['y']); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); torch.cuda.synchronize()
                        steps.append(time.perf_counter()-t); sums+=np.array([loss.item(),mse.item(),rank.item()]); step+=1
                    pred=inference(model,val_fs,device,cfg['batch_size']); vm,_=metrics(val_rows,ys,pred)
                    record=dict(epoch=epoch,step=step,train_loss=float(sums[0]/len(steps)),
                        train_mse=float(sums[1]/len(steps)),train_rank_loss=float(sums[2]/len(steps)),validation=vm,
                        epoch_seconds=time.perf_counter()-epoch_start,step_seconds_p50_p95=np.quantile(steps,[.5,.95]).tolist())
                    log.write(json.dumps(record)+'\n'); log.flush(); curves.append(record)
                    if epoch==1:
                        resume=check_resume(model,opt,collate(train_fs[:cfg['batch_size']],device),ckdir/'resume_check.pt',cfg,shuffle_rng,device,epoch,step)
                    # Fixed validation regret criterion; no architecture/horizon search on validation.
                    if vm['top1_regret_pp'] < best-1e-9:
                        best=vm['top1_regret_pp']; best_epoch=epoch; bad=0
                        save_checkpoint(ckdir/'best.pt',model,opt,epoch,step,cfg,shuffle_rng)
                    else: bad+=1
                    save_checkpoint(ckdir/'last.pt',model,opt,epoch,step,cfg,shuffle_rng)
                    print(f"{name} epoch {epoch}: train={record['train_loss']:.5f}, val regret={vm['top1_regret_pp']:.6f} pp",flush=True)
                    if epoch>=cfg['min_epochs'] and bad>=cfg['patience']: break
            elapsed=time.perf_counter()-model_start
            restore(ckdir/'best.pt',model,opt,shuffle_rng)
            pred=inference(model,val_fs,device,cfg['batch_size']); vm,raw=metrics(val_rows,ys,pred)
            (out/f'{name}_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
            train_pred=inference(model,train_fs,device,cfg['batch_size']); tm,_=metrics(train_rows,np.stack([f['y'] for f in train_fs]),train_pred)
            info=dict(validation=vm,train=tm,best_epoch=best_epoch,epochs=len(curves),steps=step,
                parameter_count=sum(p.numel() for p in model.parameters()),training_validation_wall_seconds=elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                resume_check=resume,symmetry=check_symmetry(model,val_fs,device),checkpoint=str(ckdir/'best.pt'),
                checkpoint_sha256=digest(ckdir/'best.pt'))
            summary[name]=info; (out/f'{name}_summary.json').write_text(json.dumps(info,indent=2)+'\n')
            del model,opt; torch.cuda.empty_cache()
    final=dict(isolation=audit,provenance=provenance,methods=summary,wall_seconds=time.perf_counter()-started,
        process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='One fixed train/validation subset; validation selects checkpoint. Seeds repeat optimization only; final data unopened.')
    (out/'summary.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final,indent=2))


if __name__=='__main__': main()
