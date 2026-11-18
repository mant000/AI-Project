"""Paired CPU retraining ablation, followed by newly isolated action-value holdout."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from stage3_common import CODE, read, write, digest, stamp
from core_teacher_train import load_role, evaluate
from core_teacher_data import bank_row, label_state
from core_mechanism import select_slice
from stage2_labels import states_for
from stage2_train import check_symmetry
from tsp_value.history_features import history_features
from tsp_value.value_model import EdgeValue, collate, objective
from tsp_value.solver import clone, select_edge
from tsp_value.data import length, distances, validate_tour


def predict(model, fs):
    model.eval()
    with torch.inference_mode(): return [model(collate([f], 'cpu')).numpy()[0] for f in fs]


def save(path, model, opt, epoch, step, cfg, rng, progress=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch, step=step, config=cfg,
        python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
        shuffle_rng=rng.bit_generator.state, progress=progress), tmp); tmp.replace(path)


def restore(path, model, opt, rng):
    ck = torch.load(path, map_location='cpu'); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
    random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'])
    rng.bit_generator.state = ck['shuffle_rng']; return ck


def resume_check(model, opt, batch, path, cfg, rng, epoch, step):
    assert len(opt.state) > 0
    save(path, model, opt, epoch, step, cfg, rng)
    def update():
        model.train(); opt.zero_grad(set_to_none=True); loss, _, _ = objective(model(batch), batch['y'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); return float(loss.detach())
    a = update(); expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    restore(path, model, opt, rng); b = update()
    error = max(float((expected[k]-v.detach()).abs().max()) for k, v in model.state_dict().items())
    restore(path, model, opt, rng); assert error == 0 and a == b
    return dict(status='passed', max_parameter_error=error, loss_error=abs(a-b),
        populated_optimizer_states=len(opt.state), extra_updates=2, rolled_back=True)


def encode(rows, arm):
    fs = []
    for r in rows:
        f = history_features(r['state'], r['edges'], arm); f['y'] = np.asarray(r['true_scores_pp'], dtype=np.float32); fs.append(f)
    return fs


def train(root, out, cfg):
    data = CODE/cfg['data']; train_rows, _ = load_role(data, 'train'); val_rows, _ = load_role(data, 'validation')
    checkpoints = []
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            name = f'{arm}-seed{seed}'; path = out/f'{name}.json'; ckdir = root/f'checkpoints/history-ablation-v1/{name}'
            if path.exists():
                info = read(path); assert digest(Path(info['checkpoint'])) == info['checkpoint_sha256']; checkpoints.append(info); continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = EdgeValue('transformer', **cfg['architecture']); opt = torch.optim.AdamW(model.parameters(),
                lr=cfg['optimizer']['learning_rate'], weight_decay=cfg['optimizer']['weight_decay'])
            initial_sha256 = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            train_fs, val_fs = encode(train_rows, arm), encode(val_rows, arm)
            start, cpu = time.perf_counter(), time.process_time(); best = float('inf'); bad = step = 0; best_epoch = 0; curves = []
            if (ckdir/'last.pt').exists():
                ck = restore(ckdir/'last.pt', model, opt, rng)
                saved = ck['progress']; curves = saved['curves']; best = saved['best']; best_epoch = saved['best_epoch']; bad = saved['bad']; step = ck['step']
            already_stopped = len(curves) >= cfg['epochs']['min'] and bad >= cfg['epochs']['patience']
            for epoch in ([] if already_stopped else range(len(curves)+1, cfg['epochs']['max']+1)):
                batches = []
                for n in (50, 100):
                    order = rng.permutation([i for i, r in enumerate(train_rows) if r['nodes']==n])
                    batches.extend([order[i:i+cfg['batch_size']] for i in range(0, len(order), cfg['batch_size'])])
                rng.shuffle(batches); losses = []; model.train()
                for indices in batches:
                    batch = collate([train_fs[i] for i in indices], 'cpu'); opt.zero_grad(set_to_none=True)
                    loss, _, _ = objective(model(batch), batch['y']); assert torch.isfinite(loss)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); step += 1; losses.append(float(loss.detach()))
                metric, _ = evaluate(val_rows, val_fs, predict(model, val_fs)); curves.append(dict(epoch=epoch,
                    step=step, train_loss=float(np.mean(losses)), validation=metric))
                if epoch == 1:
                    check = resume_check(model, opt, collate([f for r, f in zip(train_rows, train_fs) if r['nodes']==50][:8], 'cpu'),
                        ckdir/'resume_check.pt', cfg, rng, epoch, step)
                    write(out/f'{name}-resume.json', check)
                if metric['top1_regret_pp'] < best-1e-9:
                    best = metric['top1_regret_pp']; best_epoch = epoch; bad = 0; save(ckdir/'best.pt', model, opt, epoch, step, cfg, rng)
                else: bad += 1
                save(ckdir/'last.pt', model, opt, epoch, step, cfg, rng,
                    dict(curves=curves, best=best, best_epoch=best_epoch, bad=bad))
                write(out/f'{name}-progress.json', dict(curves=curves, best=best, best_epoch=best_epoch, bad=bad))
                print(f'{name} epoch{epoch} val_regret={metric["top1_regret_pp"]:.6f}', flush=True)
                if epoch >= cfg['epochs']['min'] and bad >= cfg['epochs']['patience']: break
            restore(ckdir/'best.pt', model, opt, rng)
            symmetry = {str(n): check_symmetry(model, [f for r, f in zip(train_rows, train_fs) if r['nodes']==n], 'cpu') for n in (50, 100)}
            info = dict(arm=arm, seed=seed, epochs=len(curves), best_epoch=best_epoch, logical_updates=step,
                initial_parameter_sha256=initial_sha256,
                checkpoint=str(ckdir/'best.pt'), checkpoint_sha256=digest(ckdir/'best.pt'),
                resume=read(out/f'{name}-resume.json'), symmetry=symmetry,
                wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu, gpu_used=False)
            write(path, info); checkpoints.append(info)
    frozen = out/'all_checkpoints_frozen.json'
    for seed in cfg['seeds']:
        pair = [r for r in checkpoints if r['seed']==seed]
        assert len(pair)==2 and pair[0]['initial_parameter_sha256']==pair[1]['initial_parameter_sha256']
    if not frozen.exists(): write(frozen, dict(at_utc=stamp(), checkpoints=checkpoints, fresh_heldout_opened=False))
    return checkpoints


def fresh_data(root, out, cfg):
    assert (out/'all_checkpoints_frozen.json').exists()
    data = out/'fresh-heldout'; data.mkdir(exist_ok=True); states = data/'states'; states.mkdir(exist_ok=True)
    if (data/'completed.json').exists(): return [read(p) for p in sorted(states.glob('*.json'))]
    rawpath = root/'data/prepared/stage1/project_instances.jsonl'
    assert digest(rawpath)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    allrows = [json.loads(x) for x in rawpath.read_text().splitlines()]
    setting = cfg['fresh_heldout']; items = select_slice(allrows, setting['pool'], setting['offset_per_size'], setting['instances_per_size'])
    previous = read(CODE/cfg['data']/'selection.json')['roles']
    oldids = {iid for ids in previous.values() for iid in ids}
    assert all(r['project_use']=='value_val' and r['instance_id'] not in oldids for r in items)
    selection = data/'selection.json'
    if not selection.exists(): write(selection, dict(at_utc=stamp(), selected=[{k:r[k] for k in ('instance_id','nodes','project_use')} for r in items],
        frozen_checkpoints_sha256=digest(out/'all_checkpoints_frozen.json'), roles_disjoint_by_fixed_manifest=True, final_opened=False))
    bank = []
    for item in items:
        for event, s in states_for(item, cfg['state_events']):
            row = bank_row(item, event, s, select_edge(clone(s)), {}); row['role'] = 'fresh_history_holdout'; bank.append(row)
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs = {pool.submit(label_state, row, cfg): states/f'state{i:04d}.json' for i, row in enumerate(bank) if not (states/f'state{i:04d}.json').exists()}
        for f in as_completed(jobs): write(jobs[f], f.result())
    results = [read(p) for p in sorted(states.glob('*.json'))]; count = 0; max_error = 0.
    for r in results:
        d = distances(r['state']['coords'])
        for b in r['records']:
            assert not b['reused']; validate_tour(b['best_tour'], r['nodes']); cost = length(b['best_tour'], d)
            max_error = max(max_error, abs(cost-b['best_length'])); assert abs(cost-b['best_length']) < 1e-10; count += 1
    write(data/'completed.json', dict(at_utc=stamp(), original_instances=len(items), states=len(results),
        new_branches=count, routes_recomputed=count, max_route_cost_error=max_error,
        worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in results), final_opened=False))
    return results


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); a = ap.parse_args()
    cfgpath = CODE/'configs/history_ablation_v1.json'; cfg = read(cfgpath); out = CODE/'results/history-ablation-v1'; out.mkdir(parents=True, exist_ok=True)
    lock = (out/'run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'completed.json').exists(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    assert read((CODE/cfg['data']).parent/'data_verification.json')['status']=='passed'
    start, cpu = time.perf_counter(), time.process_time()
    protocol = dict(protocol=cfg, protocol_sha256=digest(cfgpath), source_sha256={str(p.relative_to(CODE)): digest(p) for p in
        [Path(__file__), CODE/'src/tsp_value/history_features.py', CODE/'src/tsp_value/value_model.py', CODE/'scripts/core_teacher_train.py',
         CODE/'scripts/core_teacher_data.py', CODE/'scripts/stage2_labels.py', CODE/'src/tsp_value/solver.py']})
    if (out/'protocol_frozen.json').exists():
        old = read(out/'protocol_frozen.json')
        for k, v in protocol.items(): assert old[k] == v
    else: write(out/'protocol_frozen.json', dict(at_utc=stamp(), **protocol))
    checkpoints = train(a.root, out, cfg); rows = fresh_data(a.root, out, cfg)
    methods = {}; ensembles = {}
    sources = [(arm, arm, [r for r in checkpoints if r['arm']==arm]) for arm in cfg['arms']]
    for label, prefix in [('R2-dense', 'teacher-v2'), ('stage2-frozen', 'stage2/training-v1')]:
        sources.append((label, 'full', [dict(seed=seed, checkpoint=str(a.root/f'checkpoints/{prefix}/transformer-seed{seed}/best.pt')) for seed in cfg['seeds']]))
    for label, arm, paths in sources:
        fs = encode(rows, arm); runs = []
        for ck in paths:
            state = torch.load(ck['checkpoint'], map_location='cpu'); model = EdgeValue('transformer', **cfg['architecture']); model.load_state_dict(state['model'])
            pred = predict(model, fs); metrics, detail = evaluate(rows, fs, pred); runs.append(pred)
            write(out/f'heldout-{label}-seed{ck["seed"]}.json', dict(metrics=metrics, detail=detail)); methods[f'{label}-seed{ck["seed"]}'] = metrics
        pred = [np.stack([run[i] for run in runs]).mean(0) for i in range(len(rows))]
        metric, detail = evaluate(rows, fs, pred); ensembles[label] = metric; write(out/f'heldout-{label}-ensemble.json', dict(metrics=metric, detail=detail))
    metric, _ = evaluate(rows, encode(rows, 'full'), [np.r_[np.zeros(len(r['edges'])-1), 1] for r in rows]); ensembles['reference'] = metric
    result = dict(status='completed', at_utc=stamp(), checkpoints=checkpoints, heldout_ensembles=ensembles,
        wall_seconds=time.perf_counter()-start, process_cpu_seconds=time.process_time()-cpu,
        fresh_data=read(out/'fresh-heldout/completed.json'), gpu_used=False, fresh_heldout_used_to_tune=False,
        teacher_for_main_or_grpo_replaced=False, final_opened=False)
    write(out/'completed.json', result); print(json.dumps(ensembles, indent=2))


if __name__ == '__main__': main()
