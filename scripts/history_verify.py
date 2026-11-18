"""Independent history-ablation route/label/model audit and instance-level paired summaries."""
import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import time
import numpy as np
import torch
from stage3_common import CODE, read, write, digest, stamp
from history_ablation import encode, predict
from core_teacher_train import load_role, evaluate
from stage3_worker import restore_state
from tsp_value.value_model import EdgeValue
from tsp_value.data import distances, length, validate_tour
from tsp_value.solver import clone, penalize, run, select_edge


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True); a = ap.parse_args()
    out = CODE/'results/history-ablation-v1'; cfg = read(CODE/'configs/history_ablation_v1.json')
    assert read(out/'completed.json')['status']=='completed'; torch.set_num_threads(1)
    start, cpu = time.perf_counter(), time.process_time(); frozen = read(out/'all_checkpoints_frozen.json')
    assert frozen['at_utc'] < read(out/'fresh-heldout/selection.json')['at_utc']
    protocol = read(out/'protocol_frozen.json'); assert protocol['protocol']==cfg
    for name, expected in protocol['source_sha256'].items(): assert digest(CODE/name)==expected, name
    rows = [read(p) for p in sorted((out/'fresh-heldout/states').glob('*.json'))]
    checks = 0; maximum = 0.; replay = []
    for i, r in enumerate(rows):
        s = restore_state(r['state']); reference = select_edge(clone(s)); assert list(reference)==r['edges'][-1]
        d = distances(s['coords']); costs = []
        for branch in r['records']:
            validate_tour(branch['best_tour'],r['nodes']); actual = length(branch['best_tour'],d)
            maximum = max(maximum,abs(actual-branch['best_length'])); assert abs(actual-branch['best_length'])<1e-10
            assert branch['delta_evaluations']==cfg['horizon_by_nodes'][str(r['nodes'])]; costs.append(actual); checks += 1
        y = 100*(np.asarray(costs[-1])-np.asarray(costs))/s['best_length']
        assert np.max(np.abs(y-np.asarray(r['true_scores_pp']))) < 1e-10
        if i in (0, 8, 16, 64, 72, 80):
            for j in (0,len(r['edges'])-1):
                branch=clone(s);branch['rng_state']=np.random.default_rng(71).bit_generator.state
                penalize(branch,r['edges'][j]);run(branch,cfg['horizon_by_nodes'][str(r['nodes'])])
                assert np.array_equal(branch['best_tour'],r['records'][j]['best_tour'])
                assert abs(branch['best_length']-costs[j])<1e-10
                replay.append(dict(bank_id=r['bank_id'],edge=r['edges'][j],exact=True))
    _, vf = load_role(CODE/cfg['data'],'validation'); valrows,_ = load_role(CODE/cfg['data'],'validation')
    model_checks=[]
    for ck in frozen['checkpoints']:
        assert digest(Path(ck['checkpoint']))==ck['checkpoint_sha256']; assert ck['resume']['max_parameter_error']==0
        state=torch.load(ck['checkpoint'],map_location='cpu');model=EdgeValue('transformer',**cfg['architecture']);model.load_state_dict(state['model'])
        fs=encode(valrows,ck['arm']);metric,_=evaluate(valrows,fs,predict(model,fs))
        curves=read(out/f'{ck["arm"]}-seed{ck["seed"]}-progress.json')['curves']
        best=min(curves,key=lambda r:r['validation']['top1_regret_pp'])
        assert best['epoch']==ck['best_epoch'];assert abs(metric['top1_regret_pp']-best['validation']['top1_regret_pp'])<1e-7
        heldfs=encode(rows,ck['arm']);pred=predict(model,heldfs);_,detail=evaluate(rows,heldfs,pred)
        previous=read(out/f'heldout-{ck["arm"]}-seed{ck["seed"]}.json')['detail']
        assert all(np.max(np.abs(np.asarray(x['predictions_pp'])-np.asarray(y['predictions_pp'])))<1e-6 for x,y in zip(detail,previous))
        model_checks.append(dict(arm=ck['arm'],seed=ck['seed'],selected_epoch=ck['best_epoch'],fresh_cpu_reload_exact=True))
    predictions={name:read(out/f'heldout-{name}-ensemble.json')['detail'] for name in ['full','no_history','R2-dense','stage2-frozen']}
    ids=sorted({r['instance_id'] for r in rows}); means={name:np.asarray([np.mean([r['selected_benefit_pp'] for r in rr if r['instance_id']==iid]) for iid in ids]) for name,rr in predictions.items()}
    rng=np.random.default_rng(1801);indices=rng.integers(len(ids),size=(5000,len(ids)))
    intervals={name:np.quantile(arr[indices].mean(1),[.025,.975]).tolist() for name,arr in means.items()}
    delta=means['full']-means['no_history']
    result=dict(status='passed',at_utc=stamp(),original_instances=len(ids),states=len(rows),routes_recomputed=checks,
        labels_recomputed=checks,max_route_cost_error=maximum,independent_replay=replay,model_checks=model_checks,
        selected_benefit_pp_instance_bootstrap95=intervals,
        full_minus_no_history_selected_benefit_pp=float(delta.mean()),
        paired_delta_instance_bootstrap95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
        wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        final_opened=False,teacher_for_main_or_grpo_changed=False,verification_source_sha256=digest(Path(__file__)))
    write(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('model_checks','independent_replay')},indent=2))


if __name__=='__main__':main()
