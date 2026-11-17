"""Train-only diagnosis and fixed, instance-isolated paired label production."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback
import numpy as np
import psutil

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'src'))
from tsp_value.solver import create_state, run, clone, select_edge, penalize, paired_action_set


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def states_for(item, events):
    s = create_state(item['coords'], seed=17)
    for target in sorted(events):
        while True:
            run(s, 100000000, stop_at_decision=True)
            if s['penalty_events'] >= target:
                yield target, clone(s)
                break
            penalize(s, select_edge(s))
        # The next target must continue beyond this boundary.


def worker(item, cfg):
    start, cpu = time.perf_counter(), time.process_time()
    states = []; previous_best = None
    for target, s in states_for(item, cfg['state_penalty_events']):
        selector = clone(s)
        ref = select_edge(selector)
        edges = sorted({tuple(sorted((int(a), int(b)))) for a,b in zip(s['tour'], np.roll(s['tour'], -1))} - {ref})
        # Stable source-row seed; candidate sampling never uses future labels.
        seed = int(hashlib.sha256((item['instance_id']+':'+str(target)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        if cfg.get('candidate_sampling','uniform') == 'top_utility':
            candidates = sorted(edges, key=lambda e: (-s['distance'][e[0],e[1]]/(1+s['penalties'][e[0],e[1]]),e))[:cfg['candidates']]
        else:
            candidates = [edges[i] for i in rng.choice(len(edges), cfg['candidates'], replace=False)]
        snapshot = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in s.items()
                    if k not in ('distance','pair_i','pair_j')}
        snapshot['sampled_improved_since_previous'] = None if previous_best is None else bool(s['best_length'] < previous_best-1e-12)
        previous_best = s['best_length']
        for h in cfg['horizons']:
            horizon = int(h) if isinstance(h, int) else (int(h['by_nodes'][str(item['nodes'])]) if 'by_nodes' in h else int(h['sweeps'] * len(s['pair_i'])))
            paired = paired_action_set(s, ref, candidates, cfg['seeds'], horizon, trace=True)
            states.append(dict(instance_id=item['instance_id'], nodes=item['nodes'], project_use=item['project_use'],
                coordinate_sha256=item.get('coordinate_sha256'), state_index=target, horizon=horizon,
                horizon_setting=h, state=snapshot, candidates=candidates, reference_edge=ref, **paired))
    return dict(instance_id=item['instance_id'], states=states, wall_seconds=time.perf_counter()-start,
                cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def summarize(states, resources):
    groups=[]
    for n,h in sorted({(r['nodes'], r['horizon']) for r in states}):
        group=[r for r in states if r['nodes']==n and r['horizon']==h]
        vals=[]; duplicate_labels=[]; duplicate_trajectories=[]; state_ranges=[]; seed_std=[]; seed_sign_disagreement=[]
        for r in group:
            y=np.asarray([x['relative_benefit'] for x in r['records']]).reshape(-1,len(r['candidates']))
            vals.extend(y.mean(0).tolist())
            state_ranges.append(float(np.ptp(np.r_[0,y.mean(0)])))
            duplicate_labels.extend((np.abs(y[0]-y[1])<=1e-12).tolist())
            seed_std.extend(y.std(0).tolist())
            seed_sign_disagreement.extend(((y[0]>1e-12)!=(y[1]>1e-12)).tolist())
            b=r['branch_measurements']; k=len(r['candidates'])+1
            duplicate_trajectories.extend([b[i]['trajectory_signature']==b[k+i]['trajectory_signature'] for i in range(k)])
        vals=np.asarray(vals)
        timing=[b['wall_seconds'] for r in group for b in r['branch_measurements']]
        groups.append(dict(nodes=n,horizon=h,states=len(group),candidate_labels=len(vals),
            zero_label_fraction=float(np.mean(np.abs(vals)<=1e-12)),
            informative_state_fraction=float(np.mean(np.asarray(state_ranges)>1e-12)),
            positive_fraction=float(np.mean(vals>1e-12)),negative_fraction=float(np.mean(vals < -1e-12)),
            mean_benefit=float(vals.mean()),benefit_p05_p50_p95=np.quantile(vals,[.05,.5,.95]).tolist(),
            identical_seed_label_fraction=float(np.mean(duplicate_labels)),
            mean_seed_label_std=float(np.mean(seed_std)),seed_benefit_sign_disagreement_fraction=float(np.mean(seed_sign_disagreement)),
            identical_seed_terminal_trajectory_fraction=float(np.mean(duplicate_trajectories)),
            branch_seconds_p50_p95=np.quantile(timing,[.5,.95]).tolist()))
    by_state=[]
    for t in sorted({r['state_index'] for r in states}):
        g=[r for r in states if r['state_index']==t]
        by_state.append(dict(penalty_events=t,states=len(g), informative_state_fraction=float(np.mean([
            np.ptp(np.r_[0,np.asarray([x['relative_benefit'] for x in r['records']]).reshape(2,-1).mean(0)])>1e-12 for r in g]))))
    return dict(groups=groups,by_state=by_state,**resources)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--reuse',type=Path,help='Reuse nested pilot without executing its branches again')
    args=ap.parse_args(); cfg=json.loads(args.config.read_text())
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    data=args.root/'data/prepared/stage1/project_instances.jsonl'
    assert digest(data)=='43be012d1ec09a869b5071b10def2f90a1d55a1784f5ca819ddbdafb3fbab32f'
    assert cfg['pool'] in ('value_train','value_val')
    assert cfg['mode']!='diagnosis' or cfg['pool']=='value_train'
    assert len(cfg['seeds'])==2 and len(set(cfg['state_penalty_events']))==4
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    counts=Counter(); selected=[]
    for r in rows:
        if r['project_use']==cfg['pool'] and counts[r['nodes']] < cfg['instances_per_size']:
            selected.append(r); counts[r['nodes']]+=1
    assert dict(counts)=={50:cfg['instances_per_size'],100:cfg['instances_per_size']}
    (out/'selection.json').write_text(json.dumps([dict(instance_id=r['instance_id'],nodes=r['nodes'],project_use=r['project_use']) for r in selected],indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    provenance=dict(started_at_utc=datetime.now(timezone.utc).isoformat(), config_sha256=digest(args.config),
        normalized_data_sha256=digest(data),manifest_sha256=digest(CODE/'splits/manifest.csv'),
        source_sha256={str(p.relative_to(CODE)):digest(p) for p in (Path(__file__),CODE/'src/tsp_value/solver.py')},
        python=platform.python_version(), numpy=np.__version__, workers=cfg['workers'],
        threads={k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    reused=[]
    if args.reuse:
        previous=json.loads((args.reuse.parent/'config.json').read_text())
        for key in ('pool','state_penalty_events','candidates','candidate_sampling','seeds','horizons'):
            assert cfg[key]==previous[key], key
        with gzip.open(args.reuse,'rt') as f:
            reused=[json.loads(line) for line in f]
        assert {r['instance_id'] for r in reused}<={r['instance_id'] for r in selected}
        provenance['reused_dataset_sha256']=digest(args.reuse)
    stop=threading.Event(); samples=[]
    def monitor():
        proc=psutil.Process()
        while not stop.is_set():
            procs=[proc]+proc.children(recursive=True)
            rss=cpu=0
            for p in procs:
                try:
                    rss+=p.memory_info().rss; c=p.cpu_times(); cpu+=c.user+c.system
                except psutil.Error: pass
            samples.append(dict(seconds=time.perf_counter()-begin,aggregate_rss_bytes=rss,aggregate_cpu_seconds=cpu))
            stop.wait(.2)
    begin=time.perf_counter(); thread=threading.Thread(target=monitor,daemon=True); thread.start()
    all_states=[]; timing=[]; failures=[]
    with gzip.open(out/'states.jsonl.gz','xt') as dataset, (out/'branches.jsonl').open('x') as branches, (out/'tasks.jsonl').open('x') as tasks:
        for s in reused:
            dataset.write(json.dumps(s)+'\n')
            for b in s['branch_measurements']:
                branches.write(json.dumps(dict(instance_id=s['instance_id'],nodes=s['nodes'],state_index=s['state_index'],horizon=s['horizon'],reused_from=str(args.reuse),**b))+'\n')
            all_states.append(s)
        reused_ids={r['instance_id'] for r in reused}
        with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
            futures={pool.submit(worker,r,cfg):r['instance_id'] for r in selected if r['instance_id'] not in reused_ids}
            for f in as_completed(futures):
                try:
                    result=f.result(); ss=result.pop('states'); timing.append(result)
                    tasks.write(json.dumps(result)+'\n'); tasks.flush()
                    for s in ss:
                        dataset.write(json.dumps(s)+'\n'); dataset.flush()
                        for b in s['branch_measurements']:
                            branches.write(json.dumps(dict(instance_id=s['instance_id'],nodes=s['nodes'],state_index=s['state_index'],horizon=s['horizon'],**b))+'\n')
                        all_states.append(s)
                    branches.flush()
                except Exception:
                    fail=dict(instance_id=futures[f],error=traceback.format_exc()); failures.append(fail)
                    (out/'failures.json').write_text(json.dumps(failures,indent=2))
                print(f'completed {len(timing)}/{len(selected)}, failures={len(failures)}',flush=True)
    stop.set(); thread.join()
    elapsed=time.perf_counter()-begin
    branch_count=sum(s['branches_executed'] for s in all_states)
    reused_branches=sum(s['branches_executed'] for s in reused)
    summary=summarize(all_states,dict(provenance=provenance,instances=len(timing)+len(reused_ids),states=len(all_states),
        branches_executed=branch_count,reference_branches=sum(s['reference_branches_executed'] for s in all_states),
        new_branches_executed=branch_count-reused_branches,reused_branches=reused_branches,
        wall_seconds=elapsed,branches_per_second=(branch_count-reused_branches)/elapsed,worker_cpu_seconds=sum(r['cpu_seconds'] for r in timing),
        max_worker_peak_rss_kib=max((r['peak_rss_kib'] for r in timing),default=0),
        sampled_aggregate_peak_rss_bytes=max(s['aggregate_rss_bytes'] for s in samples),failures=failures,
        gpu_used=False,selection_sha256=digest(out/'selection.json'),state_dataset_sha256=digest(out/'states.jsonl.gz')))
    (out/'resource_samples.json').write_text(json.dumps(samples)+'\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    assert not failures
    assert branch_count==len(selected)*4*(cfg['candidates']+1)*len(cfg['seeds'])*len(cfg['horizons'])


if __name__=='__main__': main()
