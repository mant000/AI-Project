"""Evaluate fixed baselines on predeclared project validation instances only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
import numpy as np
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'src'))
from tsp_value.data import validate_tour, length, distances
from tsp_value.solver import create_state, run, save_state, load_state, clone, select_edge, paired_continuation, RULE_ID


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=CODE/'configs/stage1_baseline.json')
    ap.add_argument('--output',type=Path,default=CODE/'results/stage1')
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    assert cfg['pool']=='value_val' and cfg['final_evaluation'] is False
    args.output.mkdir(parents=True,exist_ok=True)
    raw=args.output/'baseline_runs.jsonl'
    if raw.exists():
        raise FileExistsError('Use another --output for reruns; do not overwrite original measurements')
    data=args.root/'data/prepared/stage1/project_instances.jsonl'
    rows=[json.loads(x) for x in data.read_text().splitlines()]
    counts=Counter()
    selected=[]
    for row in rows:
        n=row['nodes']
        if row['project_use']==cfg['pool'] and counts[n]<cfg['instances_per_size']:
            selected.append(row)
            counts[n]+=1
    assert dict(counts)=={50:cfg['instances_per_size'],100:cfg['instances_per_size']}
    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=CODE,text=True).strip())
    provenance=dict(code_commit=git_sha,code_dirty=dirty, config_sha256=digest(args.config),
                    manifest_sha256=digest(CODE/'splits/manifest.csv'),normalized_data_sha256=digest(data),
                    python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_used=False,
                    threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    started,cpu=time.perf_counter(),time.process_time()
    output=[]
    with raw.open('x') as f:
        for item in selected:
            n=item['nodes']
            for seed in cfg['seeds']:
                for method in cfg['methods']:
                    wall,proc=time.perf_counter(),time.process_time()
                    record=dict(instance_id=item['instance_id'],nodes=n,project_use=item['project_use'],seed=seed,
                                method=method,rule=RULE_ID if method=='gls' else method,
                                reference_length=item['reference_length'],gap_definition='gap-to-reference percent',
                                delta_budget=cfg['delta_budget_by_nodes'][str(n)],error_type=None,
                                code_commit=git_sha,config_sha256=provenance['config_sha256'])
                    try:
                        state=create_state(item['coords'],seed,'two_opt' if method=='two_opt' else 'gls',cfg['gls_alpha'])
                        if method!='nn':
                            run(state,record['delta_budget'])
                        validate_tour(state['best_tour'],n)
                        actual=length(state['best_tour'],distances(item['coords']))
                        assert abs(actual-state['best_length'])<1e-10
                        gap=100*(actual/item['reference_length']-1)
                        assert gap > -1e-7, 'Author reference is longer than our result; investigate conventions'
                        record.update(original_length=actual,gap_percent=gap,
                            delta_evaluations=state['delta_evaluations'],sweeps=state['sweeps'],moves=state['moves'],
                            penalty_events=state['penalty_events'],lambda_=state['lambda_'],
                            scan_cursor=state['cursor'],terminated=state['terminated'],best_tour=state['best_tour'].tolist())
                    except Exception as error:
                        record.update(error_type=type(error).__name__,error=str(error))
                    record.update(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-proc,
                                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    f.write(json.dumps(record)+'\n')
                    f.flush()
                    output.append(record)
            print(f"Finished {item['instance_id']}",flush=True)
    summary=dict(provenance=provenance,config=cfg,instances_by_nodes=dict(counts),
                 run_count=len(output),error_count=sum(r['error_type'] is not None for r in output),
                 elapsed_wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=[])
    for n in (50,100):
        for method in cfg['methods']:
            group=[r for r in output if r['nodes']==n and r['method']==method and r['error_type'] is None]
            summary['groups'].append(dict(nodes=n,method=method,runs=len(group),instances=counts[n],
                mean_gap_percent=float(np.mean([r['gap_percent'] for r in group])),
                median_gap_percent=float(np.median([r['gap_percent'] for r in group])),
                mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in group])),
                sum_cpu_seconds=sum(r['cpu_seconds'] for r in group),
                mean_delta_evaluations=float(np.mean([r['delta_evaluations'] for r in group]))))
    (args.output/'baseline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert summary['error_count']==0, 'Inspect all recorded failures'

    # Validate checkpoint replay and paired branches on real author train rows,
    # with a horizon used only to check engineering, not to choose label H.
    diagnostics=[]
    snapshots=args.root/'checkpoints/stage1'
    for n in (50,100):
        item=next(x for x in rows if x['nodes']==n and x['project_use']=='value_train')
        state=create_state(item['coords'],17)
        run(state,12347)
        path=snapshots/f'tsp{n}-partial.json'
        save_state(state,path)
        resumed=load_state(path)
        run(resumed,6789)
        run(state,6789)
        check_path=snapshots/f'tsp{n}-reference.json'
        save_state(state,check_path)
        resumed_path=snapshots/f'tsp{n}-resumed.json'
        save_state(resumed,resumed_path)
        assert digest(check_path)==digest(resumed_path)
        decision=create_state(item['coords'],17)
        run(decision,10000000,stop_at_decision=True)
        assert decision['needs_penalty']
        save_state(decision,snapshots/f'tsp{n}-decision.json')
        selector=clone(decision)
        edge=select_edge(selector)
        before=digest(snapshots/f'tsp{n}-decision.json')
        begin=time.perf_counter()
        same=paired_continuation(decision,edge,edge,[71,72],50000)
        candidate=next(tuple(sorted(map(int,e))) for e in zip(decision['tour'],np.roll(decision['tour'],-1)) if tuple(sorted(map(int,e)))!=edge)
        different=paired_continuation(decision,edge,candidate,[71,72],50000)
        elapsed=time.perf_counter()-begin
        assert all(x['relative_benefit']==0 for x in same)
        save_state(decision,snapshots/f'tsp{n}-decision-after.json')
        assert before==digest(snapshots/f'tsp{n}-decision-after.json')
        diagnostics.append(dict(instance_id=item['instance_id'],nodes=n,checkpoint_bitwise_replay=True,
                                source_unchanged=True,same_action=same,different_action=different,
                                diagnostic_only_horizon=50000,wall_seconds=elapsed,
                                branches_executed=8,reference_reused_across_candidate_sets=False))
    (args.output/'real_data_correctness.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
