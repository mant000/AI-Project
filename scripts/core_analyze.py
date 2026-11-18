"""Regenerate core mechanism and A/B/C evidence from immutable outcomes."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage3_common import read,write,stamp,digest
from tsp_value.programs import canonical_program,INITIAL
from core_mechanism import cycle_hash


def table(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    start=time.perf_counter();out=a.output;out.mkdir(parents=True,exist_ok=True);study=a.base/'discovery-v1';prepare=a.base/'prepare-v1'
    assert (study/'completed.json').exists()
    cfg=read(prepare/'protocol_frozen.json')['protocol'];candidates=[read(p) for p in (study/'candidates').glob('*.json')]
    calls=[read(p) for p in (study/'generations').glob('*.json')];callbyid={r['candidate_id']:r for r in calls}
    evaluations=[read(p) for p in (study/'evaluations').glob('*.json')];frozen=read(study/'frozen_programs.json');dev=read(study/'dev_evaluation.json')
    assert dev['frozen_programs_sha256']==digest(study/'frozen_programs.json')
    costs=[];curves=[]
    initial=canonical_program(INITIAL)['program_sha256'];baseline_eval=next(r for r in evaluations if r['pool']=='discovery' and r['program_sha256']==initial)
    initial_gap=float(np.mean([r['gap_percent'] for r in baseline_eval['results']]))
    for seed in cfg['discovery_seeds']:
        for arm in cfg['arms']:
            group=sorted([r for r in candidates if r['discovery_seed']==seed and r['arm']==arm],key=lambda r:r['index']);gg=[callbyid[r['candidate_id']] for r in group]
            best=initial_gap;curves.append(dict(seed=seed,arm=arm,outputs=0,best_gap_percent=best))
            for r in group:
                if r['valid']:best=min(best,r['mean_discovery_gap_percent'])
                curves.append(dict(seed=seed,arm=arm,outputs=r['index']+1,best_gap_percent=best))
            hashes={r['program_sha256'] for r in group if r['valid']};behaviors=set()
            for evaluation in evaluations:
                if evaluation['pool']=='discovery' and evaluation['program_sha256'] in hashes and evaluation['ok']:
                    behaviors.add(tuple((r['instance_id'],r['seed'],cycle_hash(r['best_tour'])) for r in evaluation['results']))
            costs.append(dict(seed=seed,arm=arm,outputs=len(group),valid=sum(r['valid'] for r in group),invalid=sum(not r['valid'] for r in group),
                unique_valid_programs=len(hashes),distinct_endpoint_route_vectors=len(behaviors),exact_duplicate_outputs=sum(r.get('duplicate_program',False) for r in group),
                same_parent_endpoint_route_outputs=sum(r.get('same_endpoint_routes_as_parent',False) for r in group),
                input_tokens=sum(r['input_tokens'] for r in gg),output_tokens=sum(r['output_tokens'] for r in gg),
                generation_wall_seconds=sum(r['generation_wall_seconds'] for r in gg),best_discovery_gap_percent=best))
    table(out/'discovery_costs.csv',costs);table(out/'discovery_curves.csv',curves)
    quality=[];paired=[]
    gls=next(r for r in dev['runs'] if r['label']=='gls')['evaluation'];basekeys={(r['instance_id'],r['seed']):r for r in gls['results']}
    for method in dev['runs']:
        rr=method['evaluation']['results']
        for row in rr:
            base=basekeys[row['instance_id'],row['seed']]
            paired.append(dict(label=method['label'],arm=method.get('arm','baseline'),seed=method.get('discovery_seed'),nodes=row['nodes'],
                instance_id=row['instance_id'],solver_seed=row['seed'],gap_percent=row['gap_percent'],relative_gain_vs_gls_pp=100*(base['original_length']-row['original_length'])/base['original_length']))
        for n in (50,100,'all'):
            group=[r for r in paired if r['label']==method['label'] and (n=='all' or r['nodes']==n)]
            quality.append(dict(label=method['label'],arm=method.get('arm','baseline'),seed=method.get('discovery_seed'),nodes=n,
                instances=len({r['instance_id'] for r in group}),solver_conditions=len(group),mean_gap_percent=float(np.mean([r['gap_percent'] for r in group])),
                mean_relative_gain_vs_gls_pp=float(np.mean([r['relative_gain_vs_gls_pp'] for r in group]))))
    table(out/'dev_quality.csv',quality);table(out/'dev_paired.csv',paired)
    comparisons=[]
    for seed in cfg['discovery_seeds']:
        for treatment,reference in [('B','A'),('C','A'),('C','B')]:
            for n in (50,100,'all'):
                aa={(r['instance_id'],r['solver_seed']):r for r in paired if r['seed']==seed and r['arm']==reference and (n=='all' or r['nodes']==n)}
                bb={(r['instance_id'],r['solver_seed']):r for r in paired if r['seed']==seed and r['arm']==treatment and (n=='all' or r['nodes']==n)}
                assert aa.keys()==bb.keys();byinstance=defaultdict(list)
                for key in aa:byinstance[key[0]].append(bb[key]['gap_percent']-aa[key]['gap_percent'])
                values=np.array([np.mean(v) for v in byinstance.values()])
                comparisons.append(dict(seed=seed,comparison=treatment+'-'+reference,nodes=n,mean_gap_difference_pp=float(values.mean()),
                    treatment_wins=int(np.sum(values < -1e-9)),treatment_losses=int(np.sum(values > 1e-9)),ties=int(np.sum(np.abs(values)<=1e-9))))
    table(out/'paired_comparisons.csv',comparisons)
    aggregate=[]
    for pair in ('B-A','C-A','C-B'):
        group=[r for r in comparisons if r['comparison']==pair and r['nodes']=='all'];values=[r['mean_gap_difference_pp'] for r in group]
        aggregate.append(dict(comparison=pair,independent_discovery_pairs=len(values),mean_gap_difference_pp=float(np.mean(values)),
            improving_discovery_seeds=int(np.sum(np.asarray(values)<-1e-9)),worsening_discovery_seeds=int(np.sum(np.asarray(values)>1e-9)),
            tied_discovery_seeds=int(np.sum(np.abs(values)<=1e-9)),per_seed_gap_differences_pp=values))
    mechanism=read(a.base/'mechanism-v1/summary.json');table(out/'mechanism_horizons.csv',mechanism['groups'])
    teacher=read(prepare/'teacher_diagnostic.json');bank=[json.loads(x) for x in (prepare/'bank_observables.jsonl').read_text().splitlines()]
    exact={r['bank_id']:r for r in [read(p) for p in (prepare/'exact_states').glob('*.json')]};ranking=[]
    for row in bank:
        y=np.asarray(exact[row['bank_id']]['true_scores_pp']);p=np.asarray(row['model_scores_pp']).mean(0)
        best=int(np.argmax(p));ranking.append(dict(bank_id=row['bank_id'],nodes=row['nodes'],informative=bool(np.ptp(y)>1e-7),
            oracle_benefit_pp=float(y.max()),model_selected_benefit_pp=float(y[best]),model_regret_pp=float(y.max()-y[best])))
    table(out/'teacher_all_edge_ranking.csv',ranking)
    resources=dict(mechanism=mechanism,bank=read(prepare/'resources.json'),discovery=read(study/'process_resources.json'),
        logical_generations=len(calls),input_tokens=sum(r['input_tokens'] for r in calls),output_tokens=sum(r['output_tokens'] for r in calls),
        generation_wall_seconds=sum(r['generation_wall_seconds'] for r in calls),
        endpoint_unique_batches=len(evaluations),endpoint_routes=sum(len(r.get('results',[])) for r in evaluations),
        endpoint_worker_cpu_seconds=sum(r.get('cpu_seconds',0) for r in evaluations),
        endpoint_subprocess_wall_seconds=sum(r.get('subprocess_wall_seconds',0) for r in evaluations),
        exact_branch_cost_attributable_to_C=read(prepare/'resources.json')['worker_cpu_seconds'],
        error_counts=dict(Counter(r['error'].splitlines()[-1] for r in candidates if not r['valid'])),
        by_pool={pool:dict(unique_batches=len([r for r in evaluations if r['pool']==pool]),routes=sum(len(r.get('results',[])) for r in evaluations if r['pool']==pool),worker_cpu_seconds=sum(r.get('cpu_seconds',0) for r in evaluations if r['pool']==pool)) for pool in ('discovery','program_select','program_dev_eval')},
        caveat='First bank summary failed after all rollouts; original parent-process CPU/GPU peak unavailable. Worker CPU and saved rollout counts are retained; zero GPU allocation in resume describes only the resume, not initial inference. Timing for generation is synchronized wall time, not kernel-only time.')
    write(out/'resource_accounting.json',resources)
    summary=dict(at_utc=stamp(),protocol_sha256=read(study/'provenance.json')['protocol_sha256'],discovery=costs,dev_quality=quality,
        paired_comparisons=comparisons,aggregate_comparisons=aggregate,selected_programs=frozen['programs'],
        mechanism_temporal=mechanism['temporal'],measured_branch_ties=teacher['groups'],
        teacher_mean_top1_regret_pp=float(np.mean([r['model_regret_pp'] for r in ranking])),teacher_mean_selected_benefit_pp=float(np.mean([r['model_selected_benefit_pp'] for r in ranking])),
        teacher_mean_oracle_benefit_pp=float(np.mean([r['oracle_benefit_pp'] for r in ranking])),
        distinct_frozen_programs=len({r['selected_program']['program_sha256'] for r in frozen['programs']}),
        final_opened=False,parameter_training_started=False,interpretation='Measured C is a separately costed conditional finite-horizon feedback comparator; four discovery repetitions are preliminary, not a universal verdict on state feedback.')
    write(out/'summary.json',summary)
    plt.rcParams.update({'font.size':9,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    colors={'A':'#2166ac','B':'#b2182b','C':'#1b7837'}
    for seed,ls in zip(cfg['discovery_seeds'],['-','--',':','-.']):
        for arm in cfg['arms']:
            rr=[r for r in curves if r['seed']==seed and r['arm']==arm]
            axes[0].plot([r['outputs'] for r in rr],[r['best_gap_percent'] for r in rr],color=colors[arm],ls=ls,label=f'{arm}{seed}')
    axes[0].set(xlabel='Outputs per discovery run',ylabel='Real discovery incumbent gap (%)',title='Frozen generator: endpoint / predicted / measured')
    axes[0].legend(ncol=3,fontsize=6)
    x=np.arange(len(cfg['discovery_seeds']))
    for j,arm in enumerate(cfg['arms']):
        values=[next(r['mean_gap_percent'] for r in quality if r['arm']==arm and r['seed']==seed and r['nodes']=='all') for seed in cfg['discovery_seeds']]
        axes[1].bar(x+(j-1)*.24,values,.24,label=arm,color=colors[arm])
    axes[1].set(xticks=x,xticklabels=[str(s) for s in cfg['discovery_seeds']],xlabel='Paired discovery seed',ylabel='Independent dev gap (%)',title='Frozen programs, fresh dev instances');axes[1].legend()
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'core_discovery_dev.{ext}',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,3.5))
    for n,color in [(50,'#2166ac'),(100,'#b2182b')]:
        rr=[r for r in mechanism['groups'] if r['nodes']==n];ax.plot([r['sweeps'] for r in rr],[100*(1-r['zero_fraction']) for r in rr],marker='o',label=f'TSP{n}',color=color)
    ax.set(xscale='log',xlabel='Continuation sweeps',ylabel='Nonzero best-length effects (%)',title='Fresh train diagnosis: signal can appear and disappear');ax.legend();fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'mechanism_horizons.{ext}',dpi=160)
    plt.close(fig);write(out/'analysis_resources.json',dict(wall_seconds=time.perf_counter()-start,gpu_used=False))
    print(json.dumps(dict(aggregate_comparisons=aggregate,distinct_frozen_programs=summary['distinct_frozen_programs'],
        logical_outputs=len(calls),routes=resources['endpoint_routes']),indent=2))


if __name__=='__main__':main()
