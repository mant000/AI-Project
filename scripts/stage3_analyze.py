"""Regenerate stage3 tables/figures from immutable raw outcomes; no model calls."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import resource
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage3_common import read,write,stamp,digest,CODE


def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',type=Path,required=True); ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    start,cpu=time.perf_counter(),time.process_time(); out=a.output; out.mkdir(parents=True,exist_ok=True)
    assert read(a.study/'completed.json')['final_opened'] is False
    candidates=sorted([read(p) for p in (a.study/'candidates').glob('*.json')],key=lambda r:(r['discovery_seed'],r['arm'],r['index']))
    generations=[read(p) for p in (a.study/'generations').glob('*.json')]
    assert {r['candidate_id'] for r in candidates}=={r['candidate_id'] for r in generations}
    generation_by_id={r['candidate_id']:r for r in generations}
    frozen=read(a.study/'frozen_programs.json'); dev=read(a.study/'dev_evaluation.json')
    assert dev['frozen_programs_sha256']==digest(a.study/'frozen_programs.json')
    discovery=[]
    for seed,arm in sorted({(r['discovery_seed'],r['arm']) for r in candidates}):
        group=[r for r in candidates if (r['discovery_seed'],r['arm'])==(seed,arm)]
        gg=[generation_by_id[r['candidate_id']] for r in group]
        discovery.append(dict(arm=arm,discovery_seed=seed,outputs=len(group),valid=sum(r['valid'] for r in group),
            invalid=sum(not r['valid'] for r in group),unique_valid=len({r['program_sha256'] for r in group if r['valid']}),
            duplicates=sum(r.get('duplicate_program',False) for r in group),input_tokens=sum(r['input_tokens'] for r in gg),
            output_tokens=sum(r['output_tokens'] for r in gg),generation_wall_seconds=sum(r['generation_wall_seconds'] for r in gg),
            output_cap_hits=sum(r['hit_output_cap'] for r in gg),
            best_discovery_gap_percent=min((r['mean_discovery_gap_percent'] for r in group if r['valid']),default=None)))
    table(out/'discovery_costs.csv',discovery)
    baselines={r['label']:r['evaluation'] for r in dev['runs'] if 'arm' not in r}
    gls=baselines['gls']; gls_by_key={(r['instance_id'],r['seed']):r for r in gls['results']}
    quality=[]; raw=[]
    for result in dev['runs']:
        for row in result['evaluation']['results']:
            base=gls_by_key[row['instance_id'],row['seed']]
            raw.append(dict(label=result['label'],arm=result.get('arm','baseline'),discovery_seed=result.get('discovery_seed'),
                nodes=row['nodes'],instance_id=row['instance_id'],solver_seed=row['seed'],
                gap_percent=row['gap_percent'],relative_length_gain_over_gls_pp=100*(base['original_length']-row['original_length'])/base['original_length']))
        for n in (50,100,'all'):
            rr=[r for r in raw if r['label']==result['label'] and (n=='all' or r['nodes']==n)]
            quality.append(dict(label=result['label'],arm=result.get('arm','baseline'),discovery_seed=result.get('discovery_seed'),nodes=n,
                original_instances=len({r['instance_id'] for r in rr}),solver_runs=len(rr),
                mean_gap_percent=float(np.mean([r['gap_percent'] for r in rr])),
                mean_relative_length_gain_over_gls_pp=float(np.mean([r['relative_length_gain_over_gls_pp'] for r in rr]))))
    table(out/'dev_quality.csv',quality); table(out/'dev_paired_outcomes.csv',raw)
    comparisons=[]
    for seed in sorted({r['discovery_seed'] for r in raw if r['discovery_seed'] is not None}):
        for n in (50,100,'all'):
            aa={(r['instance_id'],r['solver_seed']):r for r in raw if r['arm']=='A' and r['discovery_seed']==seed and (n=='all' or r['nodes']==n)}
            bb={(r['instance_id'],r['solver_seed']):r for r in raw if r['arm']=='B' and r['discovery_seed']==seed and (n=='all' or r['nodes']==n)}
            assert aa.keys()==bb.keys()
            diffs={k:bb[k]['gap_percent']-aa[k]['gap_percent'] for k in aa}
            by_instance=defaultdict(list)
            for (iid,solver),v in diffs.items(): by_instance[iid].append(v)
            per_instance=np.array([np.mean(v) for v in by_instance.values()])
            comparisons.append(dict(discovery_seed=seed,nodes=n,original_instances=len(by_instance),
                mean_B_minus_A_gap_pp=float(per_instance.mean()),B_wins_instances=int(np.sum(per_instance < -1e-9)),
                B_losses_instances=int(np.sum(per_instance > 1e-9)),tied_instances=int(np.sum(np.abs(per_instance)<=1e-9))))
    table(out/'ab_comparison.csv',comparisons)
    evaluations=[read(p) for p in (a.study/'evaluations').glob('*.json')]
    decisions=[read(p) for p in (a.study/'decisions').glob('*.json')]
    resources=dict(prepare=read(a.prepare/'prepare_resources.json'),study_process=read(a.study/'process_resources.json'),
        total_logical_generations=len(generations),input_tokens=sum(r['input_tokens'] for r in generations),
        output_tokens=sum(r['output_tokens'] for r in generations),
        generation_synchronized_wall_seconds=sum(r['generation_wall_seconds'] for r in generations),
        generation_process_cpu_seconds=sum(r['process_cpu_seconds'] for r in generations),
        endpoint_evaluation_unique_batches=len(evaluations),endpoint_successful_routes=sum(len(r['results']) for r in evaluations if r['ok']),
        endpoint_partial_routes_in_failed_batches=sum(len(r.get('results',[])) for r in evaluations if not r['ok']),
        endpoint_failed_batches=sum(not r['ok'] for r in evaluations),
        endpoint_evaluation_worker_cpu_seconds=sum(r.get('cpu_seconds',0) for r in evaluations),
        endpoint_evaluation_subprocess_wall_seconds=sum(r.get('subprocess_wall_seconds',0) for r in evaluations),
        decision_unique_batches=len(decisions),decision_worker_cpu_seconds=sum(r.get('cpu_seconds',0) for r in decisions),
        notes='Process child CPU includes isolated evaluation/decision Python startup and nvidia-smi. Worker CPU excludes startup. Cached identical programs are not rerun or recounted. No claimed GPU-kernel-only time. Stage2 label/training costs remain separate, not free.')
    resources['by_pool']={pool:dict(unique_batches=len([r for r in evaluations if r['pool']==pool]),
        successful_routes=sum(len(r['results']) for r in evaluations if r['pool']==pool and r['ok']),
        partial_routes_in_failed_batches=sum(len(r.get('results',[])) for r in evaluations if r['pool']==pool and not r['ok']),
        worker_cpu_seconds=sum(r.get('cpu_seconds',0) for r in evaluations if r['pool']==pool))
        for pool in ('discovery','program_select','program_dev_eval')}
    write(out/'resource_accounting.json',resources)
    probe=read(a.prepare/'probe_summary.json')
    p=probe['comparisons']; zero=[r for r in p if abs(r['true_pp'])<=1e-7]
    diagnostic=dict(**{k:v for k,v in probe.items() if k!='comparisons'},
        confident_nonzero_prediction_on_true_tie_count=sum(r['confident'] for r in zero),
        true_tie_count=len(zero),prediction_threshold_is_not_calibrated_uncertainty=True,
        probe_is_separate_diagnostic_not_a_direct_feedback_discovery_arm=True)
    bank=[json.loads(x) for x in (a.prepare/'bank.jsonl').read_text().splitlines()]
    cfg=read(a.prepare/'protocol_frozen.json')['protocol']
    diagnostic['bank_states_past_endpoint_budget']={str(n):sum(r['state']['delta_evaluations']>cfg['delta_budget_by_nodes'][str(n)] for r in bank if r['nodes']==n) for n in (50,100)}
    diagnostic['invalid_generation_reasons']=dict(Counter(r['error'].splitlines()[-1] for r in candidates if not r['valid']))
    diagnostic['feedback_scope_mismatch']='Values use conventional one-action continuation on a static conventional bank; generated rules change all later decisions. Full candidate quality must be measured by endpoints.'
    diagnostic['actual_evaluation_concurrency']=1
    diagnostic['configured_evaluation_worker_capacity']=cfg['evaluation_workers']
    verification=a.study.parent/'verification-v1/summary.json'
    if verification.exists():
        resources['independent_verification']=read(verification)
        resources['plotting_and_analysis_previous_command']=read(out/'analysis_resources.json') if (out/'analysis_resources.json').exists() else None
        resources['verification_launch_failure']=dict(log='stage3-verification-v1.log',reason='PowerShell expanded a shell variable in the SSH command before remote argparse; no solver/generation executed',
            observed_orchestration_wall_seconds=1.2252704,cpu_seconds=None)
    write(out/'resource_accounting.json',resources)
    write(out/'feedback_diagnostic.json',diagnostic)
    # Discovery curves include the initial GLS incumbent and all failures/ties as flat steps.
    initial_hash=__import__('tsp_value.programs',fromlist=['canonical_program']).canonical_program('d / (1 + p)')['program_sha256']
    initial=next(r for r in evaluations if r['pool']=='discovery' and r['program_sha256']==initial_hash)
    initial_gap=float(np.mean([r['gap_percent'] for r in initial['results']]))
    curves=[]
    for seed,arm in sorted({(r['discovery_seed'],r['arm']) for r in candidates}):
        best=initial_gap
        curves.append(dict(discovery_seed=seed,arm=arm,outputs=0,best_gap_percent=best))
        for r in [x for x in candidates if x['discovery_seed']==seed and x['arm']==arm]:
            if r['valid']: best=min(best,r['mean_discovery_gap_percent'])
            curves.append(dict(discovery_seed=seed,arm=arm,outputs=r['index']+1,best_gap_percent=best))
    table(out/'discovery_curves.csv',curves)
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for seed in sorted({r['discovery_seed'] for r in candidates}):
        for arm,color in [('A','#2166ac'),('B','#b2182b')]:
            rr=[r for r in curves if r['discovery_seed']==seed and r['arm']==arm]
            axes[0].plot([r['outputs'] for r in rr],[r['best_gap_percent'] for r in rr],label=f'{arm}, {seed}',color=color,ls='-' if seed==401 else '--')
    axes[0].set(xlabel='Generated outputs (failures included)',ylabel='Discovery incumbent gap to reference (%)',title='Frozen 7B discovery')
    axes[0].legend(fontsize=8)
    labels=[]; vals50=[]; vals100=[]
    for run in dev['runs']:
        if run['label']=='nn': continue
        labels.append(run['label']); vals50.append(next(r['mean_gap_percent'] for r in quality if r['label']==run['label'] and r['nodes']==50))
        vals100.append(next(r['mean_gap_percent'] for r in quality if r['label']==run['label'] and r['nodes']==100))
    x=np.arange(len(labels)); axes[1].bar(x-.18,vals50,.36,label='TSP50',color='#74add1');axes[1].bar(x+.18,vals100,.36,label='TSP100',color='#f46d43')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Dev gap to reference (%)',title='Frozen programs: 32 independent dev instances')
    axes[1].tick_params(axis='x',rotation=35); axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf','svg'): fig.savefig(out/f'discovery_and_dev.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.6)); ax.scatter([r['true_pp'] for r in p],[r['predicted_pp'] for r in p],s=22,alpha=.65)
    lo=min([r['true_pp'] for r in p]+[r['predicted_pp'] for r in p]); hi=max([r['true_pp'] for r in p]+[r['predicted_pp'] for r in p])
    ax.plot([lo,hi],[lo,hi],'k--',lw=1); ax.axvline(0,color='grey',lw=.5);ax.axhline(0,color='grey',lw=.5)
    ax.set(xlabel='Real paired branch benefit (pp)',ylabel='Transformer ensemble prediction (pp)',
        title=f'Discovery probe: {len(zero)}/{len(p)} true ties')
    fig.tight_layout()
    for ext in ('png','pdf','svg'): fig.savefig(out/f'feedback_probe.{ext}')
    plt.close(fig)
    summary=dict(generated_at_utc=stamp(),discovery=discovery,dev_quality=quality,paired_AB=comparisons,
        selected_programs=frozen['programs'],feedback_diagnostic=diagnostic,
        initial_discovery_gap_percent=initial_gap,
        no_claim_of_robust_effect=True,independent_discovery_repetitions=len({r['discovery_seed'] for r in candidates}),
        final_opened=False,sft_grpo_started=False,
        conclusion='Only two discovery repetitions; sparse provisional value feedback. Selection uses real routes only. No expansion or success claim from an isolated favorable seed.')
    write(out/'summary.json',summary)
    write(out/'analysis_resources.json',dict(wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_used=False))
    print(json.dumps(dict(discovery=discovery,paired_AB=comparisons),indent=2))


if __name__=='__main__':main()
