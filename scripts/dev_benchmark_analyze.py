"""Five-seed independent development controls, with a separately labeled timing pilot."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage3_common import CODE,read,write,digest,stamp
from control_metrics import summarize,csv_write,route_matrix


def main():
    source=CODE/'results/solver-controls/dev-v1';assert read(source/'verification.json')['status']=='passed'
    protocol=read(source/'protocol_frozen.json');selection=read(source/'selection.json');out=source/'analysis';out.mkdir(exist_ok=True)
    items={i:dict(group='joshi/'+('50' if ':tsp50_' in i else '100')) for i in selection['exact']}
    matrix={};raw=[];cost=[];timing=[];profiles=defaultdict(float)
    for kind in ('exact','external'):
        for file in sorted((source/kind).glob('*.json')):
            label=file.stem.removeprefix('gls-');rr=read(file)['results'];matrix[label]=route_matrix(rr,items,[0,1,2,3,4])
            for r in rr:
                raw.append(dict(label=label,instance_id=r['instance_id'],seed=r['seed'],gap_percent=r['gap_percent']))
                if 'profile' in r:
                    for k,v in r['profile'].items():profiles[k]+=v
            cost.append(dict(label=label,kind=kind,original_instances=64,runs=len(rr),mean_gap_percent=float(np.mean(list(matrix[label].values()))),
                mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in rr])),
                sum_host_cpu_seconds=sum(r.get('cpu_seconds',r.get('process_cpu_seconds',r.get('parent_cpu_seconds',0))) for r in rr),
                sum_waited_child_cpu_seconds=sum(r.get('child_cpu_seconds',0) for r in rr)))
    result=summarize(matrix,protocol['aliases'],items)
    for file in sorted((source/'timing').glob('*.json')):
        for r in read(file)['results']:
            for q in r['checkpoints']:timing.append(dict(label=file.stem,instance_id=r['instance_id'],seed=r['seed'],
                deadline_seconds=q['deadline_seconds'],gap_percent=q['gap_percent'],delta_evaluations=q['delta_evaluations'],
                available_seconds=q['available_seconds'],atomic_overshoot_seconds=q['atomic_overshoot_seconds']))
    for name,rows in [('conditions',result['conditions']),('effects',result['effects']),('raw_gaps',raw),('costs',cost),('timing_pilot',timing)]:csv_write(out/f'{name}.csv',rows)
    write(out/'summary.json',dict(at_utc=stamp(),**result,costs=cost,neural_direct_exact_profile_totals=dict(profiles),
        all_64_instances_five_solver_seeds=True,timing_pilot_only_4_instances=True,no_program_selection_from_dev=True,
        verification_sha256=digest(source/'verification.json'),final_opened=False))
    names=['original/A','sft/A','sft/B','grpo-terminal/A','grpo-state/B','EoH-inspired','native_gls','neural-direct','lehd_greedy','lehd_rrc10','lkh1000']
    pooled={r['condition']:r for r in result['conditions'] if r['group']=='all (descriptive pooled)'}
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained');x=np.arange(len(names));ax.bar(x,[pooled[n]['mean_gap_percent'] for n in names],color='#2a6fbb')
    ax.set_xticks(x,[n.replace('grpo-','').replace('/','\n') for n in names],rotation=35,ha='right');ax.set_ylabel('Independent dev gap-to-reference (%)');ax.grid(axis='y',alpha=.2)
    for suffix in ('png','pdf','svg'):fig.savefig(out/f'dev_references.{suffix}',dpi=180)
    plt.close(fig);print(json.dumps(dict(costs=cost,effects=[r for r in result['effects'] if r['group']=='all (descriptive pooled)']),indent=2),flush=True)


if __name__=='__main__':main()
