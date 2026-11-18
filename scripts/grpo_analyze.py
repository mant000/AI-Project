"""Aggregate all predeclared training seeds and costs; no checkpoint or policy selection."""
from collections import Counter
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
from stage3_common import CODE,read,write,digest,stamp


def main():
    root=CODE/'results/training/grpo-v1';verification=read(root/'verification.json');assert verification['status']=='passed'
    cfg=read(CODE/'configs/grpo_v1.json');out=CODE/'results/training/grpo-analysis-v1';out.mkdir(exist_ok=True)
    runs=[];groups=[];paired=[];all_candidates=[]
    for seed in cfg['optimization_seeds']:
        per_arm={}
        for arm in cfg['arms']:
            folder=root/f'{arm}-seed{seed}';meta=read(folder/'completed.json');candidates=[read(p) for p in sorted((folder/'candidates').glob('*.json'))]
            all_candidates.extend(candidates);valid=[r for r in candidates if r['valid']]
            row={k:meta[k] for k in ['arm','seed','candidates','valid_candidates','unique_programs','logical_updates',
                'physical_optimizer_updates_including_restore','zero_variance_groups','input_tokens','output_tokens',
                'generation_wall_seconds','real_endpoint_routes','isolated_worker_cpu_seconds','last_attempt_wall_seconds',
                'last_attempt_cpu_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes']}
            row.update(mean_valid_training_improvement_gap_pp=float(np.mean([r['measured_improvement_gap_pp'] for r in valid])),
                same_routes_as_parent=sum(r['same_endpoint_routes_as_parent'] for r in valid),
                warning='training-pool descriptive diagnostics, not independent route evaluation')
            runs.append(row);per_arm[arm]=candidates
            for i in range(cfg['groups_per_run']):
                g=read(folder/f'group{i:02d}.json')
                groups.append(dict(arm=arm,seed=seed,index=i,auxiliary_bound=g['auxiliary_bound'],reward_std=g['reward_std'],
                    terminal_rewards=[r['terminal_reward'] for r in candidates if r['group_index']==i],
                    total_rewards=g['rewards'],skipped=g['skipped_zero_variance']))
        same=0;valid_pairs=0;delta=[]
        for a,b in zip(per_arm['terminal'],per_arm['state']):
            same+=a.get('program_sha256')==b.get('program_sha256') and a['valid'] and b['valid']
            if a['valid'] and b['valid']:
                valid_pairs+=1;delta.append(b['mean_discovery_gap_percent']-a['mean_discovery_gap_percent'])
        terminal_path=read(root/f'terminal-seed{seed}/adapter_frozen.json')['adapter_path']
        state_path=read(root/f'state-seed{seed}/adapter_frozen.json')['adapter_path']
        ta=load_file(str(__import__('pathlib').Path(terminal_path)/'adapter_model.safetensors'))
        tb=load_file(str(__import__('pathlib').Path(state_path)/'adapter_model.safetensors'))
        parameter_delta=max(float((ta[k]-tb[k]).abs().max()) for k in ta)
        paired.append(dict(seed=seed,same_valid_programs=same,both_valid_pairs=valid_pairs,
            max_abs_policy_parameter_delta_state_vs_terminal=parameter_delta,policy_tensors_identical=parameter_delta==0.,
            state_minus_terminal_mean_training_gap_pp=float(np.mean(delta)) if delta else None))
    with (out/'runs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=runs[0].keys());writer.writeheader();writer.writerows(runs)
    colors={'terminal':'#2a6fbb','state':'#c46b2e'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(9.6,2.7),layout='constrained')
    x=np.arange(len(cfg['optimization_seeds']))
    for arm,offset in [('terminal',-.18),('state',.18)]:
        rr=[r for r in runs if r['arm']==arm]
        axes[0].bar(x+offset,[r['valid_candidates']/r['candidates'] for r in rr],width=.34,label=arm,color=colors[arm])
        axes[1].bar(x+offset,[r['logical_updates'] for r in rr],width=.34,label=arm,color=colors[arm])
        axes[2].bar(x+offset,[r['last_attempt_wall_seconds'] for r in rr],width=.34,label=arm,color=colors[arm])
    for ax in axes:ax.set_xticks(x,cfg['optimization_seeds']);ax.set_xlabel('Predeclared optimization seed');ax.grid(axis='y',alpha=.15)
    axes[0].set(ylabel='Valid output fraction',ylim=(0,1.05));axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(ylabel='Actual optimizer updates',ylim=(0,13));axes[2].set(ylabel='Training command wall seconds')
    for ext in ('png','pdf','svg'):fig.savefig(out/f'grpo_diagnostics.{ext}',dpi=180)
    plt.close(fig)
    summary=dict(at_utc=stamp(),runs=runs,paired_training_diagnostics=paired,
        totals=dict(outputs=sum(r['candidates'] for r in runs),valid_outputs=sum(r['valid_candidates'] for r in runs),
            logical_updates=sum(r['logical_updates'] for r in runs),physical_updates=sum(r['physical_optimizer_updates_including_restore'] for r in runs),
            input_tokens=sum(r['input_tokens'] for r in runs),output_tokens=sum(r['output_tokens'] for r in runs),
            training_commands_wall_seconds=sum(r['last_attempt_wall_seconds'] for r in runs),
            training_commands_parent_cpu_seconds=sum(r['last_attempt_cpu_seconds'] for r in runs),
            isolated_endpoint_worker_cpu_seconds=sum(r['isolated_worker_cpu_seconds'] for r in runs),
            real_endpoint_routes=sum(r['real_endpoint_routes'] for r in runs)),
        measured_branch_reference=read(root/'branch_diagnostic_summary.json'),
        cost_scope='Parent command and endpoint worker compute reported separately; worker startup, monitoring and decision overhead are not all represented by endpoint CPU subtotal. Raw worker/decision/generation timing records retained. Main discovery excluded.',
        interpretation='All six final adapters retained; training diagnostics cannot establish independent route-quality gains.',
        final_opened=False,verification_sha256=digest(root/'verification.json'))
    write(out/'summary.json',summary);print(json.dumps(dict(totals=summary['totals'],paired=paired,branch_reference=summary['measured_branch_reference']),indent=2))


if __name__=='__main__':main()
