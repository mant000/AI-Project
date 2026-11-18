"""All-seed main effects with original-instance uncertainty, conditional on frozen programs."""
from collections import defaultdict
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stage3_common import CODE,read,write,digest,stamp


def main():
    source=CODE/'results/main/discovery-v1';verified=read(CODE/'results/main/verification-v1/summary.json');assert verified['status']=='passed'
    out=CODE/'results/main/analysis-v1';out.mkdir(exist_ok=True)
    dev=read(source/'dev_evaluation.json');groups=defaultdict(list);per_discovery=defaultdict(lambda:defaultdict(list))
    identities=None;runs=[]
    for row in dev['runs']:
        if 'generator' not in row:continue
        gen,arm=row['generator'],row['feedback_arm']
        name=(gen if gen in ('original','sft') else '-'.join(gen.split('-')[:2]))+'/'+arm
        byid=defaultdict(list)
        for r in row['evaluation']['results']:byid[r['instance_id']].append(r['gap_percent'])
        ids=sorted(byid);values=np.asarray([np.mean(byid[i]) for i in ids])
        if identities is None:identities=ids
        assert identities==ids;groups[name].append(values);per_discovery[name][row['discovery_seed']].append(values)
        runs.append(dict(generator=gen,feedback_arm=arm,discovery_seed=row['discovery_seed'],program_sha256=row['program_sha256'],mean_gap_percent=float(values.mean())))
    means={name:np.stack(arrays).mean(0) for name,arrays in groups.items()}
    rng=np.random.default_rng(1821);indices=rng.integers(len(identities),size=(5000,len(identities)))
    conditions=[]
    for name in groups:
        arr=means[name];ci=np.quantile(arr[indices].mean(1),[.025,.975]).tolist()
        conditions.append(dict(condition=name,mean_gap_percent=float(arr.mean()),ci95_low_percent=ci[0],ci95_high_percent=ci[1],
            discovery_mean_gap_percent={str(seed):float(np.stack(rr).mean()) for seed,rr in per_discovery[name].items()},
            optimization_repetitions=1 if name.startswith(('sft/','original/')) else 3,original_instances=len(identities)))
    contrasts=[('sft feedback','sft/B','sft/A'),('terminal-trained feedback','grpo-terminal/B','grpo-terminal/A'),
        ('state-trained feedback','grpo-state/B','grpo-state/A'),('terminal parameter learning','grpo-terminal/A','sft/A'),
        ('auxiliary reward only','grpo-state/A','grpo-terminal/A'),('auxiliary reward with prompt','grpo-state/B','grpo-terminal/B'),
        ('combined primary D-C','grpo-state/B','grpo-terminal/A'),('SFT versus original','sft/A','original/A')]
    effects=[]
    for label,after,before in contrasts:
        delta=means[after]-means[before];ci=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
        paired={str(seed):float(np.stack(per_discovery[after][seed]).mean()-np.stack(per_discovery[before][seed]).mean()) for seed in (801,802,803)}
        effects.append(dict(contrast=label,after=after,before=before,mean_gap_change_pp=float(delta.mean()),
            ci95_low_pp=ci[0],ci95_high_pp=ci[1],paired_discovery_effects_pp=paired))
    for name,records in [('runs',runs),('effects',[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in effects])]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    baseline=next(r for r in dev['runs'] if r.get('label')=='gls')['evaluation']['mean_gap_percent']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(9.5,3.2),layout='constrained');x=np.arange(len(conditions))
    y=np.array([r['mean_gap_percent'] for r in conditions]);err=np.array([[r['mean_gap_percent']-r['ci95_low_percent'] for r in conditions],[r['ci95_high_percent']-r['mean_gap_percent'] for r in conditions]])
    ax[0].errorbar(x,y,yerr=err,fmt='o',capsize=3,color='#2a6fbb');ax[0].axhline(baseline,color='gray',ls='--',lw=.8,label='Conventional GLS')
    ax[0].set_xticks(x,[r['condition'].replace('grpo-','').replace('/','\n') for r in conditions],rotation=25)
    ax[0].set_ylabel('Independent dev gap-to-reference (%)');ax[0].legend(frameon=False,fontsize=8)
    for j,r in enumerate(effects[:3]):
        vals=list(r['paired_discovery_effects_pp'].values());ax[1].scatter([j-.12,j,j+.12],vals,color='#c46b2e',s=25)
    ax[1].axhline(0,color='black',lw=.8);ax[1].set_xticks(range(3),['SFT B-A','Terminal-trained\nB-A','State-trained\nB-A'])
    ax[1].set_ylabel('Paired discovery gap change (pp)');ax[1].set_title('All three discovery seeds retained')
    for a in ax:a.grid(axis='y',alpha=.15)
    for suffix in ('png','pdf','svg'):fig.savefig(out/f'main_dev.{suffix}',dpi=180)
    plt.close(fig)
    counts=defaultdict(lambda:dict(outputs=0,valid=0))
    for path in (source/'candidates').glob('*.json'):
        c=read(path);counts[c['generator']]['outputs']+=1;counts[c['generator']]['valid']+=c['valid']
    summary=dict(at_utc=stamp(),conditions=conditions,effects=effects,conventional_gls_gap_percent=baseline,generator_validity=dict(counts),
        original_instances=len(identities),solver_seeds=[0,1,2],unique_frozen_programs=len({r['program_sha256'] for r in runs}),
        uncertainty='5000 original-instance bootstrap conditional on these frozen programs; discovery variability separately shown. Shared frozen initialization is not three training repetitions.',
        selection='all optimization/discovery seeds retained, no dev-based program or seed choice',
        source_verification_sha256=digest(CODE/'results/main/verification-v1/summary.json'),final_opened=False)
    write(out/'summary.json',summary);print(json.dumps(dict(conditions=conditions,effects=effects),indent=2))


if __name__=='__main__':main()
