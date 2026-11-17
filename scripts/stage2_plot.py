"""Generate report figures from raw stage2 measurements; never fabricate curves."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(p): return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--results',type=Path,required=True)
    a=ap.parse_args(); root=a.results; out=root/'figures'; out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':9,'figure.dpi':160,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ('png','pdf','svg'): fig.savefig(out/f'{name}.{ext}')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for n,ax in zip((50,100),axes):
        for exp,style in [('diagnosis-v3','o-'),('diagnosis-v2','s--')]:
            groups=[g for g in load(root/exp/'summary.json')['groups'] if g['nodes']==n]
            x=[g['horizon']/(n*(n-3)/2) for g in groups]
            y=[100*g['informative_state_fraction'] for g in groups]
            ax.plot(x,y,style,label='Uniform candidates' if exp=='diagnosis-v3' else 'High-utility candidates')
        ax.set(xscale='log',xlabel='H (complete-sweep equivalents)',ylabel='States with label variation (%)',title=f'TSP{n}: train-only diagnosis'); ax.legend(fontsize=7)
    fig.tight_layout(); save(fig,'horizon_diagnosis')
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    label_rows=[]
    for split in ('pilot','train','val'):
        with gzip.open(root/split/'states.jsonl.gz','rt') as f:
            rows=[json.loads(l) for l in f]
        vals=np.concatenate([np.asarray([x['relative_benefit'] for x in r['records']]).reshape(2,-1).mean(0)*100 for r in rows])
        nz=vals[np.abs(vals)>1e-10]
        label_rows.append(dict(split=split,instances=len({r['instance_id'] for r in rows}),states=len(rows),
            zero_fraction=float(np.mean(np.abs(vals)<=1e-10)),positive_fraction=float(np.mean(vals>1e-10)),negative_fraction=float(np.mean(vals < -1e-10))))
        axes[1].hist(nz,bins=25,histtype='step',label=split)
    x=np.arange(3); zeros=[100*r['zero_fraction'] for r in label_rows]
    axes[0].bar(x,zeros,color='#6c7ea0'); axes[0].set(xticks=x,xticklabels=['Pilot','Train','Val'],ylim=(0,100),ylabel='Zero candidate labels (%)')
    for i,v in enumerate(zeros): axes[0].text(i,v-8,f'{v:.1f}%',ha='center',color='white')
    axes[1].set(xlabel='Nonzero paired benefit (pp)',ylabel='Candidate count'); axes[1].legend()
    for n in (50,100):
        vals=[json.loads(l)['wall_seconds'] for l in (root/'pilot/branches.jsonl').read_text().splitlines() if json.loads(l)['nodes']==n]
        axes[2].plot(np.sort(vals),np.linspace(0,1,len(vals)),label=f'TSP{n}')
    axes[2].set(xlabel='Pilot branch wall seconds',ylabel='Empirical CDF'); axes[2].legend()
    fig.tight_layout(); save(fig,'labels_and_throughput')
    with (out/'label_quality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(label_rows[0]));w.writeheader();w.writerows(label_rows)
    train=root/'training-v1'; summary=load(train/'summary.json'); methods=summary['methods']
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    for kind,color in [('mlp','#ed8b23'),('transformer','#237ab3')]:
        for seed in (101,202,303):
            p=train/f'{kind}-seed{seed}_epochs.jsonl'
            curves=[json.loads(l) for l in p.read_text().splitlines()]; epochs=[r['epoch'] for r in curves]
            label=kind if seed==101 else None
            axes[0].plot(epochs,[r['train_loss'] for r in curves],color=color,alpha=.7,label=label)
            axes[1].plot(epochs,[r['validation']['mse'] for r in curves],color=color,alpha=.7)
            axes[2].plot(epochs,[r['validation']['top1_regret_pp'] for r in curves],color=color,alpha=.7)
    axes[0].set(xlabel='Epoch',ylabel='Train regression + ranking loss');axes[0].legend()
    axes[1].set(xlabel='Epoch',ylabel='Validation MSE (pp squared)')
    axes[2].set(xlabel='Epoch',ylabel='Validation top-1 regret (pp)')
    fig.tight_layout();save(fig,'training_curves')
    fig,axes=plt.subplots(1,3,figsize=(11,3)); table=[]
    names=['GLS','Uniform','MLP','Graph Transformer']; values={k:[] for k in ('regret','benefit','pair')}
    for label,kind in zip(names,['gls','random_expected','mlp','transformer']):
        gs=[methods[k]['validation'] for k in methods if k.startswith(kind+'-seed')] if kind in ('mlp','transformer') else [methods[kind]]
        for key,field in [('regret','top1_regret_pp'),('benefit','selected_benefit_pp'),('pair','non_tied_pair_accuracy')]:
            vals=[g.get(field) for g in gs]; values[key].append([v for v in vals if v is not None])
        for i,g in enumerate(gs): table.append(dict(method=label,optimization_seed=(101,202,303)[i] if len(gs)==3 else '',**{k:g.get(k) for k in ['top1_regret_pp','selected_benefit_pp','non_tied_pair_accuracy','spearman','mse']}))
    for ax,key,title in zip(axes,('regret','benefit','pair'),('Top-1 regret (pp), lower is better','Real paired benefit (pp)','Non-tied pair accuracy')):
        for i,vals in enumerate(values[key]):
            if vals:
                ax.bar(i,np.mean(vals),color=['#888888','#bbbbbb','#ed8b23','#237ab3'][i]);ax.scatter([i]*len(vals),vals,color='black',s=10,zorder=3)
        ax.axhline(0,color='black',lw=.5);ax.set(xticks=np.arange(4),xticklabels=names,title=title);ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Heldout project validation; dots are optimization seeds, not independent datasets',fontsize=10)
    fig.tight_layout();save(fig,'validation_quality')
    with (out/'validation_quality.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0])); w.writeheader();w.writerows(table)
    print(out)


if __name__=='__main__': main()
