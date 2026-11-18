"""Regenerate teacher and corpus tables/figures; no training or candidate generation."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stage3_common import read,write,stamp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();code=a.root/'code';out=code/'results/r2-analysis';out.mkdir(exist_ok=True)
    teacher=read(code/'results/teacher-v2/training-v1/summary.json');corpus=read(code/'results/corpus-v1/completed.json');pilot=read(code/'results/training/qlora-pilot-v1/summary.json')
    rows=[]
    for name,result in teacher['heldout_ensembles'].items():rows.append(dict(method=name,selected_benefit_pp=result['selected_benefit_pp'],top1_regret_pp=result['top1_regret_pp'],pair_accuracy=result['non_tied_pair_accuracy']))
    with (out/'teacher_holdout.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    counts=[dict(split=split,**v) for split,v in corpus['coverage'].items()]
    with (out/'corpus_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    plt.rcParams.update({'font.size':9,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));x=np.arange(len(rows));colors=['#4c78a8']*3+['#f58518']*2+['#777777']
    axes[0].bar(x,[r['selected_benefit_pp'] for r in rows],color=colors);axes[0].axhline(0,color='black',lw=.7)
    axes[0].set(xticks=x,xticklabels=[r['method'] for r in rows],ylabel='Selected action benefit (pp)',title='Fresh teacher holdout: 16 original instances');axes[0].tick_params(axis='x',rotation=40)
    xx=np.arange(3);axes[1].bar(xx-.18,[v['examples'] for v in counts],.36,label='examples');axes[1].bar(xx+.18,[v['distinct_target_behaviors'] for v in counts],.36,label='distinct target behaviors')
    axes[1].set(xticks=xx,xticklabels=[v['split'] for v in counts],ylabel='Count',title='Executed grammar corpus; family/target isolation');axes[1].legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'teacher_and_corpus.{ext}',dpi=160)
    plt.close(fig)
    write(out/'summary.json',dict(at_utc=stamp(),teacher_holdout=rows,corpus=corpus['coverage'],
        all_nine_resume_checks_exact=all(v['resume']['max_parameter_abs_error']==0 for v in teacher['training'].values()),
        corpus_verification=read(code/'results/corpus-v1/verification.json'),qlora_infrastructure_pilot=pilot,
        final_opened=False,research_sft_grpo_not_yet_run=True))
    print('R2 tables/figure regenerated')


if __name__=='__main__':main()
