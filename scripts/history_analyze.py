"""Standalone uncertainty plot for the fixed history ablation, clustered by original instance."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stage3_common import CODE,read,write,digest,stamp


def main():
    root=CODE/'results/history-ablation-v1';audit=read(root/'verification.json');assert audit['status']=='passed'
    data=read(root/'completed.json');out=root/'analysis';out.mkdir(exist_ok=True)
    names=['full','no_history','R2-dense','stage2-frozen'];rows=[]
    for name in names:
        metric=data['heldout_ensembles'][name];ci=audit['selected_benefit_pp_instance_bootstrap95'][name]
        rows.append(dict(method=name,selected_benefit_pp=metric['selected_benefit_pp'],ci95_low_pp=ci[0],ci95_high_pp=ci[1],
            top1_regret_pp=metric['top1_regret_pp'],pair_accuracy=metric['non_tied_pair_accuracy']))
    with (out/'ensembles.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(6.6,2.8),layout='constrained');x=np.arange(4);means=np.array([r['selected_benefit_pp'] for r in rows])
    error=np.array([[r['selected_benefit_pp']-r['ci95_low_pp'] for r in rows],[r['ci95_high_pp']-r['selected_benefit_pp'] for r in rows]])
    ax.bar(x,means,color=['#2a6fbb','#c46b2e','#7f8b96','#a2abb3'],width=.6)
    ax.errorbar(x,means,yerr=error,fmt='none',color='#263442',capsize=4,lw=1)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,['Paired CPU\nfull history','Paired CPU\nno history','R2 GPU dense\ntransfer','Stage2 frozen\ntransfer'])
    ax.set_ylabel('Selected action benefit (pp)');ax.set_title('Three-seed ensembles; 32 fresh heldout instances')
    ax.grid(axis='y',alpha=.15)
    for extension in ('png','pdf','svg'):fig.savefig(out/f'history_ablation.{extension}',dpi=180)
    plt.close(fig)
    write(out/'summary.json',dict(at_utc=stamp(),source_verification_sha256=digest(root/'verification.json'),
        error_bars='95% original-instance bootstrap, not states or seeds as independent units',final_opened=False))


if __name__=='__main__':main()
