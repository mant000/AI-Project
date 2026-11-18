"""Publication figures from verified numeric records; no hand-entered results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
CODE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',type=Path,default=CODE/'results/milestone-fragment-expression-v1/analysis')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--font-directory',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    font='DejaVu Serif'
    if a.font_directory:
        for name in ('times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf'):
            font_manager.fontManager.addfont(str(a.font_directory/name))
        font=font_manager.FontProperties(fname=a.font_directory/'times.ttf').get_name()
        assert font=='Times New Roman',font
    plt.rcParams.update({'font.family':font,'font.size':12,'axes.titlesize':12,'axes.labelsize':12,
                         'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    mechanism=read(a.analysis/'mechanism_summary.json');study=read(a.analysis/'study_summary.json')
    fig,axes=plt.subplots(1,2,figsize=(6.3,2.1),layout='constrained');labels=['One penalty','Rule fragment'];colors=['#62778b','#126e66']
    vals=[100*mechanism['overall'][k]['nonzero_fraction'] for k in ('single','fragment')]
    axes[0].bar(labels,vals,color=colors,width=.58);axes[0].set_ylabel('Nonzero outcomes (%)');axes[0].set_ylim(0,12)
    for i,v in enumerate(vals):axes[0].text(i,v+.3,f'{v:.2f}%',ha='center')
    rhos=[mechanism['overall'][k]['spearman'] for k in ('single','fragment')]
    axes[1].bar(labels,rhos,color=colors,width=.58);axes[1].set_ylabel('Spearman to persistent');axes[1].set_ylim(0,.36)
    for i,v in enumerate(rhos):axes[1].text(i,v+.008,f'{v:.3f}',ha='center')
    for ext in ('png','pdf','svg'):fig.savefig(a.output/f'mechanism.{ext}',dpi=250)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,1.9),layout='constrained')
    for i,(key,color) in enumerate([('F-A','#126e66'),('F-P','#875323')]):
        obj=study['contrasts'][key];values=obj['per_discovery_seed_gap_pp']
        offsets=np.linspace(-.16,.16,len(values));ax.scatter(np.array(values),i+offsets,c=color,s=28,zorder=3)
        ax.errorbar([obj['mean_gap_pp']],[i],xerr=[[obj['mean_gap_pp']-obj['ci95_low']],[obj['ci95_high']-obj['mean_gap_pp']]],fmt='D',color='black',capsize=4,markersize=5,zorder=4)
    ax.axvline(0,color='#888888',linewidth=1);ax.set_yticks([0,1],['Fragment − endpoint','Fragment − one penalty'])
    ax.set_xlabel('Independent dev gap difference (pp; lower is better)');ax.set_ylim(-.4,1.4)
    for ext in ('png','pdf','svg'):fig.savefig(a.output/f'discovery_effects.{ext}',dpi=250)
    plt.close(fig)
    write(a.output/'figure_provenance.json',dict(font=font,font_size=12,
          analysis_inputs={p.name:digest(p) for p in [a.analysis/'mechanism_summary.json',a.analysis/'study_summary.json']},
          source_sha256=digest(Path(__file__)),note='Colored dots are four discovery repetitions; black diamonds/intervals are crossed paired bootstrap mean/95% interval. No independent-state confidence intervals.'))


if __name__=='__main__':main()
