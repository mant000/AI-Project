"""Original-instance aggregation for all frozen program aliases, never winner selection."""
from collections import defaultdict
import csv
import numpy as np

CONTRASTS=[('primary state contribution','grpo-state/B','grpo-terminal/A'),
    ('SFT versus original','sft/A','original/A'),('SFT feedback','sft/B','sft/A'),
    ('terminal parameter learning','grpo-terminal/A','sft/A'),('terminal-trained feedback','grpo-terminal/B','grpo-terminal/A'),
    ('state-trained feedback','grpo-state/B','grpo-state/A'),('auxiliary reward only','grpo-state/A','grpo-terminal/A'),
    ('auxiliary reward with prompt','grpo-state/B','grpo-terminal/B')]


def condition(alias):
    parts=alias['label'].split('/')
    if parts[0]=='main':
        generator=parts[1]
        if generator.startswith('grpo-'):generator='-'.join(generator.split('-')[:2])
        return generator+'/'+parts[2]
    if parts[0]=='eoh':return 'EoH-inspired'
    return alias['label']


def summarize(matrix,aliases,items,seed=1821):
    # matrix[method][instance_id] already averages all five solver seeds.
    identities=sorted(items);groups=defaultdict(list);per_seed=defaultdict(lambda:defaultdict(list))
    for alias in aliases:
        key=alias['program_sha256'];name=condition(alias)
        vector=np.array([matrix[key][i] for i in identities]);groups[name].append(vector)
        if alias['label'].startswith('main/'):per_seed[name][alias['label'].split('/')[-1]].append(vector)
    for key in matrix:
        if key not in {a['program_sha256'] for a in aliases}:groups[key].append(np.array([matrix[key][i] for i in identities]))
    means={name:np.stack(arrays).mean(0) for name,arrays in groups.items()};rng=np.random.default_rng(seed)
    summary=[];effects=[]
    for group in ['all (descriptive pooled)',*sorted({r['group'] for r in items.values()})]:
        indices=np.array([j for j,i in enumerate(identities) if group=='all (descriptive pooled)' or items[i]['group']==group])
        bootstrap=rng.integers(len(indices),size=(5000,len(indices)))
        for name,vector in means.items():
            values=vector[indices];low,high=np.quantile(values[bootstrap].mean(1),[.025,.975])
            summary.append(dict(group=group,condition=name,original_instances=len(indices),mean_gap_percent=float(values.mean()),
                ci95_low_percent=float(low),ci95_high_percent=float(high),frozen_program_aliases=len(groups[name])))
        for label,after,before in CONTRASTS:
            delta=(means[after]-means[before])[indices];low,high=np.quantile(delta[bootstrap].mean(1),[.025,.975])
            dseed={s:float((np.stack(per_seed[after][s]).mean(0)-np.stack(per_seed[before][s]).mean(0))[indices].mean()) for s in ('801','802','803')}
            effects.append(dict(group=group,contrast=label,after=after,before=before,original_instances=len(indices),
                mean_gap_change_pp=float(delta.mean()),ci95_low_pp=float(low),ci95_high_pp=float(high),paired_discovery_effects_pp=dseed))
    return dict(conditions=summary,effects=effects,uncertainty='5000 original-instance bootstrap conditional on fixed programs; solver seeds averaged first, all discovery/training seeds retained; small TSPLIB strata are descriptive, not population guarantees')


def csv_write(path,records):
    records=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records]
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)


def route_matrix(records,identities,seeds):
    groups=defaultdict(list)
    for r in records:groups[r['instance_id']].append(r)
    assert set(groups)==set(identities)
    for values in groups.values():assert sorted(r['seed'] for r in values)==seeds
    return {key:float(np.mean([r['gap_percent'] for r in rr])) for key,rr in groups.items()}
